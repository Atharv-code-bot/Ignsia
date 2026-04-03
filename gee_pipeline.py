
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

try:
    import ee
except ImportError:
    ee = None

try:
    import geemap
except ImportError:
    geemap = None

try:
    import geopandas as gpd
    from shapely.geometry import shape
except ImportError:
    gpd = None

try:
    import osmnx as ox
except ImportError:
    ox = None

from config.settings import (
    GEE_SERVICE_ACCOUNT_KEY, PROJECT_ID, GEE_DATASETS,
    S2_BAND_NAMES, LANDSAT_THERMAL, WORLDPOP_DEPENDENCY_BANDS,
    WATER_FILTER_PARAMS, DATA_DIR, setup_logging,
)

logger = setup_logging("teora.gee_pipeline")


def initialize_gee(service_account_key=None, project_id=None):
    """Authenticate and initialize Google Earth Engine."""
    if ee is None:
        raise RuntimeError("earthengine-api not installed")
    key = service_account_key or GEE_SERVICE_ACCOUNT_KEY
    proj = project_id or PROJECT_ID
    try:
        if key and key != "<INSERT_GEE_CREDENTIALS>":
            creds = ee.ServiceAccountCredentials(email=None, key_file=key)
            # Try initializing with project first, fallback to just credentials
            try:
                ee.Initialize(credentials=creds, project=proj)
            except ee.EEException:
                ee.Initialize(credentials=creds)
        else:
            ee.Authenticate()
            ee.Initialize(project=proj if proj != "<INSERT_PROJECT_ID>" else None)
        logger.info("GEE initialized successfully")
        return True
    except Exception as e:
        logger.error(f"GEE init failed: {e}")
        raise


def load_aoi(geojson_input: Union[str, dict, Path]) -> "ee.Geometry":
    """Convert GeoJSON input to ee.Geometry."""
    if isinstance(geojson_input, (str, Path)):
        p = Path(geojson_input)
        geojson = json.load(open(p)) if p.exists() else json.loads(str(geojson_input))
    else:
        geojson = geojson_input

    if geojson.get("type") == "FeatureCollection":
        geometry = geojson["features"][0]["geometry"]
    elif geojson.get("type") == "Feature":
        geometry = geojson["geometry"]
    else:
        geometry = geojson

    logger.info(f"AOI loaded: {geometry.get('type')} geometry")
    return ee.Geometry(geometry)


def cloud_mask_s2(image):
    """Mask clouds from Sentinel-2 SR using SCL band."""
    scl = image.select("SCL")
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return image.updateMask(mask).copyProperties(image, ["system:time_start"])


def cloud_mask_landsat(image):
    """Mask clouds from Landsat using QA_PIXEL."""
    qa = image.select("QA_PIXEL")
    return (image.updateMask(qa.bitwiseAnd(1 << 3).eq(0))
            .updateMask(qa.bitwiseAnd(1 << 5).eq(0))
            .copyProperties(image, ["system:time_start"]))


def fetch_sentinel2(aoi, start_date, end_date, max_cloud=20):
    """Fetch Sentinel-2 SR median composite [H×W×13] @ 10m."""
    logger.info(f"Fetching Sentinel-2: {start_date} → {end_date}")
    col = (ee.ImageCollection(GEE_DATASETS["sentinel2"])
           .filterBounds(aoi).filterDate(start_date, end_date)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
           .map(cloud_mask_s2))
    count = col.size().getInfo()
    logger.info(f"Sentinel-2: {count} images")
    if count == 0:
        raise ValueError("No Sentinel-2 images found")
    return col.select(S2_BAND_NAMES).median().clip(aoi)


def fetch_landsat_thermal(aoi, start_date, end_date):
    """Fetch Landsat 8/9 thermal ST_B10 median @ 100m."""
    logger.info(f"Fetching Landsat thermal: {start_date} → {end_date}")
    for key in ["landsat9", "landsat8"]:
        col = (ee.ImageCollection(GEE_DATASETS[key])
               .filterBounds(aoi).filterDate(start_date, end_date)
               .map(cloud_mask_landsat))
        if col.size().getInfo() > 0:
            return col.select([LANDSAT_THERMAL["band"]]).median().clip(aoi)
    raise ValueError("No Landsat thermal images found")


def fetch_worldpop(aoi, year=2020):
    """Fetch WorldPop dependency ratio @ 100m."""
    logger.info(f"Fetching WorldPop for {year}")
    wp = (ee.ImageCollection(GEE_DATASETS["worldpop"])
          .filterDate(f"{year}-01-01", f"{year}-12-31")
          .filterBounds(aoi).first())
    young = wp.select(WORLDPOP_DEPENDENCY_BANDS["young"]).reduce(ee.Reducer.sum())
    elderly = wp.select(WORLDPOP_DEPENDENCY_BANDS["elderly"]).reduce(ee.Reducer.sum())
    total = wp.reduce(ee.Reducer.sum())
    dep = young.add(elderly).divide(total.add(0.001)).rename("dependency_ratio")
    return dep.clip(aoi)


def fetch_sentinel5p_no2(aoi, start_date, end_date):
    """Fetch Sentinel-5P NO2 mean as health burden proxy @ ~1km."""
    logger.info(f"Fetching S5P NO2: {start_date} → {end_date}")
    col = (ee.ImageCollection(GEE_DATASETS["sentinel5p_no2"])
           .filterBounds(aoi).filterDate(start_date, end_date)
           .select(["tropospheric_NO2_column_number_density"]))
    return col.mean().clip(aoi).rename("no2_health_burden")


def fetch_meta_rwi(aoi):
    """Fetch Meta RWI poverty proxy (inverted, normalized)."""
    logger.info("Fetching Meta RWI")
    try:
        fc = ee.FeatureCollection(GEE_DATASETS["meta_rwi"]).filterBounds(aoi)
        rwi = fc.reduceToImage(properties=["rwi"], reducer=ee.Reducer.mean()).rename("rwi")
        stats = rwi.reduceRegion(ee.Reducer.minMax(), aoi, 2400, maxPixels=1e9)
        rmin, rmax = ee.Number(stats.get("rwi_min")), ee.Number(stats.get("rwi_max"))
        normalized = rwi.subtract(rmin).divide(rmax.subtract(rmin).add(0.001))
        return ee.Image(1).subtract(normalized).rename("poverty_proxy").clip(aoi)
    except Exception as e:
        logger.warning(f"Meta RWI fetch failed: {e}. Using flat 0.5 proxy.")
        return ee.Image(0.5).rename("poverty_proxy").clip(aoi)


def fetch_osm_water(aoi_geojson, output_path=None):
    """Fetch water infrastructure from OSM via osmnx."""
    if ox is None:
        logger.warning("osmnx not installed")
        return None
    logger.info("Fetching OSM water network")
    if aoi_geojson.get("type") == "FeatureCollection":
        geom = shape(aoi_geojson["features"][0]["geometry"])
    elif aoi_geojson.get("type") == "Feature":
        geom = shape(aoi_geojson["geometry"])
    else:
        geom = shape(aoi_geojson)
    try:
        water = ox.features_from_polygon(geom, tags=WATER_FILTER_PARAMS["osm_tags"])
        logger.info(f"Fetched {len(water)} water features")
        if output_path:
            water.to_file(str(output_path), driver="GeoJSON")
        return water
    except Exception as e:
        logger.error(f"OSM fetch failed: {e}")
        return None


def export_to_geotiff(image, filename, aoi, scale=10, output_dir=None):
    """Export ee.Image to local GeoTIFF via geemap with scale back-off."""
    if geemap is None:
        raise RuntimeError("geemap not installed")
    out = (output_dir or DATA_DIR) / f"{filename}.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting {filename} @ {scale}m")
    
    current_scale = scale
    for attempt in range(4):
        try:
            geemap.ee_export_image(image, filename=str(out), scale=current_scale, region=aoi, file_per_band=False)
            if out.exists() and out.stat().st_size > 0:
                logger.info(f"Exported: {out} at {current_scale}m")
                return out
        except Exception:
            pass
        logger.warning(f"Export truncated/failed at {current_scale}m. Retrying with larger scale...")
        current_scale = int(current_scale * 1.5)
        
    if not out.exists() or out.stat().st_size == 0:
        raise IOError(f"Failed to cleanly export {filename} after all attempts. (Payload too large)")
        
    return out


def run_acquisition_pipeline(aoi_geojson, start_date, end_date, export_scale=10, output_dir=None):
    """Run complete Stage 1 acquisition pipeline."""
    logger.info("=" * 60)
    logger.info("STAGE 1: DATA ACQUISITION")
    logger.info("=" * 60)
    out = output_dir or DATA_DIR
    results = {}

    initialize_gee()
    aoi = load_aoi(aoi_geojson)

    results["sentinel2"] = fetch_sentinel2(aoi, start_date, end_date)
    results["s2_path"] = export_to_geotiff(results["sentinel2"], "sentinel2_composite", aoi, export_scale, out)

    results["landsat_thermal"] = fetch_landsat_thermal(aoi, start_date, end_date)
    results["thermal_path"] = export_to_geotiff(results["landsat_thermal"], "landsat_thermal", aoi, 100, out)

    results["worldpop"] = fetch_worldpop(aoi)
    results["worldpop_path"] = export_to_geotiff(results["worldpop"], "worldpop_dependency", aoi, 100, out)

    results["no2"] = fetch_sentinel5p_no2(aoi, start_date, end_date)
    results["no2_path"] = export_to_geotiff(results["no2"], "sentinel5p_no2", aoi, 1000, out)

    results["rwi"] = fetch_meta_rwi(aoi)
    results["rwi_path"] = export_to_geotiff(results["rwi"], "meta_rwi_poverty", aoi, 2400, out)

    # OSM Water
    if isinstance(aoi_geojson, (str, Path)):
        p = Path(aoi_geojson)
        aoi_dict = json.load(open(p)) if p.exists() else json.loads(str(aoi_geojson))
    else:
        aoi_dict = aoi_geojson
    water_path = out / "water_network.geojson"
    results["water_gdf"] = fetch_osm_water(aoi_dict, water_path)
    results["water_path"] = water_path

    logger.info("STAGE 1 COMPLETE")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TEORA Stage 1: GEE Acquisition")
    parser.add_argument("--aoi", required=True, help="AOI GeoJSON file")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2023-12-31")
    parser.add_argument("--scale", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    run_acquisition_pipeline(args.aoi, args.start_date, args.end_date, args.scale,
                             Path(args.output_dir) if args.output_dir else None)
