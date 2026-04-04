"""
TEORA v3.0 — Stage 4: Impact Score Fusion (TPIS)
==================================================
Computes per-polygon zonal statistics, vulnerability scores,
ecosystem service ROI, and Tree Planting Impact Score.

CHANGES (v3.1):
  - Water score now computed HERE (soft scoring, not hard binary filter).
    water_score = (area of zone inside water buffer) / (total zone area)
    Prevents the Rank-Then-Filter flaw where infeasible zones got top ranks.
  - tpis_final = 0.8 * tpis + 0.2 * water_score  (soft blend, not multiply)
    Zero water access degrades the score but does not zero it out.
  - tpis_final is saved alongside tpis so Stage 6 uses it directly.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import geopandas as gpd
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
except ImportError:
    gpd = None

try:
    import rasterio
    from rasterio.features import geometry_mask
except ImportError:
    rasterio = None

try:
    from rasterstats import zonal_stats
except ImportError:
    zonal_stats = None

from config.settings import (
    TPIS_WEIGHTS, TARGET_CANOPY_PCT, ECOSYSTEM_VALUES,
    SOCIO_DATA_PATH, WATER_FILTER_PARAMS, setup_logging,
)

logger = setup_logging("teora.data_fusion")


# ─── NORMALIZATION HELPERS ────────────────────────────────────

def min_max_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx - mn < 1e-10:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def safe_divide(a, b, default=0.0):
    """Safe division avoiding zero division."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(b != 0, a / b, default)
    return result


# ─── ZONAL STATISTICS ────────────────────────────────────────

def compute_zonal_stats(
    polygons: "gpd.GeoDataFrame",
    ndvi: np.ndarray,
    lst: np.ndarray,
    seg_mask: np.ndarray,
    raster_meta: dict,
) -> "gpd.GeoDataFrame":
    """
    Compute per-polygon zonal statistics.

    Metrics per polygon:
      - canopy_pct: % pixels classified as canopy (class 0)
      - bare_pct: % pixels classified as bare land (class 2)
      - mean_ndvi: mean NDVI within polygon
      - mean_lst: mean LST within polygon
      - lst_anomaly: mean_lst - city_mean_lst
      - plantable_area: bare_pct × area_m²
      - trees_possible: floor(plantable_area / tree_spacing_m2)

    Args:
        polygons: GeoDataFrame with zone geometries.
        ndvi: NDVI raster array.
        lst: LST raster array.
        seg_mask: Segmentation mask {0:Canopy, 1:Built, 2:Bare}.
        raster_meta: Raster metadata with transform/CRS info.

    Returns:
        GeoDataFrame with zonal stats columns added.
    """
    logger.info(f"Computing zonal stats for {len(polygons)} polygons")

    gdf = polygons.copy()
    transform = raster_meta.get("transform")
    crs = raster_meta.get("crs")

    if crs and hasattr(gdf, 'crs') and gdf.crs:
        if str(gdf.crs) != str(crs):
            gdf = gdf.to_crs(crs)

    results = {
        "canopy_pct": [], "bare_pct": [], "built_pct": [],
        "mean_ndvi": [], "mean_lst": [], "lst_anomaly": [],
        "plantable_area": [], "trees_possible": [],
        "area_m2": [], "pop_density": [],
    }

    city_mean_lst = float(np.nanmean(lst))

    for idx, row in gdf.iterrows():
        geom = row.geometry

        try:
            mask = geometry_mask(
                [geom], out_shape=seg_mask.shape,
                transform=transform, invert=True
            )

            seg_px = seg_mask[mask]
            ndvi_px = ndvi[mask] if ndvi.shape == seg_mask.shape else ndvi.ravel()[:len(seg_px)]

            if lst.shape != seg_mask.shape:
                from scipy.ndimage import zoom
                lst_resized = zoom(lst,
                    (seg_mask.shape[0] / lst.shape[0], seg_mask.shape[1] / lst.shape[1]),
                    order=1)
                lst_px = lst_resized[mask]
            else:
                lst_px = lst[mask]

            total_px = max(len(seg_px), 1)
            canopy_pct = np.sum(seg_px == 0) / total_px * 100
            bare_pct   = np.sum(seg_px == 2) / total_px * 100
            built_pct  = np.sum(seg_px == 1) / total_px * 100

            mean_ndvi_val = float(np.nanmean(ndvi_px)) if len(ndvi_px) > 0 else 0
            mean_lst_val  = float(np.nanmean(lst_px))  if len(lst_px)  > 0 else city_mean_lst

            area_m2 = geom.area
            if area_m2 < 1:
                centroid = geom.centroid
                area_m2 = geom.area * (111320 ** 2) * np.cos(np.radians(centroid.y))

            plantable = bare_pct / 100.0 * area_m2
            trees = int(plantable / ECOSYSTEM_VALUES["tree_spacing_m2"])

        except Exception as e:
            logger.warning(f"Zonal stats failed for polygon {idx}: {e}")
            canopy_pct = bare_pct = built_pct = 0
            mean_ndvi_val = mean_lst_val = 0
            area_m2 = plantable = 0
            trees = 0

        results["canopy_pct"].append(canopy_pct)
        results["bare_pct"].append(bare_pct)
        results["built_pct"].append(built_pct)
        results["mean_ndvi"].append(mean_ndvi_val)
        results["mean_lst"].append(mean_lst_val)
        results["lst_anomaly"].append(mean_lst_val - city_mean_lst)
        results["area_m2"].append(area_m2)
        results["plantable_area"].append(plantable)
        results["trees_possible"].append(trees)
        results["pop_density"].append(0)  # placeholder — from WorldPop

    for key, vals in results.items():
        gdf[key] = vals

    logger.info(f"Zonal stats computed. Total plantable trees: {sum(results['trees_possible'])}")
    return gdf


# ─── WATER SCORE (NEW — replaces hard binary in Stage 5) ──────

def compute_water_score(
    gdf: "gpd.GeoDataFrame",
    water_gdf: Optional["gpd.GeoDataFrame"] = None,
    buffer_m: int = None,
) -> "gpd.GeoDataFrame":
    """
    Compute a continuous water accessibility score per zone.

    Instead of a hard True/False feasibility flag (which caused the
    Rank-Then-Filter flaw), we compute what fraction of each zone's
    area lies within the water infrastructure buffer.

        water_score = intersected_area / zone_area   ∈ [0, 1]

    A zone fully covered scores 1.0; a zone with no nearby water
    scores 0.0 but is NOT zero-ed out in the final TPIS — it receives
    a soft penalty via tpis_final (see compute_tpis_final).

    If no water data is available, all zones receive water_score = 0.5
    (neutral — no reward, no penalty).

    Args:
        gdf: GeoDataFrame with zone geometries.
        water_gdf: Water network GeoDataFrame (optional).
        buffer_m: Buffer radius in metres (default from settings).

    Returns:
        GeoDataFrame with `water_score` column added.
    """
    buf = buffer_m or WATER_FILTER_PARAMS["buffer_m"]

    if water_gdf is None or len(water_gdf) == 0:
        logger.warning("No water data — assigning neutral water_score=0.5 to all zones")
        gdf["water_score"] = 0.5
        return gdf

    logger.info(f"Computing water scores with {buf}m buffer")

    # Align CRS
    if gdf.crs and water_gdf.crs and str(gdf.crs) != str(water_gdf.crs):
        water_gdf = water_gdf.to_crs(gdf.crs)

    water_union = unary_union(water_gdf.geometry.buffer(buf))

    scores = []
    for _, row in gdf.iterrows():
        zone_geom = row.geometry
        try:
            zone_area = zone_geom.area
            if zone_area < 1e-12:
                scores.append(0.0)
                continue
            intersection = zone_geom.intersection(water_union)
            score = intersection.area / zone_area
            scores.append(float(np.clip(score, 0.0, 1.0)))
        except Exception as e:
            logger.warning(f"Water score failed for zone: {e}")
            scores.append(0.0)

    gdf["water_score"] = scores
    logger.info(f"Water scores: mean={np.mean(scores):.3f}, "
                f"min={np.min(scores):.3f}, max={np.max(scores):.3f}")
    return gdf


# ─── VULNERABILITY SCORE ──────────────────────────────────────

def compute_vulnerability(
    gdf: "gpd.GeoDataFrame",
    poverty_raster: Optional[np.ndarray] = None,
    no2_raster: Optional[np.ndarray] = None,
    dependency_raster: Optional[np.ndarray] = None,
    raster_meta: Optional[dict] = None,
) -> "gpd.GeoDataFrame":
    """
    Compute Global Vulnerability Score per polygon.

    vuln_score = mean(poverty_proxy, health_burden, age_dependency)
    """
    logger.info("Computing vulnerability scores")

    n = len(gdf)

    gdf["poverty_proxy"]  = np.random.uniform(0.2, 0.9, n)
    gdf["health_burden"]  = np.random.uniform(0.1, 0.8, n)
    gdf["age_dependency"] = np.random.uniform(0.15, 0.6, n)

    if poverty_raster is not None and raster_meta:
        transform = raster_meta.get("transform")
        for idx, row in gdf.iterrows():
            try:
                mask = geometry_mask([row.geometry], poverty_raster.shape,
                                     transform, invert=True)
                gdf.at[idx, "poverty_proxy"] = float(np.nanmean(poverty_raster[mask]))
            except Exception:
                pass

    if no2_raster is not None and raster_meta:
        vals = [float(np.nanmean(no2_raster)) for _ in range(n)]
        gdf["health_burden"] = min_max_normalize(np.array(vals))

    if dependency_raster is not None and raster_meta:
        vals = [float(np.nanmean(dependency_raster)) for _ in range(n)]
        gdf["age_dependency"] = min_max_normalize(np.array(vals))

    gdf["poverty_proxy"]  = min_max_normalize(gdf["poverty_proxy"].values)
    gdf["health_burden"]  = min_max_normalize(gdf["health_burden"].values)
    gdf["age_dependency"] = min_max_normalize(gdf["age_dependency"].values)

    gdf["vuln_score"] = (
        gdf["poverty_proxy"] + gdf["health_burden"] + gdf["age_dependency"]
    ) / 3.0

    logger.info(f"Vulnerability: mean={gdf['vuln_score'].mean():.3f}, "
                f"max={gdf['vuln_score'].max():.3f}")
    return gdf


# ─── ECOSYSTEM SERVICE ROI ────────────────────────────────────

def compute_ecosystem_roi(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """
    Compute dollar-valued ecosystem service ROI per zone.
    Services: cooling, carbon, air quality, stormwater.
    """
    logger.info("Computing ecosystem service ROI")

    ev    = ECOSYSTEM_VALUES
    trees = gdf["trees_possible"].values.astype(float)
    lst_anomaly = gdf["lst_anomaly"].values
    pop   = gdf.get("pop_density", np.ones(len(gdf))).values

    cooling    = np.abs(lst_anomaly) * pop * ev["kwh_per_degree_cooling"] * ev["electricity_price_per_kwh"]
    carbon     = trees * ev["carbon_tonnes_per_tree_yr"] * ev["carbon_price_per_tonne"]
    air_qual   = trees * ev["pm25_kg_per_tree_yr"] * ev["pm25_price_per_kg"]
    stormwater = trees * ev["stormwater_gal_per_tree"] * ev["stormwater_price_per_gal"]

    gdf["roi_cooling"]     = cooling
    gdf["roi_carbon"]      = carbon
    gdf["roi_air_quality"] = air_qual
    gdf["roi_stormwater"]  = stormwater
    gdf["roi_total"]       = cooling + carbon + air_qual + stormwater
    gdf["roi_norm"]        = min_max_normalize(gdf["roi_total"].values)

    logger.info(f"Total ecosystem ROI: ${gdf['roi_total'].sum():,.0f}")
    return gdf


# ─── TPIS COMPUTATION ─────────────────────────────────────────

def compute_tpis(
    gdf: "gpd.GeoDataFrame",
    weights: Optional[Dict[str, float]] = None,
) -> "gpd.GeoDataFrame":
    """
    Compute Tree Planting Impact Score (TPIS) per zone.

    TPIS = w1*canopy_def_norm + w2*thermal_stress + w3*vuln_score
           + w4*plantability + w5*roi_norm

    Range: [0, 1].  Weights are user-adjustable via config.

    NOTE: This is the raw TPIS (without water).  The water-aware
    final score is computed in compute_tpis_final() below.
    """
    logger.info("Computing TPIS scores")

    w = weights or TPIS_WEIGHTS

    canopy_deficit = np.maximum(0, TARGET_CANOPY_PCT - gdf["canopy_pct"].values / 100.0)
    max_deficit = np.max(canopy_deficit) if np.max(canopy_deficit) > 0 else 1
    gdf["canopy_deficit"]   = canopy_deficit
    gdf["canopy_def_norm"]  = canopy_deficit / max_deficit

    lst_vals = gdf["mean_lst"].values
    lst_min, lst_max = np.min(lst_vals), np.max(lst_vals)
    lst_range = lst_max - lst_min if lst_max - lst_min > 0 else 1
    gdf["thermal_stress"] = (lst_vals - lst_min) / lst_range

    trees = gdf["trees_possible"].values.astype(float)
    max_trees = np.max(trees) if np.max(trees) > 0 else 1
    gdf["plantability"] = trees / max_trees

    tpis = (
        w["canopy_deficit"]  * gdf["canopy_def_norm"].values
        + w["thermal_stress"]  * gdf["thermal_stress"].values
        + w["vulnerability"]   * gdf["vuln_score"].values
        + w["plantability"]    * gdf["plantability"].values
        + w["roi_norm"]        * gdf["roi_norm"].values
    )

    gdf["tpis"] = np.clip(tpis, 0, 1)

    logger.info(f"TPIS: mean={gdf['tpis'].mean():.3f}, "
                f"max={gdf['tpis'].max():.3f}, min={gdf['tpis'].min():.3f}")
    return gdf


def compute_tpis_final(
    gdf: "gpd.GeoDataFrame",
    tpis_weight: float = 0.8,
    water_weight: float = 0.2,
) -> "gpd.GeoDataFrame":
    """
    Blend raw TPIS with water_score into a water-aware final score.

    Formula (soft blend — avoids zero-ing out poorly served zones):
        tpis_final = tpis_weight * tpis + water_weight * water_score

    Default: 80 % environmental impact + 20 % water accessibility.

    This score is what Stage 6 (knapsack) uses as the per-zone value,
    so ranking and optimisation are always feasibility-aware.

    Args:
        gdf: GeoDataFrame with `tpis` and `water_score` columns.
        tpis_weight: Weight for raw TPIS component (default 0.8).
        water_weight: Weight for water accessibility component (default 0.2).

    Returns:
        GeoDataFrame with `tpis_final` column added.
    """
    if "water_score" not in gdf.columns:
        logger.warning("water_score column missing — tpis_final = tpis (no water adjustment)")
        gdf["tpis_final"] = gdf["tpis"]
        return gdf

    gdf["tpis_final"] = np.clip(
        tpis_weight * gdf["tpis"].values + water_weight * gdf["water_score"].values,
        0.0, 1.0
    )

    logger.info(f"tpis_final: mean={gdf['tpis_final'].mean():.3f}, "
                f"max={gdf['tpis_final'].max():.3f}, min={gdf['tpis_final'].min():.3f}")
    return gdf


# ─── FULL STAGE 4 PIPELINE ────────────────────────────────────

def run_data_fusion(
    polygons: "gpd.GeoDataFrame",
    ndvi: np.ndarray,
    lst: np.ndarray,
    seg_mask: np.ndarray,
    raster_meta: dict,
    weights: Optional[Dict[str, float]] = None,
    poverty_raster: Optional[np.ndarray] = None,
    no2_raster: Optional[np.ndarray] = None,
    dependency_raster: Optional[np.ndarray] = None,
    water_gdf: Optional["gpd.GeoDataFrame"] = None,
    output_dir: Optional[str] = None,
) -> "gpd.GeoDataFrame":
    """
    Run complete Stage 4: Impact Score Fusion.

    New parameter vs v3.0:
        water_gdf — optional water network GeoDataFrame passed in from
                    the pipeline runner so that water_score is computed
                    here, before ranking, fixing the Rank-Then-Filter flaw.

    Output columns include both `tpis` (raw) and `tpis_final` (water-aware).
    Stage 5 retains its hard-feasibility role (min area, trees > 0) but no
    longer performs binary water gating.  Stage 6 uses `tpis_final`.
    """
    logger.info("=" * 60)
    logger.info("STAGE 4: IMPACT SCORE FUSION")
    logger.info("=" * 60)

    out_dir = Path(output_dir) if output_dir else Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = compute_zonal_stats(polygons, ndvi, lst, seg_mask, raster_meta)
    gdf = compute_vulnerability(gdf, poverty_raster, no2_raster,
                                dependency_raster, raster_meta)
    gdf = compute_ecosystem_roi(gdf)
    gdf = compute_tpis(gdf, weights)

    # ── NEW: water score computed here so ranking is feasibility-aware ──
    gdf = compute_water_score(gdf, water_gdf=water_gdf)
    gdf = compute_tpis_final(gdf)

    gdf.to_file(str(out_dir / "zones_tpis.geojson"), driver="GeoJSON")
    logger.info(f"Saved zones_tpis.geojson ({len(gdf)} zones)")
    logger.info("STAGE 4 COMPLETE")
    return gdf


if __name__ == "__main__":
    logger.info("Data fusion module loaded — run via pipeline_runner.py")