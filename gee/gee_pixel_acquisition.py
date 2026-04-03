import ee
import json
import argparse
import os
import re
import time
import geopandas as gpd
from shapely.geometry import mapping


CONFIG = {
    "gee_project": "metal-lantern-492210-m6",       # ← replace
    "meta_rwi_asset":    "projects/YOUR-GEE-PROJECT/assets/meta_rwi_india",
    "start_date":        "2024-01-01",
    "end_date":          "2024-12-31",
    "max_cloud_pct":     20,
    "export_scale_m":    10,           # 10 m native Sentinel-2
    "lst_scale_m":       30,           # 30 m native Landsat
    "tsharp_scale_m":    30,           # predictor bands downsampled to 30m
    "export_destination": "drive",     # 'drive' or 'gcs'
    "drive_folder":      "TEORA_Pune_PixelWise",
    "gcs_bucket":        "teora-pune-pixels",
    "geojson_path":      "./pune-admin-wards_2017.geojson",
    "local_output_dir":  "./ward_outputs",
}

WARD_NAMES = [
    "Admin Ward 01 Aundh",          "Admin Ward 02 Ghole Road",
    "Admin Ward 03 Kothrud Karveroad","Admin Ward 04 Warje Karvenagar",
    "Admin Ward 05 Dhole Patil Rd",  "Admin Ward 06 Yerawda - Sangamwadi",
    "Admin Ward 07 Nagar Road",      "Admin Ward 08 KasbaVishrambaugwada",
    "Admin Ward 09 Tilak Road",      "Admin Ward 10 Sahakarnagar",
    "Admin Ward 11 Bibwewadi",       "Admin Ward 12 Bhavani Peth",
    "Admin Ward 13 Hadapsar",        "Admin Ward 14 Dhankawadi",
    "Admin Ward 15 Kondhwa Wanavdi",
]



def init_gee(project: str):
    try:
        ee.Initialize(project=project)
    except ee.EEException:
        ee.Authenticate()
        ee.Initialize(project=project)
    print(f"[GEE] ✅ Initialised  project={project}")



def load_wards(geojson_path: str) -> dict:
    """Returns {ward_name: ee.Geometry}"""
    with open(geojson_path) as f:
        gj = json.load(f)
    wards = {}
    for feat in gj["features"]:
        name = feat["properties"]["name"]
        wards[name] = ee.Geometry(feat["geometry"])
    print(f"[WARDS] Loaded {len(wards)} wards")
    return wards


def safe_name(ward_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", ward_name).strip("_")



def mask_s2_clouds(img: ee.Image) -> ee.Image:
    qa  = img.select("QA60")
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return img.updateMask(mask).divide(10000).copyProperties(img, ["system:time_start"])


def build_s2_stack(aoi: ee.Geometry, start: str, end: str, max_cloud: int) -> ee.Image:
    
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
            .map(mask_s2_clouds)
            .median().clip(aoi))

    proj10 = s2.select("B2").projection()
    s2 = s2.resample("bilinear").reproject(crs=proj10, scale=10)

    ndvi  = s2.normalizedDifference(["B8","B4"]).rename("NDVI")
    ndbi  = s2.normalizedDifference(["B11","B8"]).rename("NDBI")
    ndwi  = s2.normalizedDifference(["B3","B8"]).rename("NDWI")
    mndwi = s2.normalizedDifference(["B3","B11"]).rename("MNDWI")
    bsi   = s2.expression(
                "((SWIR+RED)-(NIR+BLUE))/((SWIR+RED)+(NIR+BLUE))",
                {"SWIR":s2.select("B11"),"RED":s2.select("B4"),
                 "NIR":s2.select("B8"),"BLUE":s2.select("B2")}).rename("BSI")
    evi   = s2.expression(
                "2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))",
                {"NIR":s2.select("B8"),"RED":s2.select("B4"),
                 "BLUE":s2.select("B2")}).rename("EVI")

    bands = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]
    return s2.select(bands).addBands([ndvi,ndbi,ndwi,mndwi,bsi,evi])



def scale_landsat(img: ee.Image) -> ee.Image:
    thermal = img.select("ST_B10").multiply(0.00341802).add(149.0)  # → Kelvin
    return img.addBands(thermal, overwrite=True)


def build_lst_30m(aoi: ee.Geometry, start: str, end: str,
                  ndvi_10m: ee.Image) -> ee.Image:
   
    l9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterBounds(aoi).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", 30)).map(scale_landsat))
    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(aoi).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", 30)).map(scale_landsat))

    median_img = l9.merge(l8).median().clip(aoi)
    bt = median_img.select("ST_B10")  # Kelvin

    ndvi_30m = ndvi_10m.resample("bilinear").reproject(
        crs=bt.projection(), scale=30)

    pv      = ndvi_30m.subtract(0.2).divide(0.3).pow(2).clamp(0, 1)
    epsilon = pv.multiply(0.004).add(0.986)

    lst = bt.expression(
        "BT/(1+(lam*BT/rho)*log(eps))-273.15",
        {"BT": bt, "eps": epsilon, "lam": 10.8, "rho": 14388.0}
    ).rename("LST_celsius_30m")

    return lst   



def build_tsharp_predictors_30m(s2_stack: ee.Image,
                                 lst_30m: ee.Image,
                                 aoi: ee.Geometry) -> ee.Image:
    
    proj30 = lst_30m.projection()
    s2_30m = s2_stack.select(["NDVI","NDBI","NDWI","MNDWI","BSI","EVI"]) \
                     .resample("bilinear").reproject(crs=proj30, scale=30)

    return s2_30m.addBands(lst_30m)   # 7-band image at 30m



def build_auxiliary_stack(aoi: ee.Geometry,
                          start: str, end: str,
                          s2_proj: ee.Projection) -> ee.Image:
   
    pop = (ee.ImageCollection("WorldPop/GP/100m/pop_age_sex")
             .filter(ee.Filter.eq("country","IND"))
             .filter(ee.Filter.eq("year", 2020))
             .sum().clip(aoi).rename("population_density"))

   
    no2 = (ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_NO2")
             .filterBounds(aoi).filterDate(start, end)
             .select("NO2_column_number_density")
             .mean().clip(aoi).rename("NO2_mol_m2"))

    
    srtm    = ee.Image("USGS/SRTMGL1_003").clip(aoi)
    terrain = ee.Terrain.products(srtm)
    elev    = terrain.select("elevation").rename("elevation_m")
    slope   = terrain.select("slope").rename("slope_deg")
    aspect  = terrain.select("aspect").rename("aspect_deg")

   
    jrc_water = (ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
                   .select("occurrence")
                   .gte(50)               # ≥50% occurrence = permanent water
                   .clip(aoi)
                   .rename("jrc_water_mask"))

    aux = pop.addBands([no2, elev, slope, aspect, jrc_water])
    aux_10m = aux.resample("bilinear").reproject(crs=s2_proj, scale=10)
    return aux_10m   # 6-band image, 10m, each pixel = one 10m cell



def submit_export(image: ee.Image, description: str, aoi: ee.Geometry,
                  scale: int, cfg: dict) -> ee.batch.Task:
    kwargs = dict(
        image=image.toFloat(),
        description=description,
        region=aoi.bounds(),
        scale=scale,
        crs="EPSG:4326",
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    if cfg["export_destination"] == "gcs":
        task = ee.batch.Export.image.toCloudStorage(
            bucket=cfg["gcs_bucket"],
            fileNamePrefix=f"pune_wards/{description}",
            **kwargs)
    else:
        task = ee.batch.Export.image.toDrive(
            folder=cfg["drive_folder"],
            fileNamePrefix=description,
            **kwargs)
    task.start()
    return task


def process_one_ward(ward_name: str, ward_geom: ee.Geometry,
                     cfg: dict) -> dict:
    
    sn  = safe_name(ward_name)
    s   = cfg["start_date"]
    e   = cfg["end_date"]
    sc  = cfg["export_scale_m"]

    print(f"\n  ── {ward_name} ──")

    print(f"    [A] Building S2 stack…")
    s2_stack = build_s2_stack(ward_geom, s, e, cfg["max_cloud_pct"])
    task_a   = submit_export(s2_stack, f"s2_stack_{sn}",
                              ward_geom, sc, cfg)
    print(f"    [A] ✅ s2_stack   task={task_a.status()['id']}")

    print(f"    [B] Building Landsat LST 30m…")
    ndvi_img  = s2_stack.select("NDVI")
    lst_30m   = build_lst_30m(ward_geom, s, e, ndvi_img)
    task_b    = submit_export(lst_30m, f"lst_30m_{sn}",
                               ward_geom, cfg["lst_scale_m"], cfg)
    print(f"    [B] ✅ lst_30m    task={task_b.status()['id']}")

    print(f"    [C] Building auxiliary bands…")
    proj10  = s2_stack.select("B2").projection()
    aux     = build_auxiliary_stack(ward_geom, s, e, proj10)
    task_c  = submit_export(aux, f"auxiliary_{sn}",
                             ward_geom, sc, cfg)
    print(f"    [C] ✅ auxiliary  task={task_c.status()['id']}")

    print(f"    [D] Building TsHARP predictor bands…")
    tsharp  = build_tsharp_predictors_30m(s2_stack, lst_30m, ward_geom)
    task_d  = submit_export(tsharp, f"tsharp_predictor_{sn}",
                             ward_geom, cfg["tsharp_scale_m"], cfg)
    print(f"    [D] ✅ tsharp     task={task_d.status()['id']}")

    return {
        "ward_name": ward_name,
        "s2_stack":  task_a.status()["id"],
        "lst_30m":   task_b.status()["id"],
        "auxiliary": task_c.status()["id"],
        "tsharp":    task_d.status()["id"],
    }



def run(cfg: dict, selected_wards: list = None):
    os.makedirs(cfg["local_output_dir"], exist_ok=True)
    wards = load_wards(cfg["geojson_path"])

    if selected_wards:
        wards = {k: v for k, v in wards.items()
                 if any(s.lower() in k.lower() for s in selected_wards)}
        print(f"[FILTER] Processing: {list(wards.keys())}")

    task_manifest = []
    for ward_name, ward_geom in wards.items():
        result = process_one_ward(ward_name, ward_geom, cfg)
        task_manifest.append(result)

    manifest_path = os.path.join(cfg["local_output_dir"], "gee_pixel_tasks.json")
    with open(manifest_path, "w") as f:
        json.dump(task_manifest, f, indent=2)

    total_tasks = len(task_manifest) * 4
    print(f"\n{'='*65}")
    print(f"  ✅ {len(task_manifest)} ward(s) × 4 exports = {total_tasks} GEE tasks submitted")
    print(f"  📁 Task manifest: {manifest_path}")
    print(f"  ☁️  Check GEE Tasks panel or run: python check_tasks.py")
    print(f"{'='*65}")
    return task_manifest



def check_all_tasks(manifest_path: str):
    with open(manifest_path) as f:
        manifest = json.load(f)

    all_task_ids = []
    for rec in manifest:
        for k, v in rec.items():
            if k != "ward_name":
                all_task_ids.append((rec["ward_name"], k, v))

    tasks = {t.id: t for t in ee.batch.Task.list()}
    print(f"\n{'Ward':<40} {'Export':<20} {'State'}")
    print("-" * 75)
    for ward, export_type, task_id in all_task_ids:
        task  = tasks.get(task_id)
        state = task.status()["state"] if task else "NOT_FOUND"
        icon  = {"COMPLETED":"✅","RUNNING":"🔄","READY":"⏳",
                 "FAILED":"❌","CANCELLED":"⛔"}.get(state, "?")
        print(f"  {ward:<38} {export_type:<18} {icon} {state}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TEORA v2 — Pixel-wise GEE acquisition")
    parser.add_argument("--ward",        type=str,  default=None,
                        help="Ward name substring, e.g. 'Kothrud'")
    parser.add_argument("--all_wards",   action="store_true")
    parser.add_argument("--check_tasks", action="store_true",
                        help="Poll task statuses from saved manifest")
    parser.add_argument("--start_date",  default=CONFIG["start_date"])
    parser.add_argument("--end_date",    default=CONFIG["end_date"])
    parser.add_argument("--export_to",   default="drive",
                        choices=["drive","gcs"])
    args = parser.parse_args()

    cfg = {**CONFIG,
           "start_date": args.start_date,
           "end_date":   args.end_date,
           "export_destination": args.export_to}

    init_gee(cfg["gee_project"])

    if args.check_tasks:
        check_all_tasks(os.path.join(cfg["local_output_dir"], "gee_pixel_tasks.json"))
    elif args.all_wards:
        run(cfg, selected_wards=None)
    elif args.ward:
        run(cfg, selected_wards=[args.ward])
    else:
        print("Usage: --ward 'Kothrud'  OR  --all_wards  OR  --check_tasks")
