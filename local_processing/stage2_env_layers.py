import os, re, argparse, pickle, warnings
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings("ignore")

S2_BANDS = {
    "B2":1,"B3":2,"B4":3,"B5":4,"B6":5,"B7":6,
    "B8":7,"B8A":8,"B11":9,"B12":10,
    "NDVI":11,"NDBI":12,"NDWI":13,"MNDWI":14,"BSI":15,"EVI":16
}
TSHARP_BANDS = {"NDVI":1,"NDBI":2,"NDWI":3,"MNDWI":4,"BSI":5,"EVI":6,"LST_celsius_30m":7}
TSHARP_FEATURES = ["NDVI","NDBI","NDWI","MNDWI","BSI","EVI"]


def safe_name(ward: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]","_", ward).strip("_")


def read_band(tif_path: str, band_idx: int) -> tuple:
    with rasterio.open(tif_path) as src:
        data    = src.read(band_idx).astype(np.float32)
        profile = src.profile.copy()
        data[data == src.nodata] = np.nan if src.nodata else data
    return data, profile


def read_all_bands(tif_path: str) -> tuple:
    with rasterio.open(tif_path) as src:
        data    = src.read().astype(np.float32)
        profile = src.profile.copy()
    return data, profile


def write_raster(arr: np.ndarray, profile: dict, out_path: str,
                 band_names: list = None):
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    p = profile.copy()
    p.update(count=arr.shape[0], dtype="float32", compress="lzw",
             tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(arr.astype(np.float32))
        if band_names:
            for i, name in enumerate(band_names, 1):
                dst.update_tags(i, name=name)
    print(f"    → Written: {os.path.basename(out_path)}  shape={arr.shape}")



def run_tsharp(tsharp_tif: str, s2_stack_tif: str,
               ward_name: str, out_dir: str) -> np.ndarray:
    
    print(f"    [TsHARP] Loading 30m predictor bands…")
    tsharp_data, _ = read_all_bands(tsharp_tif)

    X_30 = np.stack([tsharp_data[i-1] for i in
                     [TSHARP_BANDS[f] for f in TSHARP_FEATURES]], axis=0)
    y_30 = tsharp_data[TSHARP_BANDS["LST_celsius_30m"] - 1]

    X_flat = X_30.reshape(len(TSHARP_FEATURES), -1).T
    y_flat = y_30.reshape(-1)
    valid  = np.isfinite(X_flat).all(axis=1) & np.isfinite(y_flat)
    X_tr, y_tr = X_flat[valid], y_flat[valid]

    print(f"    [TsHARP] Training on {X_tr.shape[0]} 30m pixels…")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = []
    for fold, (tr_i, va_i) in enumerate(kf.split(X_tr)):
        m = xgb.XGBRegressor(n_estimators=300, max_depth=6,
                              learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, n_jobs=-1,
                              objective="reg:squarederror", random_state=42)
        m.fit(X_tr[tr_i], y_tr[tr_i],
              eval_set=[(X_tr[va_i], y_tr[va_i])], verbose=False)
        r2 = r2_score(y_tr[va_i], m.predict(X_tr[va_i]))
        cv_r2.append(r2)
    print(f"    [TsHARP] CV R² = {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")

    final_model = xgb.XGBRegressor(n_estimators=400, max_depth=6,
                                    learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.8, n_jobs=-1,
                                    objective="reg:squarederror", random_state=42)
    final_model.fit(X_tr, y_tr, verbose=False)

    model_path = os.path.join(out_dir, f"tsharp_model_{safe_name(ward_name)}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    print(f"    [TsHARP] Predicting LST at 10m…")
    s2_data, s2_profile = read_all_bands(s2_stack_tif)
    H10, W10 = s2_data.shape[1], s2_data.shape[2]

    X_10 = np.stack([s2_data[S2_BANDS[f]-1] for f in TSHARP_FEATURES], axis=0)
    X_10_flat = X_10.reshape(len(TSHARP_FEATURES), -1).T

    chunk = 500_000
    lst_pred = np.full(X_10_flat.shape[0], np.nan, dtype=np.float32)
    for i in range(0, len(X_10_flat), chunk):
        blk = X_10_flat[i:i+chunk]
        valid_blk = np.isfinite(blk).all(axis=1)
        if valid_blk.any():
            lst_pred[i:i+chunk][valid_blk] = final_model.predict(blk[valid_blk])

    lst_10m = lst_pred.reshape(H10, W10)
    print(f"    [TsHARP] ✅ LST 10m: range=[{np.nanmin(lst_10m):.1f}, "
          f"{np.nanmax(lst_10m):.1f}]°C   shape={lst_10m.shape}")

    return lst_10m, s2_profile



def process_ward_stage2(ward_name: str, geotiff_dir: str, out_dir: str):
    
    sn = safe_name(ward_name)
    os.makedirs(out_dir, exist_ok=True)

    s2_tif      = os.path.join(geotiff_dir, f"s2_stack_{sn}.tif")
    tsharp_tif  = os.path.join(geotiff_dir, f"tsharp_predictor_{sn}.tif")

    for tif in [s2_tif, tsharp_tif]:
        if not os.path.exists(tif):
            print(f"  [SKIP] Missing: {tif}")
            return None

    print(f"\n  [Stage 2] {ward_name}")

    
    lst_10m, s2_profile = run_tsharp(tsharp_tif, s2_tif, ward_name, out_dir)

    s2_data, _ = read_all_bands(s2_tif)
    def b(name): return s2_data[S2_BANDS[name]-1]

    ndvi_path = os.path.join(out_dir, f"ndvi_10m_{sn}.tif")
    lst_path  = os.path.join(out_dir, f"lst_10m_{sn}.tif")
    write_raster(b("NDVI"),  s2_profile, ndvi_path)
    write_raster(lst_10m,    s2_profile, lst_path)

    env_stack = np.stack([
        b("NDVI"), b("EVI"), b("NDBI"), b("BSI"),
        b("NDWI"), b("MNDWI"), lst_10m,
        b("B4"), b("B8"), b("B11")
    ], axis=0)   

    env_path = os.path.join(out_dir, f"env_stack_{sn}.tif")
    band_names = ["NDVI","EVI","NDBI","BSI","NDWI","MNDWI","LST_10m",
                  "B4_red","B8_nir","B11_swir"]
    write_raster(env_stack, s2_profile, env_path, band_names)

    print(f"  [Stage 2] ✅ env_stack shape={env_stack.shape}  "
          f"LST range=[{np.nanmin(lst_10m):.1f},{np.nanmax(lst_10m):.1f}]°C")
    return env_path


def run_all(geotiff_dir: str, out_dir: str, selected: list = None):
    wards = [w for w in os.listdir(geotiff_dir)
             if w.startswith("s2_stack_") and w.endswith(".tif")]
    ward_names_from_files = [
        re.sub(r"^s2_stack_","",w).replace(".tif","").replace("_"," ").strip()
        for w in wards
    ]
    if selected:
        ward_names_from_files = [w for w in ward_names_from_files
                                  if any(s.lower() in w.lower() for s in selected)]

    results = []
    for wn in ward_names_from_files:
        r = process_ward_stage2(wn, geotiff_dir, out_dir)
        if r:
            results.append(r)
    print(f"\n[Stage 2] Complete. {len(results)} wards processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ward",        default=None)
    parser.add_argument("--all_wards",   action="store_true")
    parser.add_argument("--geotiff_dir", default="./ward_outputs/geotiffs/")
    parser.add_argument("--out_dir",     default="./ward_outputs/stage2/")
    args = parser.parse_args()

    if args.all_wards:
        run_all(args.geotiff_dir, args.out_dir)
    elif args.ward:
        process_ward_stage2(args.ward, args.geotiff_dir, args.out_dir)
    else:
        print("Usage: --ward 'Kothrud'  OR  --all_wards")
