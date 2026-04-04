"""
TEORA v3.0 — Stage 2: Environmental Index Computation
=======================================================
Computes NDVI, NDBI, NDWI, BSI, LST from multi-band rasters.
Includes XGBoost TsHARP thermal sharpening (100m → 10m).
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling
except ImportError:
    rasterio = None

try:
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

from config.settings import LANDSAT_THERMAL, setup_logging

logger = setup_logging("teora.env_indices")

EPSILON = 1e-10


def _normalized_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a.astype(np.float32) - b.astype(np.float32))
    den = (a.astype(np.float32) + b.astype(np.float32) + EPSILON)
    return np.clip(num / den, -1.0, 1.0)


def compute_ndvi(b8: np.ndarray, b4: np.ndarray) -> np.ndarray:
    return _normalized_diff(b8, b4)


def compute_ndbi(b11: np.ndarray, b8: np.ndarray) -> np.ndarray:
    return _normalized_diff(b11, b8)


def compute_ndwi(b3: np.ndarray, b8: np.ndarray) -> np.ndarray:
    return _normalized_diff(b3, b8)


def compute_bsi(b11: np.ndarray, b4: np.ndarray,
                b8: np.ndarray, b2: np.ndarray) -> np.ndarray:
    num = (b11.astype(np.float32) + b4.astype(np.float32) -
           b8.astype(np.float32) - b2.astype(np.float32))
    den = (b11.astype(np.float32) + b4.astype(np.float32) +
           b8.astype(np.float32) + b2.astype(np.float32) + EPSILON)
    return np.clip(num / den, -1.0, 1.0)


def compute_lst(st_b10: np.ndarray, ndvi: np.ndarray) -> np.ndarray:
    logger.info("Computing LST from Landsat thermal")

    thermal = st_b10.astype(np.float64)
    bt_k = thermal * LANDSAT_THERMAL["scale_factor"] + LANDSAT_THERMAL["offset"]

    ndvi_resized = ndvi
    if ndvi.shape != thermal.shape:
        from scipy.ndimage import zoom
        zoom_factors = (thermal.shape[0] / ndvi.shape[0],
                        thermal.shape[1] / ndvi.shape[1])
        ndvi_resized = zoom(ndvi, zoom_factors, order=1)

    ndvi_min = np.nanmin(ndvi_resized)
    ndvi_max = np.nanmax(ndvi_resized)
    ndvi_range = ndvi_max - ndvi_min + EPSILON
    pv = ((ndvi_resized - ndvi_min) / ndvi_range) ** 2

    emissivity = 0.004 * pv + 0.986

    lam = LANDSAT_THERMAL["lambda_um"]
    rho = LANDSAT_THERMAL["rho"]
    ln_e = np.log(emissivity + EPSILON)
    lst = bt_k / (1 + (lam * bt_k / rho) * ln_e)

    return (lst - 273.15).astype(np.float32)


class ThermalSharpener:
    def __init__(self, n_estimators=300, max_depth=6, random_state=42):
        if XGBRegressor is None:
            raise ImportError("xgboost not installed")
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            learning_rate=0.1,
        )
        self.is_fitted = False

    def prepare_features(self, ndvi, ndbi, ndwi, bsi):
        X = np.column_stack([
            ndvi.ravel(), ndbi.ravel(), ndwi.ravel(), bsi.ravel()
        ])
        valid = ~np.isnan(X).any(axis=1)
        return X, valid

    def fit(self, features_100m: Dict[str, np.ndarray], lst_100m: np.ndarray):
        X, valid = self.prepare_features(**features_100m)
        y = lst_100m.ravel()

        X_clean = X[valid]
        y_clean = y[valid]

        target_valid = ~np.isnan(y_clean)
        X_clean = X_clean[target_valid]
        y_clean = y_clean[target_valid]

        self.model.fit(X_clean, y_clean)
        self.is_fitted = True
        return self

    def predict(self, features_10m: Dict[str, np.ndarray],
                output_shape: Tuple[int, int]) -> np.ndarray:
        X, valid = self.prepare_features(**features_10m)

        lst_pred = np.full(X.shape[0], np.nan, dtype=np.float32)
        lst_pred[valid] = self.model.predict(X[valid]).astype(np.float32)

        return lst_pred.reshape(output_shape)


def load_bands_from_geotiff(filepath: str):
    if rasterio is None:
        raise ImportError("rasterio not installed")

    with rasterio.open(filepath) as src:
        bands = {}
        for i in range(1, src.count + 1):
            name = src.descriptions[i-1] if src.descriptions[i-1] else f"band_{i}"
            bands[name] = src.read(i).astype(np.float32)
        meta = src.meta.copy()

    return bands, meta


def save_raster(data: np.ndarray, filepath: str, meta: dict):
    if rasterio is None:
        raise ImportError("rasterio not installed")

    out_meta = meta.copy()
    out_meta.update(count=1, dtype=data.dtype)

    with rasterio.open(filepath, "w", **out_meta) as dst:
        dst.write(data, 1)


def run_index_computation(
    s2_path: str,
    thermal_path: str,
    output_dir: Optional[str] = None,
    sharpen: bool = True,
) -> Dict[str, np.ndarray]:

    out_dir = Path(output_dir) if output_dir else Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    s2_bands, s2_meta = load_bands_from_geotiff(s2_path)

    band_keys = list(s2_bands.keys())
    logger.info(f"Loaded {len(band_keys)} bands: {band_keys}")

    if len(band_keys) == 6:
        b2  = s2_bands[band_keys[0]]
        b3  = s2_bands[band_keys[1]]
        b4  = s2_bands[band_keys[2]]
        b8  = s2_bands[band_keys[3]]
        # b8a = s2_bands[band_keys[4]]
        b11 = s2_bands[band_keys[4]]
        b12 = s2_bands[band_keys[5]]
    else:
        raise ValueError(f"Unexpected band count: {len(band_keys)}")

    if any(x is None for x in [b2, b3, b4, b8, b11, b12]):
        raise ValueError("Missing required bands")

    results = {}
    results["ndvi"] = compute_ndvi(b8, b4)
    results["ndbi"] = compute_ndbi(b11, b8)
    results["ndwi"] = compute_ndwi(b3, b8)
    results["bsi"] = compute_bsi(b11, b4, b8, b2)
    
    # Also store raw bands for segmentation model (needs RGB: B4, B3, B2)
    results["B2"] = b2   # Blue
    results["B3"] = b3   # Green
    results["B4"] = b4   # Red
    results["B8"] = b8   # NIR
    results["B11"] = b11  # SWIR

    # Save only computed index rasters (not raw bands)
    for name in ["ndvi", "ndbi", "ndwi", "bsi"]:
        save_raster(results[name], str(out_dir / f"{name}.tif"), s2_meta)

    thermal_bands, thermal_meta = load_bands_from_geotiff(thermal_path)
    st_b10 = list(thermal_bands.values())[0]

    results["lst"] = compute_lst(st_b10, results["ndvi"])
    save_raster(results["lst"], str(out_dir / "lst.tif"), thermal_meta)

    results["s2_meta"] = s2_meta
    results["thermal_meta"] = thermal_meta

    return results