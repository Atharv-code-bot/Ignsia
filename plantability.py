"""
TEORA v3.2 — Stage 3: Plantability Engine
==========================================
Replaces the RGB segmentation model with a deterministic,
index-driven plantability computation.

WHY:
  The segmentation model (SegFormer) was trained on high-quality aerial
  imagery.  Running it on Sentinel-2 data creates a domain mismatch that
  produces unreliable seg_mask values for canopy_pct / bare_pct.

  This stage instead derives plantability directly from spectral indices
  already computed in Stage 2 (ndvi, ndbi, lst) — no ML, no domain gap,
  fully deterministic and Sentinel-2-native.

FORMULA:
  ndvi_score     = min-max normalize(NDVI)
  lst_score      = min-max normalize(LST) then inverted  (hot → low score)
  built_penalty  = min-max normalize(NDBI)

  plantability   = 0.4 × ndvi_score
                 + 0.3 × (1 − lst_score)
                 + 0.3 × (1 − built_penalty)

POST-PROCESSING:
  · plantability = 0 where NDVI > 0.6  (already-vegetated pixels excluded)
  · Clipped to [0, 1]
  · Saved as "plantability.tif" using Stage 2 raster_meta

OUTPUT:
  {
      "plantability_map": np.ndarray  (H, W)  float32,
      "plantability_tif": str         path to saved GeoTIFF,
  }
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None

from config.settings import setup_logging

logger = setup_logging("teora.plantability")

# ── Tunable constants ─────────────────────────────────────────────────────────

NDVI_SATURATION_THRESHOLD = 0.6   # pixels above this are already vegetated
PLANTABILITY_WEIGHTS = {
    "ndvi":          0.4,
    "lst_inverted":  0.3,
    "built_penalty": 0.3,
}

_EPSILON = 1e-10


# ── Helpers ───────────────────────────────────────────────────────────────────

def _min_max_norm(arr: np.ndarray) -> np.ndarray:
    """
    Min-max normalize a 2-D array to [0, 1].
    NaNs are ignored in range computation and preserved in output.
    Returns zeros if the range is effectively zero.
    """
    mn  = np.nanmin(arr)
    mx  = np.nanmax(arr)
    rng = mx - mn
    if rng < _EPSILON:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / rng).astype(np.float32)


def _align_shape(src: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Resize src to target_shape using bilinear zoom if shapes differ.
    No-op when shapes already match.
    """
    if src.shape == target_shape:
        return src
    from scipy.ndimage import zoom
    factors = (target_shape[0] / src.shape[0], target_shape[1] / src.shape[1])
    logger.debug(f"Aligning raster {src.shape} → {target_shape} (zoom {factors})")
    return zoom(src.astype(np.float32), factors, order=1)


# ── Core computation ──────────────────────────────────────────────────────────

def compute_plantability(
    ndvi:  np.ndarray,
    ndbi:  np.ndarray,
    lst:   np.ndarray,
    ndwi:  Optional[np.ndarray] = None,   # accepted but not used in formula — reserved
    bsi:   Optional[np.ndarray] = None,   # accepted but not used in formula — reserved
) -> np.ndarray:
    """
    Compute the plantability raster from spectral indices.

    All inputs must be 2-D float arrays.  The function handles shape
    mismatches by resampling ndbi and lst to match the ndvi shape
    (ndvi is the reference because it has the highest resolution from S2).

    Args:
        ndvi:  NDVI raster  ∈ [-1, 1]  shape (H, W)
        ndbi:  NDBI raster  ∈ [-1, 1]  shape (H, W) or smaller
        lst:   LST  raster  in °C      shape (H, W) or smaller
        ndwi:  Optional NDWI — accepted for API compatibility, unused.
        bsi:   Optional BSI  — accepted for API compatibility, unused.

    Returns:
        plantability map  ∈ [0, 1]  float32  shape (H, W)
    """
    target = ndvi.shape

    # Align all inputs to NDVI resolution (reference grid)
    ndvi_a = np.nan_to_num(ndvi.astype(np.float32), nan=0.0)
    ndbi_a = np.nan_to_num(_align_shape(ndbi.astype(np.float32), target), nan=0.0)
    lst_a  = np.nan_to_num(_align_shape(lst.astype(np.float32),  target), nan=float(np.nanmean(lst)))

    # Component scores — each in [0, 1]
    ndvi_score    = _min_max_norm(ndvi_a)                    # high NDVI → higher score
    lst_score     = _min_max_norm(lst_a)                     # high LST  → higher normalized value
    built_penalty = _min_max_norm(ndbi_a)                    # high NDBI → more built-up

    w = PLANTABILITY_WEIGHTS
    lst_inverted = 1.0 - lst_score

    plantability = (
        w["ndvi"] * ndvi_score
        + w["lst_inverted"] * lst_inverted
        + w["built_penalty"] * (1.0 - built_penalty)
    )

    # Zero out already-vegetated pixels (NDVI > threshold)
    already_vegetated = ndvi_a > NDVI_SATURATION_THRESHOLD
    plantability[already_vegetated] = 0.0

    # Final safety clip
    plantability = np.clip(plantability, 0.0, 1.0).astype(np.float32)

    n_high = int(np.sum(plantability > 0.6))
    n_zero = int(np.sum(plantability == 0.0))
    logger.info(
        f"Plantability: mean={np.nanmean(plantability):.3f}, "
        f"high-potential pixels (>0.6)={n_high:,}, "
        f"zeroed (vegetated/NaN)={n_zero:,}"
    )

    return plantability


def _save_plantability_tif(
    plantability_map: np.ndarray,
    raster_meta: dict,
    output_path: str,
) -> str:
    """
    Save the plantability raster as a single-band GeoTIFF.

    Uses raster_meta from Stage 2 so the output is spatially aligned
    with ndvi, lst, and the segmentation CRS.
    """
    if rasterio is None:
        logger.warning("rasterio not installed — skipping plantability GeoTIFF save")
        return ""

    meta = raster_meta.copy()
    meta.update(count=1, dtype=np.float32)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(plantability_map, 1)

    logger.info(f"Saved plantability raster → {output_path}")
    return output_path


# ── Public entry point ────────────────────────────────────────────────────────

def run_plantability_pipeline(
    ndvi:         np.ndarray,
    ndbi:         np.ndarray,
    lst:          np.ndarray,
    raster_meta:  dict,
    ndwi:         Optional[np.ndarray] = None,
    bsi:          Optional[np.ndarray] = None,
    output_dir:   Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stage 3: Plantability Engine.

    Drop-in replacement for run_segmentation_pipeline().
    Consumes indices already computed by Stage 2 — no model loading,
    no tiling, no domain-mismatch issues.

    Args:
        ndvi:        NDVI array from Stage 2  (H, W)
        ndbi:        NDBI array from Stage 2  (H, W)
        lst:         LST  array from Stage 2  (H, W) — may be coarser resolution
        raster_meta: Rasterio metadata dict from Stage 2 (s2_meta)
        ndwi:        Optional NDWI (reserved for future weighting)
        bsi:         Optional BSI  (reserved for future weighting)
        output_dir:  Directory to write plantability.tif

    Returns:
        {
            "plantability_map": np.ndarray  (H, W)  float32 in [0, 1],
            "plantability_tif": str         absolute path to saved GeoTIFF
                                            (empty string if rasterio unavailable),
        }

    Pipeline compatibility note:
        Stage 4 (data_fusion.py) now accepts `plantability_map` instead of
        `seg_mask`.  compute_zonal_stats() derives plantable_area and
        trees_possible directly from the plantability raster rather than
        counting segmentation class pixels.
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: PLANTABILITY ENGINE")
    logger.info("=" * 60)
    logger.info(f"NDVI shape: {ndvi.shape}  |  NDBI shape: {ndbi.shape}  |  LST shape: {lst.shape}")

    out_dir = Path(output_dir or "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    plantability_map = compute_plantability(
        ndvi=ndvi,
        ndbi=ndbi,
        lst=lst,
        ndwi=ndwi,
        bsi=bsi,
    )

    tif_path = _save_plantability_tif(
        plantability_map=plantability_map,
        raster_meta=raster_meta,
        output_path=str(out_dir / "plantability.tif"),
    )

    logger.info("STAGE 3 COMPLETE")
    return {
        "plantability_map": plantability_map,
        "plantability_tif": tif_path,
    }


if __name__ == "__main__":
    logger.info("Stage 3 Plantability Engine ready — run via pipeline_runner.py")
