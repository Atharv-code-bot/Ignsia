"""
TEORA v3.0 — Stage 5: Hard Feasibility Gate
=============================================
Filters intervention zones by physical plantability constraints.

CHANGES (v3.1):
  - Water scoring REMOVED from this stage.
    Water is now a soft continuous score (water_score) computed in
    Stage 4 (data_fusion.py → compute_water_score) so that ranking
    is always feasibility-aware. This fixes the Rank-Then-Filter flaw
    where infeasible zones could be ranked highest and only removed later.

  - Stage 5 now performs two hard physical checks ONLY:
      1. plantable_area  > min_plantable_area_m2  (default 125 m²)
      2. trees_possible  > 0

  - Zones failing these checks are flagged "❌ Insufficient Plantable Area"
    and filtered out before Stage 6.

  - `water_feasible` column is still written (True for all) for
    backward-compatibility with downstream dashboards, but it is
    informational only — not used as a gate here.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import geopandas as gpd
except ImportError:
    gpd = None

from config.settings import WATER_FILTER_PARAMS, setup_logging

logger = setup_logging("teora.water_filter")


def apply_water_filter(
    zones: "gpd.GeoDataFrame",
    water_gdf: Optional["gpd.GeoDataFrame"] = None,   # kept for API compatibility
    buffer_m: int = None,                              # kept for API compatibility
    min_plantable_area: float = None,
) -> "gpd.GeoDataFrame":
    """
    Stage 5: Hard feasibility gate.

    Checks:
      1. plantable_area > min_plantable_area_m2  (default 125 m²)
      2. trees_possible > 0

    Water proximity is now handled as a soft score (water_score) in
    Stage 4, so this function no longer performs water gating.

    The `water_gdf` and `buffer_m` parameters are accepted but ignored —
    they remain in the signature so that existing callers do not break.

    Args:
        zones: GeoDataFrame from Stage 4 (includes tpis, tpis_final, water_score).
        water_gdf: Ignored (water scoring done in Stage 4).
        buffer_m: Ignored.
        min_plantable_area: Minimum plantable area in m² (default 125).

    Returns:
        GeoDataFrame with `is_feasible`, `area_feasible`, `water_feasible`,
        and `status` columns.
    """
    logger.info("=" * 60)
    logger.info("STAGE 5: HARD FEASIBILITY GATE")
    logger.info("=" * 60)

    if water_gdf is not None:
        logger.info(
            "water_gdf passed to Stage 5 but ignored — water scoring was "
            "already applied in Stage 4 (compute_water_score). "
            "This stage only checks physical plantability constraints."
        )

    min_area = min_plantable_area or WATER_FILTER_PARAMS["min_plantable_area_m2"]

    gdf = zones.copy()

    # ── Physical plantability check ───────────────────────────
    area_col  = gdf.get("plantable_area",  0)
    trees_col = gdf.get("trees_possible",  0)

    gdf["area_feasible"] = (
        (gdf["plantable_area"].values  > min_area) &
        (gdf["trees_possible"].values  > 0)
    ) if "plantable_area" in gdf.columns and "trees_possible" in gdf.columns else True

    # ── water_feasible: informational only (always True now) ──
    # Kept so downstream dashboards that read this column still work.
    # Real water quality is encoded in tpis_final via water_score.
    gdf["water_feasible"] = True

    gdf["is_feasible"] = gdf["area_feasible"]

    # ── Status flags ──────────────────────────────────────────
    gdf["status"] = "✅ GO"
    gdf.loc[~gdf["area_feasible"], "status"] = "❌ Insufficient Plantable Area"

    feasible = gdf[gdf["is_feasible"]]
    logger.info(
        f"Feasible zones: {len(feasible)}/{len(gdf)} "
        f"({len(feasible) / max(len(gdf), 1) * 100:.1f}%)"
    )
    logger.info(
        f"Total plantable trees in feasible zones: "
        f"{int(feasible['trees_possible'].sum()) if 'trees_possible' in feasible.columns else 0}"
    )

    if "tpis_final" in gdf.columns:
        logger.info(
            f"tpis_final range in feasible zones: "
            f"min={feasible['tpis_final'].min():.3f}, "
            f"max={feasible['tpis_final'].max():.3f}"
        )
    else:
        logger.warning(
            "tpis_final column not found — ensure Stage 4 ran with water_gdf. "
            "Stage 6 will fall back to raw tpis."
        )

    logger.info("STAGE 5 COMPLETE")
    return gdf


if __name__ == "__main__":
    logger.info("Water filter module loaded — run via pipeline_runner.py")