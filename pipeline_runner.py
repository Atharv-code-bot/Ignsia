"""
TEORA v3.0 — End-to-End Pipeline Orchestrator
================================================
Chains all 8 stages: Acquisition → Indices → Segmentation →
Impact Score → Water Filter → Knapsack → Anomaly → Dashboard.

CHANGES (v3.1):
  - water_gdf loaded BEFORE Stage 4 and passed into run_data_fusion().
    This allows Stage 4 to compute water_score (continuous 0–1) and
    tpis_final (blended score) so that ranking is water-aware from
    the start.  Fixes the Rank-Then-Filter flaw.

  - Stage 5 (apply_water_filter) is now a hard physical feasibility
    gate only (min plantable area + trees > 0).  It no longer receives
    water_gdf because water scoring is done in Stage 4.

  - Stage 6 (knapsack_optimize) receives all feasible zones and uses
    tpis_final automatically.

CHANGES (v3.2):
  - Stage 3 replaced: run_stage3_segmentation() → run_stage3_plantability().
    Imports from plantability.py instead of segmentation_model.py.
    No ML model required — fully deterministic Sentinel-2-native computation.

  - run_stage4_fusion() call updated: seg_mask= → plantability_map=.
    Stage 4 (data_fusion.py) no longer accepts or uses seg_mask.

  - run_stage3_segmentation() retained as a deprecated alias pointing to
    run_stage3_plantability() to avoid breaking external callers.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timedelta

import numpy as np

try:
    import geopandas as gpd
except ImportError:
    gpd = None

from config.settings import (
    TPIS_WEIGHTS, MAX_TREES, DATA_DIR, OUTPUT_DIR, setup_logging, CITY_CONFIG
)

logger = setup_logging("teora.pipeline")


class TEORAPipeline:
    """
    End-to-end TEORA pipeline orchestrator.

    Stages:
      1. GEE Data Acquisition
      2. Environmental Index Computation
      3. Land Cover Segmentation
      3.5. Land Rights & Protected Area Validation
      4. Impact Score Fusion (TPIS + water_score → tpis_final)   ← water now here
      5. Hard Feasibility Gate (area / trees check only)          ← simplified
      6. Knapsack Optimization  (uses tpis_final)
      7. Anomaly Detection & Re-ranking
      8. Dashboard (launched separately)
    """

    def __init__(
        self,
        aoi_geojson: str,
        neighborhoods_geojson: str,
        start_date: str = "2023-01-01",
        end_date: str = "2023-12-31",
        weights: Optional[Dict[str, float]] = None,
        max_trees: int = None,
        output_dir: Optional[str] = None,
    ):
        self.aoi_path = aoi_geojson
        self.neighborhoods_path = neighborhoods_geojson
        end_date_obj   = datetime.today()
        start_date_obj = end_date_obj - timedelta(days=180)

        self.start_date = start_date_obj.strftime("%Y-%m-%d")
        self.end_date   = end_date_obj.strftime("%Y-%m-%d")
        self.weights    = weights or TPIS_WEIGHTS
        self.max_trees  = max_trees or MAX_TREES
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stage_results = {}
        self.timing = {}

        logger.info("TEORA Pipeline initialized")
        logger.info(f"  AOI: {self.aoi_path}")
        logger.info(f"  Date range: {self.start_date} → {self.end_date}")
        logger.info(f"  Budget: {self.max_trees} trees")
        logger.info(f"  Output: {self.output_dir}")

    # ─── Stage execution wrapper ──────────────────────────────

    def _time_stage(self, name, func, *args, **kwargs):
        """Execute a stage with timing and error handling."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting: {name}")
        logger.info(f"{'='*60}")
        t0 = time.time()
        try:
            result  = func(*args, **kwargs)
            elapsed = time.time() - t0
            self.timing[name] = elapsed
            logger.info(f"✅ {name} completed in {elapsed:.1f}s")
            return result
        except Exception as e:
            elapsed = time.time() - t0
            self.timing[name] = elapsed
            logger.error(f"❌ {name} failed after {elapsed:.1f}s: {e}")
            raise

    # ─── Individual stage methods ─────────────────────────────

    def run_stage1_acquisition(self) -> Dict:
        """Stage 1: GEE Data Acquisition."""
        from gee_pipeline import run_acquisition_pipeline
        return self._time_stage(
            "Stage 1: Data Acquisition",
            run_acquisition_pipeline,
            aoi_geojson=self.aoi_path,
            start_date=self.start_date,
            end_date=self.end_date,
            output_dir=self.output_dir / "data",
        )

    def run_stage2_indices(self, s2_path, thermal_path) -> Dict:
        """Stage 2: Environmental Index Computation."""
        from env_indices import run_index_computation
        return self._time_stage(
            "Stage 2: Environmental Indices",
            run_index_computation,
            s2_path=str(s2_path),
            thermal_path=str(thermal_path),
            output_dir=str(self.output_dir / "indices"),
        )

    def run_stage3_plantability(self, indices: Dict, meta: dict) -> Dict:
        """
        Stage 3: Plantability Engine (replaces RGB segmentation model).

        Consumes indices already computed by Stage 2 — no ML model, no
        domain-mismatch risk, fully deterministic and Sentinel-2-native.

        Args:
            indices: Dict returned by run_stage2_indices() — must contain
                     keys: ndvi, ndbi, lst  (and optionally ndwi, bsi).
            meta:    Rasterio metadata dict (s2_meta from Stage 2).
        """
        from plantability import run_plantability_pipeline
        return self._time_stage(
            "Stage 3: Plantability Engine",
            run_plantability_pipeline,
            ndvi=indices["ndvi"],
            ndbi=indices["ndbi"],
            lst=indices.get("lst_sharpened", indices["lst"]),
            ndwi=indices.get("ndwi"),
            bsi=indices.get("bsi"),
            raster_meta=meta,
            output_dir=str(self.output_dir / "plantability"),
        )

    def run_stage3_segmentation(self, s2_bands: Dict, meta: dict) -> Dict:
        """
        Deprecated alias → run_stage3_plantability().

        Retained so any external code calling run_stage3_segmentation()
        continues to work.  The s2_bands argument is ignored — indices are
        pulled from the Stage 2 results stored in self.stage_results.
        """
        logger.warning(
            "run_stage3_segmentation() is deprecated in v3.2. "
            "Redirecting to run_stage3_plantability(). "
            "Update callers to use run_stage3_plantability() directly."
        )
        indices = self.stage_results.get("indices", {})
        return self.run_stage3_plantability(indices, meta)

    def run_stage3_5_land_rights(self, zones) -> Any:
        """Stage 3.5: Land Rights & Protected Area Validation."""
        try:
            from land_rights_validation import (
                validate_land_rights, add_land_rights_to_knapsack,
                load_protected_areas, load_land_ownership, load_no_go_zones,
            )
        except ImportError as e:
            logger.warning(f"Stage 3.5 unavailable: {e}. Skipping land rights validation.")
            return zones

        try:
            pa   = load_protected_areas()
            lo   = load_land_ownership()
            nogo = load_no_go_zones()
        except FileNotFoundError as e:
            logger.warning(f"Land rights data files not found: {e}. Skipping validation.")
            return zones

        zones_validated = self._time_stage(
            "Stage 3.5: Land Rights Validation",
            validate_land_rights,
            zones=zones,
            protected_areas=pa,
            land_ownership=lo,
            no_go_zones=nogo,
        )
        return add_land_rights_to_knapsack(zones_validated)

    def run_stage4_fusion(
        self,
        polygons,
        ndvi,
        lst,
        plantability_map,
        meta,
        water_gdf=None,
    ) -> Any:
        """
        Stage 4: Impact Score Fusion (TPIS + water_score → tpis_final).

        v3.2: accepts plantability_map (from Stage 3 Plantability Engine)
        instead of seg_mask.  Stage 4 derives plantable_area and
        trees_possible directly from the plantability raster.

        water_gdf is loaded by run_full_pipeline() and passed here so
        that water_score is computed BEFORE ranking — fixing the
        Rank-Then-Filter flaw that existed in v3.0.
        """
        from data_fusion import run_data_fusion
        return self._time_stage(
            "Stage 4: Impact Score Fusion",
            run_data_fusion,
            polygons=polygons,
            ndvi=ndvi,
            lst=lst,
            plantability_map=plantability_map,
            raster_meta=meta,
            weights=self.weights,
            water_gdf=water_gdf,
            output_dir=str(self.output_dir / "fusion"),
        )

    def run_stage5_feasibility_gate(self, zones) -> Any:
        """
        Stage 5: Hard Feasibility Gate (area + trees check only).

        Water scoring has moved to Stage 4. This stage now only removes
        zones that are physically unplantable.
        """
        from water_filter import apply_water_filter
        return self._time_stage(
            "Stage 5: Hard Feasibility Gate",
            apply_water_filter,
            zones=zones,
            # water_gdf intentionally NOT passed — handled in Stage 4
        )

    # Legacy alias so any external code calling run_stage5_water_filter still works
    def run_stage5_water_filter(self, zones) -> Any:
        return self.run_stage5_feasibility_gate(zones)

    def run_stage6_knapsack(self, feasible_zones) -> Dict:
        """Stage 6: Knapsack Optimization (uses tpis_final)."""
        from optimizer import knapsack_optimize
        return self._time_stage(
            "Stage 6: Knapsack Optimization",
            knapsack_optimize,
            feasible_zones=feasible_zones,
            budget_rupees=self.max_trees,   # 🔥 TEMP MAPPING
        )

    def run_stage7_anomaly(self, zones) -> Any:
        """Stage 7: Anomaly Detection & Re-ranking."""
        from anomaly_layer import run_anomaly_pipeline
        return self._time_stage(
            "Stage 7: Anomaly Layer",
            run_anomaly_pipeline,
            df=zones,
            output_dir=str(self.output_dir / "anomaly"),
        )

    # ─── Full pipeline ────────────────────────────────────────

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Execute the complete pipeline."""
        logger.info("🌳 TEORA v3.0 — Full Pipeline Execution")
        logger.info("=" * 60)
        t_total = time.time()

        if gpd is None:
            raise ImportError("geopandas required for pipeline execution")

        neighborhoods = gpd.read_file(self.neighborhoods_path)
        logger.info(f"Loaded {len(neighborhoods)} neighborhood polygons")

        # ── Stage 1 (skipped — using existing outputs) ─────────
        s1 = {
            "s2_path":      str(self.output_dir / "data" / "sentinel2_composite.tif"),
            "thermal_path": str(self.output_dir / "data" / "landsat_thermal.tif"),
        }
        self.stage_results["acquisition"] = s1

        missing = [k for k in ("s2_path", "thermal_path") if k not in s1]
        if missing:
            raise RuntimeError(
                f"Stage 1 did not produce required outputs: {missing}. "
                "Check the GEE export logs above."
            )

        # ── Stage 2 ────────────────────────────────────────────
        s2 = self.run_stage2_indices(s1["s2_path"], s1["thermal_path"])
        self.stage_results["indices"] = s2

        # ── Stage 3 (Plantability Engine — replaces segmentation) ─
        s3 = self.run_stage3_plantability(
            indices=s2,
            meta=s2["s2_meta"],
        )
        self.stage_results["plantability"] = s3

        # ── Load water network BEFORE Stage 4 ──────────────────
        # Water GDF is now consumed by Stage 4 (not Stage 5) so that
        # water_score is incorporated into ranking before any filtering.
        water_path = self.output_dir / "data" / "water_network.geojson"
        if water_path.exists():
            from water_filter import load_water_network
            water_gdf = load_water_network(str(water_path))
            logger.info(f"Water network loaded for Stage 4 scoring: {water_path}")
        else:
            water_gdf = None
            logger.warning(
                f"Water network file not found at {water_path}. "
                "water_score will default to 0.5 (neutral) for all zones."
            )

        # ── Stage 4 (now includes water scoring) ───────────────
        s4 = self.run_stage4_fusion(
            polygons=neighborhoods,
            ndvi=s2["ndvi"],
            lst=s2.get("lst_sharpened", s2["lst"]),
            plantability_map=s3["plantability_map"],
            meta=s2["s2_meta"],
            water_gdf=water_gdf,
        )
        self.stage_results["fusion"] = s4

        # ── Stage 3.5 (land rights — inserted after fusion) ────
        s3_5 = self.run_stage3_5_land_rights(s4)
        self.stage_results["land_rights"] = s3_5

        # ── Stage 5 (hard feasibility gate only, no water) ─────
        s5 = self.run_stage5_feasibility_gate(s3_5)
        self.stage_results["water_filter"] = s5

        # ── Stage 6 ────────────────────────────────────────────
        feasible = s5[s5["is_feasible"]] if "is_feasible" in s5.columns else s5
        s6 = self.run_stage6_knapsack(feasible)
        self.stage_results["optimizer"] = s6

        # ── Stage 7 ────────────────────────────────────────────
        s7 = self.run_stage7_anomaly(s6.get("zones", s5))
        self.stage_results["anomaly"] = s7

        # ── Save final output ───────────────────────────────────
        final_output = s7
        final_output.to_file(
            str(self.output_dir / "final_output.geojson"),
            driver="GeoJSON",
        )

        total_time = time.time() - t_total
        self.timing["total"] = total_time

        logger.info("\n" + "=" * 60)
        logger.info("🌳 TEORA PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.1f}s")
        for stage, t in self.timing.items():
            logger.info(f"  {stage}: {t:.1f}s")
        logger.info(f"Output: {self.output_dir / 'final_output.geojson'}")

        return {
            "final_zones": final_output,
            "timing":      self.timing,
            "output_dir":  str(self.output_dir),
        }


# ─── Demo data generator ──────────────────────────────────────

def generate_demo_data(output_dir: str = "demo_data") -> Dict[str, str]:
    """Generate synthetic demo data for testing the pipeline."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    aoi = {
        "type": "Feature",
        "properties": {"name": "Demo AOI"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [77.18, 28.60], [77.28, 28.60],
                [77.28, 28.68], [77.18, 28.68],
                [77.18, 28.60],
            ]],
        },
    }
    aoi_path = out / "aoi.geojson"
    with open(aoi_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": [aoi]}, f)

    if gpd is not None:
        from shapely.geometry import box
        polys    = []
        zone_id  = 0
        for lat in np.arange(28.60, 28.68, 0.02):
            for lon in np.arange(77.18, 77.28, 0.02):
                polys.append({
                    "zone_id":  f"zone_{zone_id:03d}",
                    "name":     f"Ward {zone_id + 1}",
                    "geometry": box(lon, lat, lon + 0.02, lat + 0.02),
                })
                zone_id += 1
        gdf = gpd.GeoDataFrame(polys, crs="EPSG:4326")
        neighborhoods_path = out / "neighborhoods.geojson"
        gdf.to_file(str(neighborhoods_path), driver="GeoJSON")
    else:
        neighborhoods_path = out / "neighborhoods.geojson"
        features = []
        zone_id  = 0
        for lat in np.arange(28.60, 28.68, 0.02):
            for lon in np.arange(77.18, 77.28, 0.02):
                features.append({
                    "type": "Feature",
                    "properties": {"zone_id": f"zone_{zone_id:03d}", "name": f"Ward {zone_id+1}"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[lon, lat], [lon+0.02, lat],
                                         [lon+0.02, lat+0.02], [lon, lat+0.02],
                                         [lon, lat]]],
                    },
                })
                zone_id += 1
        with open(neighborhoods_path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)

    logger.info(f"Demo data generated in {out}")
    return {"aoi": str(aoi_path), "neighborhoods": str(neighborhoods_path)}


# ─── CLI entry point ──────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TEORA Pipeline Runner")
    parser.add_argument("--city",          help="City name from config (e.g., pune)")
    parser.add_argument("--aoi",           help="AOI GeoJSON file")
    parser.add_argument("--neighborhoods", help="Neighborhoods GeoJSON")
    parser.add_argument("--start-date",    default="2023-01-01")
    parser.add_argument("--end-date",      default="2023-12-31")
    parser.add_argument("--max-trees",     type=int, default=1000)
    parser.add_argument("--output-dir",    default="output")
    parser.add_argument("--demo",          action="store_true", help="Generate demo data")
    args = parser.parse_args()

    if args.demo:
        paths = generate_demo_data()
        print(f"Demo data: {paths}")

    elif args.city:
        if args.city not in CITY_CONFIG:
            raise ValueError(f"City '{args.city}' not found in config")
        paths    = CITY_CONFIG[args.city]
        pipeline = TEORAPipeline(
            aoi_geojson=str(paths["aoi"]),
            neighborhoods_geojson=str(paths["wards"]),
            max_trees=args.max_trees,
            output_dir=args.output_dir,
        )
        result = pipeline.run_full_pipeline()

    elif args.aoi and args.neighborhoods:
        pipeline = TEORAPipeline(
            aoi_geojson=args.aoi,
            neighborhoods_geojson=args.neighborhoods,
            start_date=args.start_date,
            end_date=args.end_date,
            max_budget=args.max_budget,
            output_dir=args.output_dir,
        )
        result = pipeline.run_full_pipeline()

    else:
        parser.print_help()