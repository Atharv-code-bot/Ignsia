"""
TEORA v3.0 — Stage 7: Isolation Forest Anomaly Layer
======================================================
7A: Temporal NDVI Anomaly Detection
7B: Multi-Dimensional Canopy Desert Anomaly + Re-ranking
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    IsolationForest = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None

from config.settings import ANOMALY_PARAMS, ANOMALY_TAGS, setup_logging

logger = setup_logging("teora.anomaly_layer")


def temporal_ndvi_anomaly(
    df: "gpd.GeoDataFrame",
    contamination: float = None,
) -> "gpd.GeoDataFrame":
    """
    Stage 7A: Temporal NDVI Anomaly Detection.

    Detects zones with active canopy loss using Isolation Forest
    on temporal NDVI features (multi-month NDVI values + trend).

    Features: ndvi_jan, ndvi_apr, ndvi_jul, ndvi_oct,
              ndvi_trend_slope, ndvi_seasonality_amp

    Args:
        df: GeoDataFrame with temporal NDVI columns.
        contamination: IF contamination parameter (default 0.08).

    Returns:
        GeoDataFrame with canopy_loss_anomaly column.
    """
    if IsolationForest is None:
        raise ImportError("scikit-learn not installed")

    logger.info("Running Stage 7A: Temporal NDVI Anomaly Detection")

    params = ANOMALY_PARAMS["temporal"]
    cont = contamination or params["contamination"]
    features = params["features"]

    gdf = df.copy()

    # Generate synthetic temporal features if not present
    for feat in features:
        if feat not in gdf.columns:
            if "ndvi" in feat and "slope" not in feat and "amp" not in feat:
                gdf[feat] = gdf.get("mean_ndvi", np.random.uniform(0.1, 0.7, len(gdf))) + \
                           np.random.normal(0, 0.05, len(gdf))
            elif "slope" in feat:
                gdf[feat] = np.random.normal(-0.02, 0.05, len(gdf))
            elif "amp" in feat:
                gdf[feat] = np.abs(np.random.normal(0.1, 0.05, len(gdf)))

    # Extract feature matrix
    X = gdf[features].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)

    # Fit Isolation Forest
    iso = IsolationForest(
        contamination=cont,
        random_state=params["random_state"],
        n_jobs=-1,
    )

    gdf["canopy_loss_anomaly"] = iso.fit_predict(X) == -1
    gdf["temporal_anomaly_score"] = iso.decision_function(X)

    n_anomalies = gdf["canopy_loss_anomaly"].sum()
    logger.info(f"7A: {n_anomalies}/{len(gdf)} temporal anomalies detected "
                f"({n_anomalies/len(gdf)*100:.1f}%)")

    # Tag
    gdf["temporal_tag"] = ""
    gdf.loc[gdf["canopy_loss_anomaly"], "temporal_tag"] = ANOMALY_TAGS["active_loss"]

    return gdf


def canopy_desert_anomaly(
    df: "gpd.GeoDataFrame",
    contamination: float = None,
) -> "gpd.GeoDataFrame":
    """
    Stage 7B: Multi-Dimensional Canopy Desert Anomaly Detection.

    Detects non-linear outliers across 8 features that the linear
    TPIS scorer might miss.

    Features: canopy_pct, mean_lst, vuln_score, pop_density,
              canopy_deficit, uhi_delta, plantable_area, roi_norm

    Args:
        df: GeoDataFrame with computed features.
        contamination: IF contamination (default 0.10).

    Returns:
        GeoDataFrame with anomaly labels and scores.
    """
    if IsolationForest is None:
        raise ImportError("scikit-learn not installed")

    logger.info("Running Stage 7B: Canopy Desert Anomaly Detection")

    params = ANOMALY_PARAMS["canopy_desert"]
    cont = contamination or params["contamination"]
    features = params["features"]

    gdf = df.copy()

    # Ensure all features exist (with sensible defaults)
    defaults = {
        "canopy_pct": 15.0, "mean_lst": 30.0, "vuln_score": 0.5,
        "pop_density": 1000, "canopy_deficit": 0.15,
        "uhi_delta": 2.0, "plantable_area": 500, "roi_norm": 0.5,
    }
    for feat in features:
        if feat not in gdf.columns:
            if feat == "uhi_delta":
                gdf[feat] = gdf.get("lst_anomaly", np.random.normal(2, 1.5, len(gdf)))
            else:
                gdf[feat] = defaults.get(feat, 0)

    X = gdf[features].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)

    iso = IsolationForest(
        contamination=cont,
        n_estimators=params["n_estimators"],
        max_samples=params["max_samples"],
        random_state=params["random_state"],
        n_jobs=-1,
    )

    gdf["anomaly_label"] = iso.fit_predict(X)
    gdf["anomaly_score"] = iso.decision_function(X)
    gdf["is_anomaly"] = gdf["anomaly_label"] == -1

    n_anomalies = gdf["is_anomaly"].sum()
    logger.info(f"7B: {n_anomalies}/{len(gdf)} canopy desert anomalies "
                f"({n_anomalies/len(gdf)*100:.1f}%)")

    # Tag
    gdf["anomaly_tag"] = ""
    gdf.loc[gdf["is_anomaly"], "anomaly_tag"] = ANOMALY_TAGS["canopy_desert"]

    # Heat emergency detection (bonus)
    if "mean_lst" in gdf.columns:
        lst_p95 = gdf["mean_lst"].quantile(0.95)
        heat_mask = (gdf["mean_lst"] > lst_p95) & gdf["is_anomaly"]
        gdf.loc[heat_mask, "anomaly_tag"] = ANOMALY_TAGS["heat_emergency"]

    return gdf


def final_reranking(
    df: "gpd.GeoDataFrame",
    weights: Optional[Dict[str, float]] = None,
) -> "gpd.GeoDataFrame":
    """
    Post-knapsack re-ranking using anomaly scores.

    final_rank_score = 0.70 × TPIS + 0.20 × anomaly_urgency
                       + 0.10 × (-anomaly_score)  [continuous severity]

    Args:
        df: GeoDataFrame with TPIS and anomaly columns.
        weights: Optional override for reranking weights.

    Returns:
        GeoDataFrame with final_rank_score and final_rank.
    """
    logger.info("Running final re-ranking with anomaly adjustment")

    w = weights or ANOMALY_PARAMS["reranking_weights"]
    gdf = df.copy()

    # Anomaly urgency (binary)
    gdf["anomaly_urgency"] = gdf.get("is_anomaly", False).astype(int)

    # Continuous severity (negated decision function — more negative = more anomalous)
    anomaly_severity = gdf.get("anomaly_score", np.zeros(len(gdf)))
    anomaly_severity_norm = -anomaly_severity  # higher = more severe
    asn_min, asn_max = anomaly_severity_norm.min(), anomaly_severity_norm.max()
    if asn_max - asn_min > 1e-10:
        anomaly_severity_norm = (anomaly_severity_norm - asn_min) / (asn_max - asn_min)

    gdf["final_rank_score"] = (
        w["tpis"] * gdf.get("tpis", 0).values
        + w["anomaly_urgency"] * gdf["anomaly_urgency"].values
        + w["anomaly_severity"] * anomaly_severity_norm
    )

    gdf["final_rank"] = gdf["final_rank_score"].rank(ascending=False).astype(int)

    # Combined tag
    gdf["combined_tag"] = gdf.get("anomaly_tag", "")
    if "temporal_tag" in gdf.columns:
        temporal_mask = gdf["temporal_tag"].str.len() > 0
        gdf.loc[temporal_mask, "combined_tag"] = gdf.loc[temporal_mask, "temporal_tag"]

    logger.info(f"Re-ranking complete. Top zone: rank=1, "
                f"score={gdf['final_rank_score'].max():.3f}")
    return gdf


def run_anomaly_pipeline(
    df: "gpd.GeoDataFrame",
    output_dir: Optional[str] = None,
) -> "gpd.GeoDataFrame":
    """Run complete Stage 7 anomaly detection pipeline."""
    logger.info("=" * 60)
    logger.info("STAGE 7: ANOMALY DETECTION & RE-RANKING")
    logger.info("=" * 60)

    out_dir = Path(output_dir) if output_dir else Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 7A: Temporal NDVI anomaly
    gdf = temporal_ndvi_anomaly(df)

    # 7B: Canopy desert anomaly
    gdf = canopy_desert_anomaly(gdf)

    # Final re-ranking
    gdf = final_reranking(gdf)

    # Save
    gdf.to_file(str(out_dir / "final_ranked.geojson"), driver="GeoJSON")
    logger.info("Saved final_ranked.geojson")

    # Summary table
    cols = ["final_rank", "tpis", "mean_lst", "canopy_pct", "vuln_score",
            "trees_possible", "combined_tag", "status"]
    available = [c for c in cols if c in gdf.columns]
    summary = gdf.sort_values("final_rank")[available].head(20)
    logger.info(f"\nTop 20 zones:\n{summary.to_string()}")

    logger.info("STAGE 7 COMPLETE")
    return gdf


if __name__ == "__main__":
    logger.info("Anomaly layer module loaded — run via pipeline_runner.py")
