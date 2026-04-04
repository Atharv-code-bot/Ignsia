"""
TEORA v3.0 — FastAPI Backend API
==================================
RESTful API serving the TEORA pipeline and dashboard data.

Endpoints:
  POST /analyze     → Run pipeline on AOI
  GET  /map-layers  → Raster tile info
  GET  /zones       → All zones GeoJSON
  GET  /zones/{id}  → Single zone detail
  POST /reoptimize  → Live re-optimization
  GET  /pareto      → Budget efficiency curve
  GET  /health      → Health check
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError("FastAPI not installed. Install: pip install fastapi uvicorn")

try:
    import geopandas as gpd
except ImportError:
    gpd = None

from config.settings import (
    API_HOST, API_PORT, CORS_ORIGINS, TPIS_WEIGHTS,
    MAX_TREES, OUTPUT_DIR, setup_logging,
)

logger = setup_logging("teora.api")

# ─── APP INITIALIZATION ──────────────────────────────────────

app = FastAPI(
    title="TEORA v3.0 API",
    description="Tree Equity Optimized Resource Allocator — Geospatial Intelligence API",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for pipeline results
pipeline_store: Dict[str, Any] = {
    "zones": None,
    "pareto": None,
    "status": "idle",
    "last_run": None,
}


# ─── REQUEST/RESPONSE MODELS ─────────────────────────────────

class AnalyzeRequest(BaseModel):
    """Request body for /analyze endpoint."""
    aoi: dict = Field(..., description="GeoJSON polygon for area of interest")
    neighborhoods: Optional[dict] = Field(None, description="Neighborhood polygons GeoJSON")
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="TPIS weight overrides {canopy_deficit, thermal_stress, vulnerability, plantability, roi_norm}",
    )
    budget: int = Field(default=1000, description="Maximum tree planting budget")
    start_date: str = Field(default="2023-01-01", description="Analysis start date")
    end_date: str = Field(default="2023-12-31", description="Analysis end date")


class ReoptimizeRequest(BaseModel):
    """Request body for /reoptimize endpoint."""
    weights: Dict[str, float] = Field(
        ..., description="New TPIS weights",
    )
    budget: int = Field(default=1000, description="New budget constraint")
    anomaly_contamination: Optional[float] = Field(
        default=None, description="Anomaly IF contamination threshold",
    )


class ZoneResponse(BaseModel):
    """Single zone detail response."""
    zone_id: str
    name: str
    tpis: float
    final_rank: int
    mean_ndvi: float
    mean_lst: float
    canopy_pct: float
    bare_pct: float
    vuln_score: float
    trees_possible: int
    plantable_area: float
    roi_total: float
    anomaly_tag: str
    status: str
    selected: bool
    geometry: dict


# ─── DEMO DATA GENERATOR ─────────────────────────────────────

def generate_demo_zones(n_zones: int = 20) -> List[dict]:
    """Generate synthetic zone data for demo/testing."""
    zones = []
    np.random.seed(42)

    for i in range(n_zones):
        lat = 28.60 + np.random.uniform(0, 0.08)
        lon = 77.18 + np.random.uniform(0, 0.10)
        size = 0.015 + np.random.uniform(0, 0.01)

        canopy = np.random.uniform(3, 35)
        lst = np.random.uniform(28, 42)
        vuln = np.random.uniform(0.1, 0.95)
        ndvi = np.random.uniform(0.05, 0.55)
        trees = int(np.random.uniform(20, 300))
        area = np.random.uniform(5000, 50000)
        tpis = np.random.uniform(0.2, 0.98)
        roi = np.random.uniform(5000, 80000)

        anomaly_tags = ["", "", "", "", ANOMALY_TAGS_LIST[i % len(ANOMALY_TAGS_LIST)]]
        tag = np.random.choice(anomaly_tags)
        selected = np.random.random() > 0.4

        zone = {
            "type": "Feature",
            "properties": {
                "zone_id": f"zone_{i:03d}",
                "name": f"Ward {i + 1}",
                "tpis": round(tpis, 3),
                "final_rank": i + 1,
                "mean_ndvi": round(ndvi, 3),
                "mean_lst": round(lst, 1),
                "canopy_pct": round(canopy, 1),
                "bare_pct": round(100 - canopy - np.random.uniform(20, 60), 1),
                "vuln_score": round(vuln, 3),
                "trees_possible": trees,
                "plantable_area": round(area, 0),
                "roi_total": round(roi, 0),
                "roi_cooling": round(roi * 0.3, 0),
                "roi_carbon": round(roi * 0.25, 0),
                "roi_air_quality": round(roi * 0.25, 0),
                "roi_stormwater": round(roi * 0.2, 0),
                "anomaly_tag": tag,
                "status": "✅ GO" if selected else "⚠️ H₂O",
                "selected": selected,
                "water_feasible": np.random.random() > 0.2,
                "thermal_stress": round(np.random.uniform(0.1, 0.9), 3),
                "canopy_deficit": round(max(0, 0.30 - canopy/100), 3),
                "poverty_proxy": round(np.random.uniform(0.1, 0.9), 3),
                "health_burden": round(np.random.uniform(0.1, 0.8), 3),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[lon, lat], [lon+size, lat],
                                [lon+size, lat+size], [lon, lat+size],
                                [lon, lat]]],
            },
        }
        zones.append(zone)

    # Sort by TPIS and assign ranks
    zones.sort(key=lambda z: z["properties"]["tpis"], reverse=True)
    for i, z in enumerate(zones):
        z["properties"]["final_rank"] = i + 1

    return zones


ANOMALY_TAGS_LIST = ["ANOMALY — URGENT", "ACTIVE LOSS", "HEAT EMERGENCY", ""]


# ─── ENDPOINTS ────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "TEORA v3.0",
        "timestamp": datetime.utcnow().isoformat(),
        "pipeline_status": pipeline_store["status"],
    }


@app.post("/analyze")
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Run full TEORA analysis pipeline.

    Input: AOI polygon + weights + budget
    Output: Ranked zones + selected zones
    """
    logger.info(f"Analyze request: budget={request.budget}")

    pipeline_store["status"] = "running"
    pipeline_store["last_run"] = datetime.utcnow().isoformat()

    try:
        # For demo: generate synthetic results
        zones = generate_demo_zones(25)

        # Apply budget/weight logic
        if request.weights:
            for z in zones:
                p = z["properties"]
                w = request.weights
                p["tpis"] = round(
                    w.get("canopy_deficit", 0.25) * p.get("canopy_deficit", 0) / 0.3 +
                    w.get("thermal_stress", 0.25) * p.get("thermal_stress", 0.5) +
                    w.get("vulnerability", 0.25) * p.get("vuln_score", 0.5) +
                    w.get("plantability", 0.15) * 0.5 +
                    w.get("roi_norm", 0.10) * 0.5
                , 3)

        # Re-rank by TPIS
        zones.sort(key=lambda z: z["properties"]["tpis"], reverse=True)
        for i, z in enumerate(zones):
            z["properties"]["final_rank"] = i + 1

        # Budget constraint: mark selected
        total = 0
        for z in zones:
            trees = z["properties"]["trees_possible"]
            if total + trees <= request.budget and z["properties"].get("water_feasible", True):
                z["properties"]["selected"] = True
                z["properties"]["status"] = "✅ GO"
                total += trees
            else:
                z["properties"]["selected"] = False

        geojson = {"type": "FeatureCollection", "features": zones}
        pipeline_store["zones"] = geojson
        pipeline_store["status"] = "complete"

        selected_zones = [z for z in zones if z["properties"]["selected"]]
        total_trees = sum(z["properties"]["trees_possible"] for z in selected_zones)
        total_impact = sum(z["properties"]["tpis"] * z["properties"]["trees_possible"]
                          for z in selected_zones)

        return {
            "status": "success",
            "summary": {
                "total_zones": len(zones),
                "selected_zones": len(selected_zones),
                "total_trees": total_trees,
                "budget_utilization": round(total_trees / max(request.budget, 1) * 100, 1),
                "total_impact": round(total_impact, 2),
            },
            "zones": geojson,
        }

    except Exception as e:
        pipeline_store["status"] = "error"
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(500, detail=str(e))


@app.get("/map-layers")
async def get_map_layers():
    """Returns available raster tile layer metadata."""
    layers = [
        {"id": "ndvi", "name": "NDVI Heatmap", "type": "raster",
         "colormap": "RdYlGn", "range": [-0.2, 0.8],
         "description": "Normalized Difference Vegetation Index"},
        {"id": "lst", "name": "Land Surface Temperature", "type": "raster",
         "colormap": "coolwarm", "range": [25, 45],
         "description": "LST in °C (TsHARP downscaled to 10m)"},
        {"id": "canopy", "name": "Canopy Coverage", "type": "choropleth",
         "colormap": "Greens", "range": [0, 50],
         "description": "Tree canopy percentage per zone"},
        {"id": "vulnerability", "name": "Social Vulnerability", "type": "choropleth",
         "colormap": "Purples", "range": [0, 1],
         "description": "Composite vulnerability index"},
        {"id": "tpis", "name": "TPIS Score", "type": "choropleth",
         "colormap": "YlOrRd", "range": [0, 1],
         "description": "Tree Planting Impact Score"},
        {"id": "segmentation", "name": "Land Cover", "type": "categorical",
         "classes": {"0": "Canopy", "1": "Built-up", "2": "Bare Land"},
         "description": "3-class segmentation mask"},
        {"id": "selected", "name": "Selected Zones", "type": "overlay",
         "description": "Budget-optimized intervention zones"},
        {"id": "water", "name": "Water Infrastructure", "type": "overlay",
         "colormap": "Blues", "description": "150m water buffer zones"},
        {"id": "anomaly", "name": "Anomaly Flags", "type": "overlay",
         "description": "IF-detected anomalous zones"},
    ]
    return {"layers": layers}


@app.get("/zones")
async def get_zones():
    """Returns GeoJSON of all analyzed zones."""
    if pipeline_store["zones"] is None:
        out_path = Path("output/final_output.geojson")
        if out_path.exists():
            with open(out_path, encoding='utf-8') as f:
                pipeline_store["zones"] = json.load(f)
                pipeline_store["zones"]["is_real_data"] = True
            logger.info("✅ Loaded real pipeline output")
            return pipeline_store["zones"]
            
        # Return demo data with flag
        logger.warning("⚠️ Using demo data - no real pipeline output found")
        zones = generate_demo_zones(20)
        return {
            "type": "FeatureCollection",
            "features": zones,
            "is_real_data": False,
            "warning": "This is demo/test data. Run the pipeline to get real results."
        }
    return pipeline_store["zones"]


@app.get("/zones/{zone_id}")
async def get_zone_detail(zone_id: str):
    """Returns detailed info for a single zone."""
    zones = pipeline_store.get("zones", {}).get("features", [])
    if not zones:
        zones = generate_demo_zones(20)

    for z in zones:
        if z["properties"].get("zone_id") == zone_id:
            return z

    raise HTTPException(404, detail=f"Zone {zone_id} not found")


@app.post("/reoptimize")
async def reoptimize(request: ReoptimizeRequest):
    """Re-run optimization with new weights and budget."""
    logger.info(f"Re-optimization: budget={request.budget}, weights={request.weights}")

    zones = pipeline_store.get("zones", {}).get("features", [])
    if not zones:
        zones = generate_demo_zones(20)

    # Recompute TPIS with new weights
    w = request.weights
    for z in zones:
        p = z["properties"]
        p["tpis"] = round(
            w.get("canopy_deficit", 0.25) * p.get("canopy_deficit", 0.15) / 0.3 +
            w.get("thermal_stress", 0.25) * p.get("thermal_stress", 0.5) +
            w.get("vulnerability", 0.25) * p.get("vuln_score", 0.5) +
            w.get("plantability", 0.15) * 0.5 +
            w.get("roi_norm", 0.10) * 0.5
        , 3)

    # Re-rank
    zones.sort(key=lambda z: z["properties"]["tpis"], reverse=True)
    total = 0
    for i, z in enumerate(zones):
        z["properties"]["final_rank"] = i + 1
        trees = z["properties"]["trees_possible"]
        if total + trees <= request.budget and z["properties"].get("water_feasible", True):
            z["properties"]["selected"] = True
            z["properties"]["status"] = "✅ GO"
            total += trees
        else:
            z["properties"]["selected"] = False
            z["properties"]["status"] = "⚠️ H₂O" if not z["properties"].get("water_feasible") else "❌ Budget"

    geojson = {"type": "FeatureCollection", "features": zones}
    pipeline_store["zones"] = geojson

    selected = [z for z in zones if z["properties"]["selected"]]
    return {
        "status": "success",
        "selected_count": len(selected),
        "total_trees": sum(z["properties"]["trees_possible"] for z in selected),
        "zones": geojson,
    }


@app.get("/pareto")
async def get_pareto_curve():
    """Returns Pareto budget efficiency frontier data."""
    if pipeline_store.get("pareto"):
        return {"pareto": pipeline_store["pareto"]}

    # Generate demo pareto curve
    np.random.seed(42)
    curve = []
    for budget in range(100, 2100, 100):
        impact = budget * 0.8 * (1 - np.exp(-budget / 800))
        zones_sel = min(int(budget / 80), 25)
        curve.append({
            "budget": budget,
            "total_impact": round(float(impact), 2),
            "zones_selected": zones_sel,
            "total_trees": min(budget, 2000),
        })

    return {"pareto": curve}


@app.get("/export/geojson")
async def export_geojson():
    """Export selected zones as GeoJSON file."""
    zones = pipeline_store.get("zones")
    if zones is None:
        zones = {"type": "FeatureCollection", "features": generate_demo_zones(20)}

    selected = {
        "type": "FeatureCollection",
        "features": [z for z in zones["features"] if z["properties"].get("selected")]
    }
    return selected


@app.get("/export/csv")
async def export_csv():
    """Export ranked intervention table as JSON array (CSV-ready)."""
    zones = pipeline_store.get("zones", {}).get("features", generate_demo_zones(20))
    table = []
    for z in sorted(zones, key=lambda x: x["properties"].get("final_rank", 99)):
        p = z["properties"]
        table.append({
            "Rank": p.get("final_rank"),
            "Zone": p.get("name"),
            "TPIS": p.get("tpis"),
            "LST_C": p.get("mean_lst"),
            "Canopy_Pct": p.get("canopy_pct"),
            "Vuln": p.get("vuln_score"),
            "Trees": p.get("trees_possible"),
            "Anomaly": p.get("anomaly_tag", ""),
            "Status": p.get("status"),
        })
    return {"table": table}


# ─── STARTUP ──────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("🌳 TEORA v3.0 API starting")
    logger.info(f"Docs: http://{API_HOST}:{API_PORT}/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )
