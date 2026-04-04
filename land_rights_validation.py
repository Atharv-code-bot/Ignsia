"""
TEORA v3.0 — Stage 3.5: Land Rights & Protected Area Validation
=================================================================
Filters zones by land ownership and conservation restrictions.
Prevents planting in protected areas, private land, or restricted zones.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import geopandas as gpd
    from shapely.ops import unary_union
except ImportError:
    gpd = None

from config.settings import setup_logging

logger = setup_logging("teora.land_rights_validation")


def load_protected_areas(protected_areas_path: str = None) -> Optional["gpd.GeoDataFrame"]:
    """
    Load protected areas (e.g., national parks, reserves).
    
    Expected format: GeoJSON with columns:
      - name (location name)
      - category (NATIONAL_PARK, NATURE_RESERVE, UNESCO_SITE, etc.)
      - protection_level (STRICT, MODERATE, BUFFER)
    
    Args:
        protected_areas_path: Path to protected areas GeoJSON.
    
    Returns:
        GeoDataFrame of protected areas or None.
    """
    if protected_areas_path is None:
        protected_areas_path = "data/protected_areas.geojson"
    
    path = Path(protected_areas_path)
    if not path.exists():
        logger.warning(f"Protected areas file not found: {path}")
        return None
    
    try:
        pa_gdf = gpd.read_file(str(path))
        logger.info(f"Loaded {len(pa_gdf)} protected areas")
        return pa_gdf
    except Exception as e:
        logger.error(f"Failed to load protected areas: {e}")
        return None


def load_land_ownership(ownership_path: str = None) -> Optional["gpd.GeoDataFrame"]:
    """
    Load cadastral/ownership data.
    
    Expected format: GeoJSON with columns:
      - owner_type (PUBLIC, PRIVATE, COMMUNAL, GOVERNMENT)
      - landuse (AGRICULTURAL, URBAN, FOREST, etc.)
      - status (AVAILABLE_FOR_PLANTING, RESTRICTED, PERMIT_REQUIRED)
    
    Args:
        ownership_path: Path to cadastral/ownership GeoJSON.
    
    Returns:
        GeoDataFrame of land parcels or None.
    """
    if ownership_path is None:
        ownership_path = "data/land_ownership.geojson"
    
    path = Path(ownership_path)
    if not path.exists():
        logger.warning(f"Land ownership file not found: {path}")
        return None
    
    try:
        own_gdf = gpd.read_file(str(path))
        logger.info(f"Loaded {len(own_gdf)} land ownership records")
        return own_gdf
    except Exception as e:
        logger.error(f"Failed to load land ownership: {e}")
        return None


def load_no_go_zones(nogo_path: str = None) -> Optional["gpd.GeoDataFrame"]:
    """
    Load no-go zones: military areas, cultural sites, sacred places.
    
    Expected format: GeoJSON with columns:
      - zone_type (MILITARY, CULTURAL_SITE, SACRED, RIPARIAN_BUFFER, etc.)
      - reason (explanation of restriction)
    
    Args:
        nogo_path: Path to no-go zones GeoJSON.
    
    Returns:
        GeoDataFrame of restricted zones or None.
    """
    if nogo_path is None:
        nogo_path = "data/no_go_zones.geojson"
    
    path = Path(nogo_path)
    if not path.exists():
        logger.warning(f"No-go zones file not found: {path}")
        return None
    
    try:
        nogo_gdf = gpd.read_file(str(path))
        logger.info(f"Loaded {len(nogo_gdf)} no-go zones")
        return nogo_gdf
    except Exception as e:
        logger.error(f"Failed to load no-go zones: {e}")
        return None


def validate_land_rights(
    zones: "gpd.GeoDataFrame",
    protected_areas: Optional["gpd.GeoDataFrame"] = None,
    land_ownership: Optional["gpd.GeoDataFrame"] = None,
    no_go_zones: Optional["gpd.GeoDataFrame"] = None,
) -> "gpd.GeoDataFrame":
    """
    Validate zones against land rights and protected areas.
    
    Process:
      1. Check if zone intersects protected area → BLOCKED
      2. Check land ownership type → FLAG if private/restricted
      3. Check if in no-go zone → BLOCKED
      4. Add permission_required flag if unclear
    
    Args:
        zones: GeoDataFrame of candidate zones.
        protected_areas: Protected areas GeoDataFrame.
        land_ownership: Land ownership GeoDataFrame.
        no_go_zones: No-go restricted zones.
    
    Returns:
        GeoDataFrame with land_rights_status column.
    """
    logger.info("=" * 60)
    logger.info("STAGE 3.5: LAND RIGHTS & PROTECTED AREA VALIDATION")
    logger.info("=" * 60)
    
    gdf = zones.copy()
    
    # Initialize columns
    gdf["land_rights_status"] = "✅ AVAILABLE"
    gdf["land_rights_reason"] = ""
    gdf["in_protected_area"] = False
    gdf["protected_area_name"] = ""
    gdf["owner_type"] = "UNKNOWN"
    gdf["in_no_go_zone"] = False
    gdf["no_go_reason"] = ""
    gdf["permission_required"] = False
    
    # ─── CHECK 1: Protected Areas ───
    if protected_areas is not None and len(protected_areas) > 0:
        logger.info(f"Checking {len(gdf)} zones against {len(protected_areas)} protected areas")
        
        # Ensure CRS alignment
        if gdf.crs and protected_areas.crs and str(gdf.crs) != str(protected_areas.crs):
            protected_areas = protected_areas.to_crs(gdf.crs)
        
        for idx, zone in gdf.iterrows():
            for pa_idx, pa in protected_areas.iterrows():
                if zone.geometry.intersects(pa.geometry):
                    gdf.at[idx, "in_protected_area"] = True
                    gdf.at[idx, "protected_area_name"] = pa.get("name", "Unknown")
                    gdf.at[idx, "land_rights_status"] = "❌ BLOCKED"
                    gdf.at[idx, "land_rights_reason"] = \
                        f"Protected area: {pa.get('name', 'Unknown')} ({pa.get('category', 'Reserve')})"
                    break
        
        blocked_pa = gdf["in_protected_area"].sum()
        logger.info(f"  {blocked_pa} zones in protected areas")
    
    # ─── CHECK 2: No-Go Zones ───
    if no_go_zones is not None and len(no_go_zones) > 0:
        logger.info(f"Checking against {len(no_go_zones)} no-go zones")
        
        if gdf.crs and no_go_zones.crs and str(gdf.crs) != str(no_go_zones.crs):
            no_go_zones = no_go_zones.to_crs(gdf.crs)
        
        for idx, zone in gdf.iterrows():
            if gdf.at[idx, "land_rights_status"] == "❌ BLOCKED":
                continue  # Already blocked
            
            for nogo_idx, nogo in no_go_zones.iterrows():
                if zone.geometry.intersects(nogo.geometry):
                    gdf.at[idx, "in_no_go_zone"] = True
                    gdf.at[idx, "no_go_reason"] = nogo.get("reason", "Restricted zone")
                    gdf.at[idx, "land_rights_status"] = "❌ BLOCKED"
                    gdf.at[idx, "land_rights_reason"] = \
                        f"No-go zone: {nogo.get('zone_type', 'Unknown')} — {nogo.get('reason', 'Restricted')}"
                    break
        
        blocked_nogo = gdf["in_no_go_zone"].sum()
        logger.info(f"  {blocked_nogo} zones in no-go zones")
    
    # ─── CHECK 3: Land Ownership ───
    if land_ownership is not None and len(land_ownership) > 0:
        logger.info(f"Checking against {len(land_ownership)} land ownership records")
        
        if gdf.crs and land_ownership.crs and str(gdf.crs) != str(land_ownership.crs):
            land_ownership = land_ownership.to_crs(gdf.crs)
        
        for idx, zone in gdf.iterrows():
            if gdf.at[idx, "land_rights_status"] == "❌ BLOCKED":
                continue  # Already blocked
            
            # Find intersecting land ownership parcel
            for own_idx, own in land_ownership.iterrows():
                if zone.geometry.intersects(own.geometry):
                    owner_type = own.get("owner_type", "UNKNOWN")
                    status = own.get("status", "AVAILABLE_FOR_PLANTING")
                    
                    gdf.at[idx, "owner_type"] = owner_type
                    
                    if owner_type == "PRIVATE":
                        gdf.at[idx, "permission_required"] = True
                        if status != "AVAILABLE_FOR_PLANTING":
                            gdf.at[idx, "land_rights_status"] = "⚠️ PERMIT_REQUIRED"
                            gdf.at[idx, "land_rights_reason"] = \
                                f"Private land — {status}"
                        else:
                            gdf.at[idx, "land_rights_status"] = "⚠️ CONTACT_OWNER"
                            gdf.at[idx, "land_rights_reason"] = \
                                "Private land — owner permission required"
                    
                    elif owner_type == "COMMUNAL":
                        gdf.at[idx, "permission_required"] = True
                        gdf.at[idx, "land_rights_status"] = "⚠️ COMMUNITY_APPROVAL"
                        gdf.at[idx, "land_rights_reason"] = \
                            "Communal land — community consent needed"
                    
                    elif status == "RESTRICTED":
                        gdf.at[idx, "land_rights_status"] = "❌ BLOCKED"
                        gdf.at[idx, "land_rights_reason"] = \
                            f"Land use restricted: {own.get('reason', 'Unknown')}"
                    
                    break
        
        private_land = (gdf["owner_type"] == "PRIVATE").sum()
        communal_land = (gdf["owner_type"] == "COMMUNAL").sum()
        logger.info(f"  {private_land} private, {communal_land} communal")
    
    # ─── SUMMARY ───
    available = (gdf["land_rights_status"] == "✅ AVAILABLE").sum()
    blocked = (gdf["land_rights_status"] == "❌ BLOCKED").sum()
    permit_required = (gdf["land_rights_status"].str.contains("⚠️", na=False)).sum()
    
    logger.info(f"Land rights validation result:")
    logger.info(f"  ✅ Available for planting: {available}/{len(gdf)}")
    logger.info(f"  ❌ Blocked: {blocked}/{len(gdf)}")
    logger.info(f"  ⚠️ Permit/approval required: {permit_required}/{len(gdf)}")
    
    logger.info("STAGE 3.5 COMPLETE")
    return gdf


def add_land_rights_to_knapsack(
    zones: "gpd.GeoDataFrame",
) -> "gpd.GeoDataFrame":
    """
    Adjust knapsack optimization by deprioritizing restricted zones.
    
    Approach:
      - Keep available zones at full TPIS weight
      - Reduce TPIS for zones needing permits (×0.7 penalty)
      - Exclude blocked zones entirely from optimization
    
    Args:
        zones: GeoDataFrame with land_rights_status column.
    
    Returns:
        GeoDataFrame with adjusted TPIS for knapsack.
    """
    logger.info("Adjusting TPIS scores by land rights status")
    
    gdf = zones.copy()
    gdf["tpis_adjusted"] = gdf["tpis"].copy()
    
    # Available: keep as-is
    available_mask = gdf["land_rights_status"] == "✅ AVAILABLE"
    
    # Permit required: reduce by 30% (still plantable but more complex)
    permit_mask = gdf["land_rights_status"].str.contains("⚠️", na=False)
    gdf.loc[permit_mask, "tpis_adjusted"] = gdf.loc[permit_mask, "tpis"] * 0.7
    logger.info(f"  Permit zones: TPIS reduced by 30% ({permit_mask.sum()} zones)")
    
    # Blocked: exclude from optimization
    blocked_mask = gdf["land_rights_status"] == "❌ BLOCKED"
    gdf.loc[blocked_mask, "tpis_adjusted"] = 0.0
    logger.info(f"  Blocked zones: Excluded from optimization ({blocked_mask.sum()} zones)")
    
    return gdf


if __name__ == "__main__":
    logger.info("Land rights validation module loaded — run via pipeline_runner.py")
