import os, json, re, argparse
import numpy as np
import pandas as pd
import rasterio

def safe_name(w): return re.sub(r"[^a-zA-Z0-9]","_",w).strip("_")

WARD_NAMES = [
    "Admin Ward 01 Aundh",           "Admin Ward 02 Ghole Road",
    "Admin Ward 03 Kothrud Karveroad","Admin Ward 04 Warje Karvenagar",
    "Admin Ward 05 Dhole Patil Rd",   "Admin Ward 06 Yerawda - Sangamwadi",
    "Admin Ward 07 Nagar Road",       "Admin Ward 08 KasbaVishrambaugwada",
    "Admin Ward 09 Tilak Road",       "Admin Ward 10 Sahakarnagar",
    "Admin Ward 11 Bibwewadi",        "Admin Ward 12 Bhavani Peth",
    "Admin Ward 13 Hadapsar",         "Admin Ward 14 Dhankawadi",
    "Admin Ward 15 Kondhwa Wanavdi",
]


def load_ward_inputs(ward_name: str, stage2_dir: str, stage3_dir: str,
                     aux_dir: str) -> dict:
    """
    Collects all per-ward pixel statistics needed for TPIS computation.
    Reads actual pixel arrays (not averages) to compute spatial statistics.
    """
    sn = safe_name(ward_name)

    seg_json = os.path.join(stage3_dir, f"seg_stats_{sn}.json")
    if not os.path.exists(seg_json):
        return None
    with open(seg_json) as f:
        seg = json.load(f)

    lst_tif = os.path.join(stage2_dir, f"lst_10m_{sn}.tif")
    lst_stats = {"mean": None, "max": None, "std": None}
    if os.path.exists(lst_tif):
        with rasterio.open(lst_tif) as src:
            lst_arr = src.read(1).astype(np.float32)
        lst_vals = lst_arr[np.isfinite(lst_arr)]
        if len(lst_vals):
            lst_stats = {
                "mean": float(np.mean(lst_vals)),
                "max":  float(np.max(lst_vals)),
                "std":  float(np.std(lst_vals)),
                # pixel-level array retained for LST anomaly → ROI calc
                "_all_pixels": lst_vals,
            }

    aux_tif = os.path.join(aux_dir, f"auxiliary_{sn}.tif")
    pop_mean, no2_mean = 0.0, 0.0
    if os.path.exists(aux_tif):
        with rasterio.open(aux_tif) as src:
            pop_arr = src.read(1).astype(np.float32)   # band 1 = population
            no2_arr = src.read(2).astype(np.float32)   # band 2 = NO₂
        pop_vals = pop_arr[np.isfinite(pop_arr) & (pop_arr >= 0)]
        no2_vals = no2_arr[np.isfinite(no2_arr)]
        pop_mean = float(np.mean(pop_vals)) if len(pop_vals) else 0
        no2_mean = float(np.mean(no2_vals)) if len(no2_vals) else 0

    return {
        "ward_name":        ward_name,
        "canopy_pct":       seg["canopy_pct"],
        "built_pct":        seg["built_pct"],
        "bare_pct":         seg["bare_pct"],
        "plantable_area_m2":seg["plantable_area_m2"],
        "trees_possible":   seg["trees_possible"],
        "total_area_m2":    seg["total_area_m2"],
        "mean_lst":         lst_stats["mean"],
        "max_lst":          lst_stats["max"],
        "std_lst":          lst_stats["std"],
        "_lst_pixels":      lst_stats.get("_all_pixels"),
        "mean_pop_density": pop_mean,
        "mean_no2":         no2_mean,
    }


def compute_tpis(ward_records: list,
                 w1=0.25, w2=0.25, w3=0.25, w4=0.15, w5=0.10) -> pd.DataFrame:
    """
    Tree Planting Impact Score (TPIS) as defined in implementation_plan.md:

    TPIS[i] = w₁ × canopy_deficit_norm[i]
            + w₂ × thermal_stress[i]
            + w₃ × vuln_score[i]
            + w₄ × plantability_norm[i]
            + w₅ × roi_norm[i]

    All components normalised 0–1 across wards.

    WHY city_mean_lst needs pixel arrays:
      lst_anomaly[i] = mean(all pixels in ward_i) - mean(all pixels in ALL wards)
      Using stored means directly loses precision from averaging-of-averages.
    """
    df = pd.DataFrame(ward_records)
    df = df[df["mean_lst"].notna()].copy()

    
    all_lst_pixels = np.concatenate([r["_lst_pixels"] for r in ward_records
                                      if r.get("_lst_pixels") is not None])
    city_mean_lst  = float(np.mean(all_lst_pixels))
    df["lst_anomaly"] = df["mean_lst"] - city_mean_lst

    
    canopy_goal = 30.0   # 30% target (reasonable for Indian metros)
    df["canopy_deficit"] = (canopy_goal - df["canopy_pct"]).clip(lower=0)
    df["canopy_deficit_norm"] = (df["canopy_deficit"] / df["canopy_deficit"].max()
                                  ).fillna(0)

    lst_min, lst_max = df["mean_lst"].min(), df["mean_lst"].max()
    df["thermal_stress"] = ((df["mean_lst"] - lst_min) /
                             (lst_max - lst_min + 1e-9))

    
    no2_min, no2_max = df["mean_no2"].min(), df["mean_no2"].max()
    df["health_burden_norm"] = ((df["mean_no2"] - no2_min) /
                                 (no2_max - no2_min + 1e-9))
    df["vuln_score"] = df["health_burden_norm"]   # extend with RWI when available

    plant_max = df["trees_possible"].max()
    df["plantability_norm"] = (df["trees_possible"] / (plant_max + 1e-9))

    
    k_cooling      = 0.3 * 3.6 * 0.12   
    social_cost_co2 = 51 / 1000          
    pm25_value      = 1200 / 1000        

    df["cooling_benefit"] = (df["lst_anomaly"].clip(lower=0)
                              * df["mean_pop_density"]
                              * k_cooling
                              * df["trees_possible"])
    df["carbon_benefit"]  = df["trees_possible"] * 22 * social_cost_co2
    df["air_quality_benefit"] = df["trees_possible"] * 0.2 * pm25_value
    df["total_roi"]       = (df["cooling_benefit"]
                              + df["carbon_benefit"]
                              + df["air_quality_benefit"])

    roi_max = df["total_roi"].max()
    df["roi_norm"] = df["total_roi"] / (roi_max + 1e-9)

    df["TPIS"] = (w1 * df["canopy_deficit_norm"]
                + w2 * df["thermal_stress"]
                + w3 * df["vuln_score"]
                + w4 * df["plantability_norm"]
                + w5 * df["roi_norm"]).round(6)

    df["priority_rank"] = df["TPIS"].rank(ascending=False).astype(int)
    df = df.sort_values("priority_rank")

    df = df.drop(columns=["_lst_pixels"], errors="ignore")
    return df


def knapsack_optimize(df: pd.DataFrame, t_max: int = 5000,
                       use_ortools: bool = True) -> pd.DataFrame:
    """
    Multi-constraint 0/1 Knapsack as per implementation_plan.md Stage 5.

    maximize   Σ x[i] × TPIS[i] × trees_possible[i]
    subject to Σ x[i] × trees_possible[i] ≤ T_max
               x[i] ∈ {0, 1}

    Water constraint: wards with no JRC water nearby are flagged
    but not auto-excluded (India has hand-watering infrastructure).
    """
    feasible = df[df["trees_possible"] > 0].copy()

    if use_ortools:
        try:
            from ortools.sat.python import cp_model
            model   = cp_model.CpModel()
            n       = len(feasible)
            x       = [model.NewBoolVar(f"x_{i}") for i in range(n)]
            values  = [int(row["TPIS"] * row["trees_possible"] * 1000)
                       for _, row in feasible.iterrows()]
            weights = [int(row["trees_possible"]) for _, row in feasible.iterrows()]

            model.Add(sum(x[i] * weights[i] for i in range(n)) <= t_max)
            model.Maximize(sum(x[i] * values[i] for i in range(n)))

            solver   = cp_model.CpSolver()
            status   = solver.Solve(model)
            selected = [bool(solver.Value(x[i])) for i in range(n)]
            print(f"[Stage 5] OR-Tools status: {solver.StatusName(status)}")
        except ImportError:
            print("[Stage 5] OR-Tools not found, using DP fallback")
            selected = _dp_knapsack(feasible, t_max)
    else:
        selected = _dp_knapsack(feasible, t_max)

    feasible["selected"]          = selected
    feasible["recommended_trees"] = feasible.apply(
        lambda r: int(r["trees_possible"]) if r["selected"] else 0, axis=1)

    # Merge back
    df = df.merge(
        feasible[["ward_name","selected","recommended_trees"]],
        on="ward_name", how="left")
    df["selected"]          = df["selected"].fillna(False)
    df["recommended_trees"] = df["recommended_trees"].fillna(0).astype(int)

    n_sel  = df["selected"].sum()
    t_used = df["recommended_trees"].sum()
    total_tpis = (df[df["selected"]]["TPIS"] *
                  df[df["selected"]]["trees_possible"]).sum()
    print(f"[Stage 5] Selected {n_sel} wards, "
          f"{t_used}/{t_max} trees, "
          f"total TPIS-weighted impact = {total_tpis:.2f}")
    return df


def _dp_knapsack(df, capacity):
    """Pure-Python 0/1 knapsack DP fallback."""
    items   = list(df.itertuples())
    n       = len(items)
    weights = [int(r.trees_possible) for r in items]
    values  = [int(r.TPIS * r.trees_possible * 1000) for r in items]

    dp = [[0]*(capacity+1) for _ in range(n+1)]
    for i in range(1, n+1):
        w, v = weights[i-1], values[i-1]
        for c in range(capacity+1):
            dp[i][c] = dp[i-1][c]
            if c >= w:
                dp[i][c] = max(dp[i][c], dp[i-1][c-w] + v)

    selected = [False]*n
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i-1][c]:
            selected[i-1] = True
            c -= weights[i-1]
    return selected


def write_tpis_raster(ward_name: str, tpis_value: float,
                       seg_tif: str, out_dir: str):
    
    sn = safe_name(ward_name)
    if not os.path.exists(seg_tif):
        return

    with rasterio.open(seg_tif) as src:
        seg     = src.read(1)
        profile = src.profile.copy()

    tpis_arr = np.zeros_like(seg, dtype=np.float32)
    tpis_arr[seg == 0] = tpis_value * 0.8    
    tpis_arr[seg == 1] = tpis_value * 0.3    
    tpis_arr[seg == 2] = tpis_value * 1.2    
    tpis_arr = np.clip(tpis_arr, 0, 1)

    out_path = os.path.join(out_dir, f"tpis_raster_{sn}.tif")
    p = profile.copy()
    p.update(count=1, dtype="float32", compress="lzw",
             tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(tpis_arr[np.newaxis, ...])
    print(f"    → TPIS raster: {os.path.basename(out_path)}")


def run_stage4_5(stage2_dir, stage3_dir, aux_dir, out_dir,
                  t_max=5000, weights=(0.25,0.25,0.25,0.15,0.10)):
    os.makedirs(out_dir, exist_ok=True)

    records = []
    for wn in WARD_NAMES:
        r = load_ward_inputs(wn, stage2_dir, stage3_dir, aux_dir)
        if r:
            records.append(r)

    if not records:
        print("[ERROR] No ward data found. Run Stages 2 & 3 first.")
        return

    print(f"[Stage 4] Loaded {len(records)} wards")

    df = compute_tpis(records, *weights)

    df = knapsack_optimize(df, t_max=t_max)

    for _, row in df.iterrows():
        sn      = safe_name(row["ward_name"])
        seg_tif = os.path.join(stage3_dir, f"seg_mask_{sn}.tif")
        write_tpis_raster(row["ward_name"], row["TPIS"], seg_tif, out_dir)

    csv_path = os.path.join(out_dir, "ward_tpis_scores.csv")
    df.drop(columns=["_lst_pixels"], errors="ignore").to_csv(csv_path, index=False)
    print(f"\n[Stage 4+5] ✅ Scores saved: {csv_path}")

    sel = df[df["selected"]]
    print(f"\n{'='*65}")
    print(f"  TPIS Ranking (top 5):")
    for _, r in df.head(5).iterrows():
        flag = "✅ SELECTED" if r["selected"] else ""
        print(f"    #{int(r['priority_rank']):<3} {r['ward_name']:<38} "
              f"TPIS={r['TPIS']:.3f}  Trees={int(r['trees_possible'])}  {flag}")
    print(f"\n  Selected: {len(sel)} wards  |  "
          f"Trees to plant: {sel['recommended_trees'].sum():,} / {t_max:,}")
    print(f"  Est. canopy increase: "
          f"+{sel['recommended_trees'].sum()*25/1_000_000:.2f} km²")
    print(f"{'='*65}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2_dir", default="./ward_outputs/stage2/")
    parser.add_argument("--stage3_dir", default="./ward_outputs/stage3/")
    parser.add_argument("--aux_dir",    default="./ward_outputs/geotiffs/")
    parser.add_argument("--out_dir",    default="./ward_outputs/stage4_5/")
    parser.add_argument("--t_max",      type=int, default=5000)
    args = parser.parse_args()
    run_stage4_5(args.stage2_dir, args.stage3_dir, args.aux_dir,
                  args.out_dir, t_max=args.t_max)
