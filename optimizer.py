"""
TEORA v3.0 — Stage 6: Multi-Constraint Knapsack Optimization
==============================================================
0/1 Knapsack using Google OR-Tools CP-SAT solver.
Maximizes total tree planting impact under budget constraints.
Includes Pareto budget sweep for efficiency frontier.

CHANGES (v3.1):
  - Knapsack value now uses `tpis_final` (water-aware score from Stage 4)
    instead of raw `tpis`. Falls back to `tpis` if column is absent.

  - Value formula changed to prevent large-zone bias:

        PREVIOUS (biased):
            value = tpis × trees_possible
            → A low-quality zone with 300 trees beats a high-quality
              zone with 50 trees purely due to size.

        NEW (balanced):
            value = (0.7 × tpis_final + 0.3 × trees_norm) × trees_possible

            where trees_norm = trees_possible / max(trees_possible across zones)

        This blends absolute tree count with quality density so the
        optimizer rewards both high impact AND capacity, without letting
        large but mediocre zones dominate.

  - _greedy_fallback updated with the same value formula.
  - pareto_sweep unchanged in logic; inherits the fix automatically.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from ortools.sat.python import cp_model
except ImportError:
    cp_model = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None

from config.settings import MAX_TREES, OPTIMIZER_PARAMS, setup_logging

logger = setup_logging("teora.optimizer")

# Weights for the balanced value formula
_TPIS_WEIGHT  = 0.7   # quality component
_TREES_WEIGHT = 0.3   # capacity component


def _compute_balanced_values(
    tpis_final: np.ndarray,
    trees_vals: np.ndarray,
    scaling: int,
) -> np.ndarray:
    """
    Compute integer-scaled knapsack values using the balanced formula.

    value_i = (0.7 × tpis_final_i + 0.3 × trees_norm_i) × trees_possible_i × scaling

    Normalising trees_possible prevents large zones from dominating purely
    because of their capacity, while still rewarding zones that can absorb
    more trees if they are also high quality.

    Args:
        tpis_final: Per-zone tpis_final array  ∈ [0, 1].
        trees_vals: Per-zone trees_possible array (integer counts).
        scaling:    Integer scaling factor (e.g. 1000) for CP-SAT.

    Returns:
        Integer array of CP-SAT values.
    """
    max_trees = np.max(trees_vals) if np.max(trees_vals) > 0 else 1
    trees_norm = trees_vals.astype(float) / max_trees

    balanced_score = _TPIS_WEIGHT * tpis_final + _TREES_WEIGHT * trees_norm
    values = (balanced_score * trees_vals * scaling).astype(int)
    return values


def knapsack_optimize(
    feasible_zones: "gpd.GeoDataFrame",
    max_trees: int = None,
    timeout_sec: float = None,
) -> Dict[str, Any]:
    """
    Multi-constraint 0/1 Knapsack optimization using CP-SAT.

    Objective:
        Maximize Σ balanced_value_i × x_i
        where balanced_value_i = (0.7×tpis_final_i + 0.3×trees_norm_i) × trees_possible_i

    Constraint:
        Σ (trees_possible_i × x_i) ≤ T_max

    Uses `tpis_final` (water-aware score) if available, otherwise falls
    back to `tpis` with a logged warning.

    Args:
        feasible_zones: GeoDataFrame of feasible zones with tpis_final.
        max_trees: Maximum total trees budget constraint.
        timeout_sec: Solver timeout in seconds.

    Returns:
        Dict with selected zones, total impact, budget utilization.
    """
    logger.info("=" * 60)
    logger.info("STAGE 6: KNAPSACK OPTIMIZATION")
    logger.info("=" * 60)

    T_max   = max_trees or MAX_TREES
    timeout = timeout_sec or OPTIMIZER_PARAMS["solver_timeout_sec"]
    scaling = OPTIMIZER_PARAMS["integer_scaling"]

    if cp_model is None:
        logger.warning("OR-Tools not installed — using greedy fallback")
        return _greedy_fallback(feasible_zones, T_max)

    N = len(feasible_zones)
    if N == 0:
        logger.warning("No feasible zones to optimize")
        return {"selected": [], "total_impact": 0, "total_trees": 0, "selected_indices": []}

    logger.info(f"Optimizing {N} zones with budget T_max={T_max} trees")

    zones = feasible_zones.reset_index(drop=True)

    # ── Score column selection ────────────────────────────────
    if "tpis_final" in zones.columns:
        tpis_vals = zones["tpis_final"].values
        logger.info("Using tpis_final (water-aware score) as knapsack value base")
    else:
        tpis_vals = zones["tpis"].values
        logger.warning(
            "tpis_final column not found — falling back to raw tpis. "
            "Run Stage 4 with water_gdf to get water-aware scores."
        )

    trees_vals = zones["trees_possible"].values.astype(int)

    # ── Balanced value formula ────────────────────────────────
    values = _compute_balanced_values(tpis_vals, trees_vals, scaling)
    logger.info(
        f"Value formula: 0.7×tpis_final + 0.3×trees_norm × trees × {scaling}  "
        f"(range {values.min()}–{values.max()})"
    )

    # ── Build CP-SAT model ────────────────────────────────────
    model_cp = cp_model.CpModel()
    x = [model_cp.NewBoolVar(f"x_{i}") for i in range(N)]

    model_cp.Maximize(
        cp_model.LinearExpr.WeightedSum(x, values.tolist())
    )
    model_cp.Add(
        cp_model.LinearExpr.WeightedSum(x, trees_vals.tolist()) <= T_max
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_search_workers  = 4

    logger.info(f"Solving CP-SAT (timeout={timeout}s)...")
    status = solver.Solve(model_cp)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected_idx  = [i for i in range(N) if solver.Value(x[i]) == 1]
        total_impact  = solver.ObjectiveValue() / scaling
        total_trees   = int(sum(trees_vals[i] for i in selected_idx))
        utilization   = total_trees / T_max * 100

        status_str = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
        logger.info(f"Solution: {status_str}")
        logger.info(f"Selected {len(selected_idx)}/{N} zones")
        logger.info(f"Total trees: {total_trees}/{T_max} ({utilization:.1f}% budget)")
        logger.info(f"Total impact: {total_impact:.2f}")

        zones["selected"]      = False
        zones.loc[selected_idx, "selected"] = True
        zones["knapsack_rank"] = 0
        zones.loc[selected_idx, "knapsack_rank"] = range(1, len(selected_idx) + 1)

        return {
            "zones":              zones,
            "selected_indices":   selected_idx,
            "total_impact":       total_impact,
            "total_trees":        total_trees,
            "budget_utilization": utilization,
            "status":             status_str,
        }
    else:
        logger.warning(f"Solver status: {solver.StatusName(status)}")
        return _greedy_fallback(feasible_zones, T_max)


def _greedy_fallback(
    zones: "gpd.GeoDataFrame",
    max_trees: int,
) -> Dict[str, Any]:
    """
    Greedy knapsack approximation when CP-SAT is unavailable.

    Uses the same balanced value density as the CP-SAT path:
        density_i = 0.7×tpis_final_i + 0.3×trees_norm_i
    Selects zones in descending density order until budget exhausted.
    """
    logger.info("Using greedy fallback optimization")

    gdf   = zones.reset_index(drop=True).copy()
    trees = gdf["trees_possible"].values.astype(int)

    tpis_vals = (
        gdf["tpis_final"].values
        if "tpis_final" in gdf.columns
        else gdf["tpis"].values
    )

    max_t      = np.max(trees) if np.max(trees) > 0 else 1
    trees_norm = trees.astype(float) / max_t
    density    = _TPIS_WEIGHT * tpis_vals + _TREES_WEIGHT * trees_norm

    order    = np.argsort(-density)
    selected = []
    remaining = max_trees
    total_impact = 0.0

    for i in order:
        if trees[i] <= remaining and trees[i] > 0:
            selected.append(int(i))
            remaining    -= trees[i]
            total_impact += tpis_vals[i] * trees[i]

    total_trees = max_trees - remaining
    gdf["selected"]      = False
    gdf.loc[selected, "selected"] = True

    logger.info(f"Greedy: {len(selected)} zones, {total_trees} trees")
    return {
        "zones":              gdf,
        "selected_indices":   selected,
        "total_impact":       total_impact,
        "total_trees":        int(total_trees),
        "budget_utilization": total_trees / max(max_trees, 1) * 100,
        "status":             "GREEDY",
    }


def pareto_sweep(
    feasible_zones: "gpd.GeoDataFrame",
    max_budget: int = None,
    step: int = None,
) -> List[Dict[str, Any]]:
    """
    Pareto budget sweep for efficiency frontier.

    Solves the (fixed) knapsack for multiple budget levels to generate
    (budget, total_impact, zones_selected) curve.  The balanced value
    formula is used at every budget level automatically.

    Args:
        feasible_zones: GeoDataFrame of feasible zones.
        max_budget: Maximum budget to sweep to.
        step: Budget increment step.

    Returns:
        List of (budget, impact, zones_count, total_trees) dicts.
    """
    max_b = max_budget or MAX_TREES
    s     = step or OPTIMIZER_PARAMS["pareto_step"]

    logger.info(f"Running Pareto sweep: 0 → {max_b}, step={s}")

    pareto_curve = []
    for budget in range(s, max_b + 1, s):
        result = knapsack_optimize(feasible_zones, max_trees=budget, timeout_sec=10)
        pareto_curve.append({
            "budget":         budget,
            "total_impact":   result["total_impact"],
            "zones_selected": len(result["selected_indices"]),
            "total_trees":    result["total_trees"],
        })
        logger.debug(
            f"  Budget={budget}: impact={result['total_impact']:.2f}, "
            f"zones={len(result['selected_indices'])}"
        )

    logger.info(f"Pareto sweep complete: {len(pareto_curve)} points")
    return pareto_curve


if __name__ == "__main__":
    logger.info("Optimizer module loaded — run via pipeline_runner.py")