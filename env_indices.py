"""
TEORA v3.0 — Stage 6: Multi-Constraint Knapsack Optimization
==============================================================
0/1 Knapsack using Google OR-Tools CP-SAT solver.
Maximizes total tree planting impact under budget constraints.
Includes Pareto budget sweep for efficiency frontier.
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


def knapsack_optimize(
    feasible_zones: "gpd.GeoDataFrame",
    max_trees: int = None,
    timeout_sec: float = None,
) -> Dict[str, Any]:
    """
    Multi-constraint 0/1 Knapsack optimization using CP-SAT.

    Objective: Maximize Σ (TPIS_i × trees_possible_i × x_i)
    Constraint: Σ (trees_possible_i × x_i) ≤ T_max

    Args:
        feasible_zones: GeoDataFrame of feasible zones with TPIS.
        max_trees: Maximum total trees budget constraint.
        timeout_sec: Solver timeout in seconds.

    Returns:
        Dict with selected zones, total impact, budget utilization.
    """
    logger.info("=" * 60)
    logger.info("STAGE 6: KNAPSACK OPTIMIZATION")
    logger.info("=" * 60)

    T_max = max_trees or MAX_TREES
    timeout = timeout_sec or OPTIMIZER_PARAMS["solver_timeout_sec"]
    scaling = OPTIMIZER_PARAMS["integer_scaling"]

    if cp_model is None:
        logger.warning("OR-Tools not installed — using greedy fallback")
        return _greedy_fallback(feasible_zones, T_max)

    N = len(feasible_zones)
    if N == 0:
        logger.warning("No feasible zones to optimize")
        return {"selected": [], "total_impact": 0, "total_trees": 0}

    logger.info(f"Optimizing {N} zones with budget T_max={T_max} trees")

    # Prepare data
    zones = feasible_zones.reset_index(drop=True)
    tpis_vals = zones["tpis"].values
    trees_vals = zones["trees_possible"].values.astype(int)

    # Integer-scaled value: TPIS × trees_possible × 1000
    values = (tpis_vals * trees_vals * scaling).astype(int)

    # Build CP-SAT model
    model_cp = cp_model.CpModel()
    x = [model_cp.NewBoolVar(f"x_{i}") for i in range(N)]

    # Maximize total impact
    model_cp.Maximize(
        cp_model.LinearExpr.WeightedSum(x, values.tolist())
    )

    # Budget constraint
    model_cp.Add(
        cp_model.LinearExpr.WeightedSum(x, trees_vals.tolist()) <= T_max
    )

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_search_workers = 4

    logger.info(f"Solving CP-SAT (timeout={timeout}s)...")
    status = solver.Solve(model_cp)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected_idx = [i for i in range(N) if solver.Value(x[i]) == 1]
        total_impact = solver.ObjectiveValue() / scaling
        total_trees = sum(trees_vals[i] for i in selected_idx)
        utilization = total_trees / T_max * 100

        status_str = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
        logger.info(f"Solution: {status_str}")
        logger.info(f"Selected {len(selected_idx)}/{N} zones")
        logger.info(f"Total trees: {total_trees}/{T_max} ({utilization:.1f}% budget)")
        logger.info(f"Total impact: {total_impact:.2f}")

        # Mark selected zones
        zones["selected"] = False
        zones.loc[selected_idx, "selected"] = True
        zones["knapsack_rank"] = 0
        zones.loc[selected_idx, "knapsack_rank"] = range(1, len(selected_idx) + 1)

        return {
            "zones": zones,
            "selected_indices": selected_idx,
            "total_impact": total_impact,
            "total_trees": total_trees,
            "budget_utilization": utilization,
            "status": status_str,
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
    Sort by TPIS density (TPIS / trees), select greedily.
    """
    logger.info("Using greedy fallback optimization")

    gdf = zones.reset_index(drop=True).copy()
    trees = gdf["trees_possible"].values.astype(int)
    tpis = gdf["tpis"].values

    # Value density = TPIS (already per-zone impact indicator)
    density = tpis.copy()
    order = np.argsort(-density)

    selected = []
    remaining_budget = max_trees
    total_impact = 0

    for i in order:
        if trees[i] <= remaining_budget and trees[i] > 0:
            selected.append(int(i))
            remaining_budget -= trees[i]
            total_impact += tpis[i] * trees[i]

    total_trees = max_trees - remaining_budget
    gdf["selected"] = False
    gdf.loc[selected, "selected"] = True

    logger.info(f"Greedy: {len(selected)} zones, {total_trees} trees")
    return {
        "zones": gdf,
        "selected_indices": selected,
        "total_impact": total_impact,
        "total_trees": total_trees,
        "budget_utilization": total_trees / max(max_trees, 1) * 100,
        "status": "GREEDY",
    }


def pareto_sweep(
    feasible_zones: "gpd.GeoDataFrame",
    max_budget: int = None,
    step: int = None,
) -> List[Dict[str, Any]]:
    """
    Pareto budget sweep for efficiency frontier.

    Solves the knapsack for multiple budget levels to generate
    (budget, total_impact, zones_selected) curve.

    Args:
        feasible_zones: GeoDataFrame of feasible zones.
        max_budget: Maximum budget to sweep to.
        step: Budget increment step.

    Returns:
        List of (budget, impact, zones_count) dicts.
    """
    max_b = max_budget or MAX_TREES
    s = step or OPTIMIZER_PARAMS["pareto_step"]

    logger.info(f"Running Pareto sweep: 0 → {max_b}, step={s}")

    pareto_curve = []
    for budget in range(s, max_b + 1, s):
        result = knapsack_optimize(feasible_zones, max_trees=budget, timeout_sec=10)
        pareto_curve.append({
            "budget": budget,
            "total_impact": result["total_impact"],
            "zones_selected": len(result["selected_indices"]),
            "total_trees": result["total_trees"],
        })
        logger.debug(f"  Budget={budget}: impact={result['total_impact']:.2f}, "
                     f"zones={len(result['selected_indices'])}")

    logger.info(f"Pareto sweep complete: {len(pareto_curve)} points")
    return pareto_curve


if __name__ == "__main__":
    logger.info("Optimizer module loaded — run via pipeline_runner.py")
