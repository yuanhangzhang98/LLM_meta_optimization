"""Consensus objective evaluation system.

Aggregates information from all existing objective functions to produce a robust
"consensus objective" for design ranking. Uses Kendall tau correlation to measure
agreement between objectives and weights them by agreement and age.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
from scipy.stats import kendalltau

import objective_loader
import objective_metadata as obj_meta
from database.manager import DatabaseManager

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Caching for consensus objective                                             #
# --------------------------------------------------------------------------- #

@dataclass
class _ConsensusCache:
    """Cache for consensus objective computation."""
    design_ids: FrozenSet[int]
    objective_ids: FrozenSet[int]
    age_decay_base: float
    consensus_fn: Callable[[List[Dict[str, Any]]], float]
    stats: Dict[str, Any]


_consensus_cache: Optional[_ConsensusCache] = None


def _get_cache_key(
    db: DatabaseManager, age_decay_base: float
) -> Tuple[FrozenSet[int], FrozenSet[int], float]:
    """Get cache key from current state.

    Args:
        db: Database manager
        age_decay_base: Age decay parameter

    Returns:
        Tuple of (design_ids, objective_ids, age_decay_base)
    """
    designs = db.get_all_designs()
    design_ids = frozenset(d.design_id for d in designs)
    objective_ids = frozenset(objective_loader.list_available_objectives())
    return design_ids, objective_ids, age_decay_base


def invalidate_consensus_cache() -> None:
    """Manually invalidate the consensus cache.

    Call this if you need to force recomputation of the consensus objective,
    for example after modifying objective functions without changing their IDs.
    """
    global _consensus_cache
    _consensus_cache = None
    logger.debug("Consensus cache invalidated")


def evaluate_all_objectives_on_designs(
    db: DatabaseManager,
) -> Tuple[Dict[int, Dict[int, float]], List[int], List[int]]:
    """Evaluate all objectives on all designs to build a score matrix.

    Args:
        db: Database manager

    Returns:
        Tuple of:
        - score_matrix: objective_id -> design_id -> score
        - objective_ids: list of objective IDs evaluated
        - design_ids: list of design IDs evaluated
    """
    designs = db.get_all_designs()
    objective_ids = objective_loader.list_available_objectives()

    if not designs:
        logger.warning("No designs in database for consensus evaluation")
        return {}, objective_ids, []

    if not objective_ids:
        logger.warning("No objectives available for consensus evaluation")
        return {}, [], [d.design_id for d in designs]

    design_ids = [d.design_id for d in designs]
    score_matrix: Dict[int, Dict[int, float]] = {}

    for obj_id in objective_ids:
        try:
            obj_fn = objective_loader.load_objective(obj_id)
        except Exception as e:
            logger.warning(f"Failed to load objective {obj_id}: {e}")
            continue

        score_matrix[obj_id] = {}
        for design in designs:
            if not design.experiments:
                score_matrix[obj_id][design.design_id] = float("inf")
                continue

            try:
                score = obj_fn(design.experiments)
                # Handle invalid scores
                if score is None or not isinstance(score, (int, float)):
                    score = float("inf")
                elif score != score:  # NaN check
                    score = float("inf")
            except Exception as e:
                logger.warning(
                    f"Objective {obj_id} failed on design {design.design_id}: {e}"
                )
                score = float("inf")

            score_matrix[obj_id][design.design_id] = score

    return score_matrix, list(score_matrix.keys()), design_ids


def build_ranking_matrix(
    score_matrix: Dict[int, Dict[int, float]],
) -> Dict[int, Dict[int, int]]:
    """Convert score matrix to ranking matrix.

    For each objective, rank designs by their scores (lower score = rank 0 = best).
    Ties are broken by design_id for stability.

    Args:
        score_matrix: objective_id -> design_id -> score

    Returns:
        ranking_matrix: objective_id -> design_id -> rank (0 = best)
    """
    ranking_matrix: Dict[int, Dict[int, int]] = {}

    for obj_id, scores in score_matrix.items():
        # Sort by score (ascending), then by design_id for stability
        sorted_designs = sorted(scores.items(), key=lambda x: (x[1], x[0]))
        ranking_matrix[obj_id] = {
            design_id: rank for rank, (design_id, _) in enumerate(sorted_designs)
        }

    return ranking_matrix


def compute_kendall_tau_matrix(
    ranking_matrix: Dict[int, Dict[int, int]],
    objective_ids: List[int],
) -> Dict[Tuple[int, int], float]:
    """Compute pairwise Kendall tau correlations between all objectives.

    Args:
        ranking_matrix: objective_id -> design_id -> rank
        objective_ids: list of objective IDs to compare

    Returns:
        kendall_tau_matrix: (obj_i, obj_j) -> tau value in [-1, 1]
    """
    tau_matrix: Dict[Tuple[int, int], float] = {}

    if len(objective_ids) < 2:
        return tau_matrix

    # Get common design IDs across all objectives
    if not ranking_matrix:
        return tau_matrix

    common_design_ids = set.intersection(
        *[set(ranking_matrix[obj_id].keys()) for obj_id in objective_ids]
    )
    design_id_list = sorted(common_design_ids)

    if len(design_id_list) < 2:
        # Need at least 2 designs to compute correlation
        return tau_matrix

    for i, obj_i in enumerate(objective_ids):
        for obj_j in objective_ids[i + 1 :]:
            ranks_i = [ranking_matrix[obj_i][d] for d in design_id_list]
            ranks_j = [ranking_matrix[obj_j][d] for d in design_id_list]

            try:
                tau, _ = kendalltau(ranks_i, ranks_j)
                # Handle NaN (can happen with constant rankings)
                if tau != tau:  # NaN check
                    tau = 0.0
            except Exception:
                tau = 0.0

            tau_matrix[(obj_i, obj_j)] = tau
            tau_matrix[(obj_j, obj_i)] = tau

    return tau_matrix


def compute_objective_weights(
    kendall_tau_matrix: Dict[Tuple[int, int], float],
    objective_ids: List[int],
    age_decay_base: float = 0.9,
) -> Dict[int, float]:
    """Compute weights for each objective based on agreement and age.

    Weight formula: weight = normalized_agreement * age_decay
    - normalized_agreement: mean tau with other objectives, shifted from [-1,1] to [0,1]
    - age_decay: age_decay_base ^ (rounds_since_creation)

    Args:
        kendall_tau_matrix: (obj_i, obj_j) -> tau value
        objective_ids: list of objective IDs
        age_decay_base: base for exponential age decay (default: 0.9)

    Returns:
        weights: objective_id -> normalized weight (sum to 1)
    """
    if not objective_ids:
        return {}

    # Get metadata for age computation
    all_metadata = obj_meta.list_objective_metadata()

    # Find max round (most recent)
    max_round = 0
    for obj_id in objective_ids:
        meta = all_metadata.get(str(obj_id), {})
        created_round = meta.get("created_round", 0)
        max_round = max(max_round, created_round)

    weights: Dict[int, float] = {}

    for obj_id in objective_ids:
        # Compute agreement score
        if len(objective_ids) == 1:
            # Only one objective, no agreement to compute
            agreement = 1.0
        else:
            # Mean tau with all other objectives
            tau_values = [
                kendall_tau_matrix.get((obj_id, other_id), 0.0)
                for other_id in objective_ids
                if other_id != obj_id
            ]
            if tau_values:
                median_tau = np.median(tau_values)
            else:
                median_tau = 0.0
            agreement = max(median_tau, 0.0)

        # Compute age decay
        meta = all_metadata.get(str(obj_id), {})
        created_round = meta.get("created_round", 0)
        age = max_round - created_round
        age_decay = age_decay_base**age

        # Combined weight
        weights[obj_id] = agreement * age_decay

    # Normalize to sum to 1
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    else:
        # Fallback: equal weights
        n = len(objective_ids)
        weights = {obj_id: 1.0 / n for obj_id in objective_ids}

    return weights


def build_consensus_ranking(
    ranking_matrix: Dict[int, Dict[int, int]],
    objective_weights: Dict[int, float],
    design_ids: List[int],
) -> Dict[int, float]:
    """Build consensus ranking using weighted Borda count.

    For each design, computes weighted average of normalized ranks across objectives.

    Args:
        ranking_matrix: objective_id -> design_id -> rank
        objective_weights: objective_id -> weight
        design_ids: list of design IDs to rank

    Returns:
        consensus_ranks: design_id -> consensus score (lower is better)
    """
    if not design_ids or not ranking_matrix:
        return {}

    num_designs = len(design_ids)
    consensus: Dict[int, float] = {}

    for design_id in design_ids:
        weighted_rank = 0.0
        weight_sum = 0.0

        for obj_id, weight in objective_weights.items():
            if obj_id not in ranking_matrix:
                continue
            if design_id not in ranking_matrix[obj_id]:
                continue

            rank = ranking_matrix[obj_id][design_id]
            # Normalize rank to [0, 1]
            if num_designs > 1:
                normalized_rank = rank / (num_designs - 1)
            else:
                normalized_rank = 0.0

            weighted_rank += weight * normalized_rank
            weight_sum += weight

        # Normalize by total weight used
        if weight_sum > 0:
            consensus[design_id] = weighted_rank / weight_sum
        else:
            consensus[design_id] = 1.0  # Worst possible if no weights

    return consensus


def _hash_experiments(experiments: List[Dict[str, Any]]) -> str:
    """Create a hash of experiment results for lookup.

    Args:
        experiments: List of experiment dicts

    Returns:
        Hash string for lookup
    """
    # Sort experiments by N for consistency
    sorted_exps = sorted(experiments, key=lambda x: x.get("N", 0))
    # Create a canonical representation
    canonical = json.dumps(sorted_exps, sort_keys=True, default=str)
    return hashlib.md5(canonical.encode()).hexdigest()


def _estimate_normalized_rank(score: float, sorted_scores: List[float]) -> float:
    """Estimate normalized rank [0,1] for a score in a distribution.

    Uses binary search to find where the score would rank among existing scores,
    then normalizes to [0,1] range. This allows scale-invariant comparison
    across objectives with different score ranges.

    Args:
        score: The score to rank
        sorted_scores: Sorted list of existing scores (ascending)

    Returns:
        Normalized rank in [0, 1] where 0 = best, 1 = worst
    """
    if not sorted_scores:
        return 0.5  # No data, assume middle

    if score == float("inf") or score != score:  # inf or NaN
        return 1.0  # Worst possible

    # Binary search to find insertion point (lower scores = better = lower rank)
    rank = bisect.bisect_left(sorted_scores, score)

    # Normalize to [0, 1]
    if len(sorted_scores) <= 1:
        return 0.0
    return rank / len(sorted_scores)


def create_consensus_objective(
    db: DatabaseManager,
    age_decay_base: float = 0.9,
    meta_weights: Optional[Dict[int, float]] = None,
) -> Tuple[Callable[[List[Dict[str, Any]]], float], Dict[str, Any]]:
    """Create a consensus objective function (cached).

    Main entry point. Evaluates all objectives on all designs, computes
    Kendall tau correlations, weights objectives by agreement and age,
    and returns a callable consensus objective function.

    Results are cached and only recomputed when:
    - New designs are added (design IDs change)
    - New objectives are added (objective IDs change)
    - The age_decay_base parameter changes

    Note: meta_weights are NOT cached - they are applied fresh each time
    since they change between meta-agent rounds.

    Args:
        db: Database manager
        age_decay_base: base for exponential age decay (default: 0.9)
        meta_weights: Optional dict of objective_id -> weight multiplier from
            meta-agent. If provided, final weight = consensus_weight * meta_weight.
            Use 0.0 to effectively disable an objective.

    Returns:
        Tuple of:
        - consensus_fn: callable with signature (experiment_results) -> float
        - stats: dict with weights, tau matrix, num_objectives, num_designs
    """
    global _consensus_cache

    # Check cache validity
    design_ids_key, objective_ids_key, decay_key = _get_cache_key(db, age_decay_base)

    if (_consensus_cache is not None
        and _consensus_cache.design_ids == design_ids_key
        and _consensus_cache.objective_ids == objective_ids_key
        and _consensus_cache.age_decay_base == decay_key):
        # Cache hit
        logger.debug("Consensus cache hit")
        return _consensus_cache.consensus_fn, _consensus_cache.stats

    # Cache miss - recompute
    logger.debug("Consensus cache miss, recomputing...")

    # Step 1: Build score matrix
    score_matrix, objective_ids, design_ids = evaluate_all_objectives_on_designs(db)

    # Edge case: No objectives
    if len(objective_ids) == 0:

        def fallback_fn(experiment_results: List[Dict[str, Any]]) -> float:
            return float("inf")

        stats = {
            "weights": {},
            "tau_matrix": {},
            "num_objectives": 0,
            "num_designs": len(design_ids),
            "fallback": True,
            "fallback_reason": "no_objectives",
        }
        return fallback_fn, stats

    # Step 2: Build ranking matrix
    ranking_matrix = build_ranking_matrix(score_matrix)

    # Step 3: Compute Kendall tau matrix
    tau_matrix = compute_kendall_tau_matrix(ranking_matrix, objective_ids)

    # Step 4: Compute objective weights
    weights = compute_objective_weights(tau_matrix, objective_ids, age_decay_base)

    # Step 4b: Apply meta-agent weight multipliers if provided
    if meta_weights is not None:
        for obj_id in list(weights.keys()):
            multiplier = meta_weights.get(obj_id, 1.0)
            weights[obj_id] *= multiplier

        # Re-normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            # All weights are 0 - fall back to equal weights
            n = len(objective_ids)
            weights = {obj_id: 1.0 / n for obj_id in objective_ids}
            logger.warning("All meta_weights are 0, falling back to equal weights")

    # Step 5: Build consensus ranking
    consensus_ranks = build_consensus_ranking(ranking_matrix, weights, design_ids)

    # Build lookup table by experiment hash
    designs = db.get_all_designs()
    experiment_hash_to_design_id: Dict[str, int] = {}
    for design in designs:
        if design.experiments:
            exp_hash = _hash_experiments(design.experiments)
            experiment_hash_to_design_id[exp_hash] = design.design_id

    # Load all objective functions for on-the-fly computation
    objective_fns: Dict[int, Callable] = {}
    for obj_id in objective_ids:
        try:
            objective_fns[obj_id] = objective_loader.load_objective(obj_id)
        except Exception:
            pass

    # Store sorted score distributions for rank estimation (scale-invariant)
    score_distributions: Dict[int, List[float]] = {}
    for obj_id in objective_ids:
        if obj_id in score_matrix:
            # Filter out inf scores and sort for binary search
            finite_scores = [
                s for s in score_matrix[obj_id].values() if s != float("inf")
            ]
            score_distributions[obj_id] = sorted(finite_scores)

    def consensus_objective(experiment_results: List[Dict[str, Any]]) -> float:
        """Consensus objective function.

        Looks up the design by experiment hash and returns the pre-computed
        consensus rank. For new designs not in cache, estimates normalized
        rank using percentile-based ranking (scale-invariant).

        Args:
            experiment_results: List of experiment dicts

        Returns:
            float: consensus score (lower is better)
        """
        if not experiment_results:
            return float("inf")

        # Try to find design by experiment hash
        exp_hash = _hash_experiments(experiment_results)
        design_id = experiment_hash_to_design_id.get(exp_hash)

        if design_id is not None and design_id in consensus_ranks:
            return consensus_ranks[design_id]

        # On-the-fly computation for new designs using rank estimation
        # (scale-invariant: converts scores to normalized ranks before averaging)
        weighted_rank = 0.0
        weight_sum = 0.0
        for obj_id, weight in weights.items():
            if obj_id not in objective_fns or obj_id not in score_distributions:
                continue
            try:
                score = objective_fns[obj_id](experiment_results)
                if score is None or score != score:  # None or NaN
                    score = float("inf")
            except Exception:
                score = float("inf")

            # Estimate normalized rank (not raw score) for scale invariance
            norm_rank = _estimate_normalized_rank(score, score_distributions[obj_id])
            weighted_rank += weight * norm_rank
            weight_sum += weight

        if weight_sum > 0:
            return weighted_rank / weight_sum
        return 1.0  # Worst possible rank

    # Prepare stats
    stats = {
        "weights": weights,
        "tau_matrix": {f"{k[0]}-{k[1]}": v for k, v in tau_matrix.items()},
        "num_objectives": len(objective_ids),
        "num_designs": len(design_ids),
        "fallback": False,
        "objective_ids": objective_ids,
        "meta_weights_applied": meta_weights is not None,
        "meta_weight_multipliers": meta_weights,
    }

    # Update cache
    _consensus_cache = _ConsensusCache(
        design_ids=design_ids_key,
        objective_ids=objective_ids_key,
        age_decay_base=age_decay_base,
        consensus_fn=consensus_objective,
        stats=stats,
    )

    return consensus_objective, stats


def get_consensus_summary(stats: Dict[str, Any]) -> str:
    """Format consensus stats as a human-readable summary.

    Args:
        stats: stats dict from create_consensus_objective

    Returns:
        Formatted string summary
    """
    lines = []
    lines.append(f"Consensus Objective Summary:")
    lines.append(f"  Objectives: {stats['num_objectives']}")
    lines.append(f"  Designs: {stats['num_designs']}")

    if stats.get("meta_weights_applied"):
        lines.append("  Meta-agent weights: applied")

    if stats.get("fallback"):
        lines.append(f"  Fallback mode: {stats.get('fallback_reason', 'unknown')}")
    else:
        lines.append("  Objective weights:")
        meta_multipliers = stats.get("meta_weight_multipliers") or {}
        for obj_id, weight in sorted(stats["weights"].items()):
            mult = meta_multipliers.get(obj_id)
            if mult is not None and mult != 1.0:
                lines.append(f"    objective_{obj_id}: {weight:.3f} (meta: {mult:.2f}x)")
            else:
                lines.append(f"    objective_{obj_id}: {weight:.3f}")

    return "\n".join(lines)


def get_consensus_description(db: DatabaseManager, current_objective_id: int) -> str:
    """Generate description of the consensus objective.

    Args:
        db: Database manager
        current_objective_id: Fallback objective ID if only one objective exists

    Returns:
        Description string for the consensus objective
    """
    _, stats = create_consensus_objective(db, age_decay_base=0.9)

    if stats.get("fallback"):
        # Single or no objectives - use original description
        return obj_meta.get_objective_description(current_objective_id)

    # Build consensus description
    lines = [
        "**Consensus Objective** (aggregates all generated objectives)",
        "Weighting objectives by Kendall tau agreement and age decay",
        "Returning weighted average of normalized ranks",
        "",
        f"Active objectives: {stats['num_objectives']}",
        "Objective weights:",
    ]
    for obj_id, weight in sorted(stats["weights"].items()):
        desc = obj_meta.get_objective_description(obj_id, default="No description")
        # Truncate long descriptions
        if len(desc) > 80:
            desc = desc[:77] + "..."
        lines.append(f"  - objective_{obj_id} (weight={weight:.2f}): {desc}")

    return "\n".join(lines)




__all__ = [
    "create_consensus_objective",
    "get_consensus_summary",
    "evaluate_all_objectives_on_designs",
    "build_ranking_matrix",
    "compute_kendall_tau_matrix",
    "compute_objective_weights",
    "build_consensus_ranking",
    "get_consensus_description",
    "invalidate_consensus_cache",
]
