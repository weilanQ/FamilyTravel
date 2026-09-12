"""Comparison baselines for the QG-BMP family-travel multi-agent framework.

Four baselines forming a 2x2 matrix ({single, multi/aggregated} x {non-resolving,
resolving}), each isolating one question about where the gains come from:
  - single_agent_direct    : minimal generation-only anchor (no conflict resolution)
  - preference_aggregation : the preference-aggregation paradigm the paper argues against
  - sa_resolve             : budget-matched CENTRAL single-agent conflict resolution
                             (explicit conflict list + merge/substitute/split/relax +
                             4 verifier-feedback repair rounds)
  - generic_ma_debate      : generic multi-agent debate (same personas, 4 free-form
                             rounds + coordinator) WITHOUT the typed conflict graph
                             or SHARED/SPLIT/UNSAT dispatch

(single_agent_cot and independent_multi_agent were removed 2026-07: each converged
with its sibling anchor and is subsumed by sa_resolve / generic_ma_debate.)

All baselines reuse the QG-BMP grounding / cost / evaluation; the two resolving
baselines additionally reuse the full pipeline's rule verifier and get their
self-authored splits credited under the same completeness bar (credit_llm_splits),
so metric differences reflect the coordination mechanism only.
"""
