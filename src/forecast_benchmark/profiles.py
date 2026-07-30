"""Pre-registered SmartCast configurations for real-board selection.

The profile board is intentionally small and interpretable. It isolates the
layers implicated by the real 299-well development result: anchor choice,
candidate routing, terminal-upset recovery, and short-history cohort shrinkage.
It is not a hyperparameter search.
"""
from __future__ import annotations

from dataclasses import asdict
from math import log

from forecast_benchmark.smartcast import SmartCastConfig


def smartcast_profiles() -> dict[str, SmartCastConfig]:
    """Return the fixed v1.3 profile board.

    ``scipy_control`` is the exact frozen original benchmark. ``anchor_only``
    adds only the optional 6% terminal-decline tail so that rule is measured
    rather than silently bundled into the control. Other profiles add only one
    or a small number of testable layers.
    """
    scipy_control = SmartCastConfig(
        anchor_impl="scipy",
        use_recovery=False,
        use_cohort=False,
        use_ratio_coupling=False,
        use_terminal_decline=False,
        default_smart_weight=0.0,
        depressed_cutoff_smart_weight=0.0,
        terminal_streak_smart_weight=0.0,
        anchor_better_smart_weight=0.0,
        candidate_better_smart_weight=0.0,
        thin_history_smart_weight=0.0,
        allow_global_cohort_fallback=False,
        require_target_group_metadata=False,
    )
    anchor_only = SmartCastConfig(
        anchor_impl="scipy",
        use_recovery=False,
        use_cohort=False,
        use_ratio_coupling=False,
        default_smart_weight=0.0,
        depressed_cutoff_smart_weight=0.0,
        terminal_streak_smart_weight=0.0,
        anchor_better_smart_weight=0.0,
        candidate_better_smart_weight=0.0,
        thin_history_smart_weight=0.0,
        allow_global_cohort_fallback=False,
        require_target_group_metadata=False,
    )

    return {
        "scipy_control": scipy_control,
        "anchor_only": anchor_only,
        # Candidate board may deviate only when visible-history hindcasts are
        # materially better. Unconditional branches remain on the anchor.
        "candidate_gate": SmartCastConfig(
            anchor_impl="scipy",
            use_recovery=False,
            use_cohort=False,
            use_ratio_coupling=False,
            default_smart_weight=0.0,
            depressed_cutoff_smart_weight=0.0,
            terminal_streak_smart_weight=0.0,
            anchor_better_smart_weight=0.0,
            candidate_better_smart_weight=0.40,
            thin_history_smart_weight=0.0,
            candidate_better_ratio=0.80,
            allow_global_cohort_fallback=False,
            require_target_group_metadata=False,
        ),
        # Operational-upset layer only. The v1.2 80% deviation was too
        # aggressive for an unverified branch; this caps it at 35%.
        "recovery_conservative": SmartCastConfig(
            anchor_impl="scipy",
            use_recovery=True,
            use_cohort=False,
            use_ratio_coupling=False,
            default_smart_weight=0.0,
            depressed_cutoff_smart_weight=0.0,
            terminal_streak_smart_weight=0.35,
            anchor_better_smart_weight=0.0,
            candidate_better_smart_weight=0.0,
            thin_history_smart_weight=0.0,
            allow_global_cohort_fallback=False,
            require_target_group_metadata=False,
        ),
        # Cohort shrinkage only, restricted to formation/play cohorts with
        # support-aware weights. No mixed-play global fallback.
        "cohort_conservative": SmartCastConfig(
            anchor_impl="scipy",
            use_recovery=False,
            use_cohort=True,
            use_ratio_coupling=False,
            default_smart_weight=0.0,
            depressed_cutoff_smart_weight=0.0,
            terminal_streak_smart_weight=0.0,
            anchor_better_smart_weight=0.0,
            candidate_better_smart_weight=0.0,
            thin_history_smart_weight=0.0,
            allow_global_cohort_fallback=False,
            require_target_group_metadata=True,
            cohort_min_wells=8,
            cohort_full_support_wells=30,
        ),
        # v1.4 dispersion attack: exact SciPy plus only the replicated
        # cohort signal, with its authority bounded by evidence available at
        # forecast time. No candidate router, recovery model, ratio coupling,
        # global fallback, or learned black-box gate.
        # v1.5 selective deployment of the only strong spent-board signal:
        # unchanged conservative cohort shrinkage for oil-primary targets in
        # DJ, Eagle Ford, and Delaware; exact SciPy for every other target and
        # for secondary phases.  No provenance or producing-days fields enter
        # routing.
        "selective_cohort_v1": SmartCastConfig(
            anchor_impl="scipy",
            use_recovery=False,
            use_cohort=True,
            use_ratio_coupling=False,
            use_terminal_decline=False,
            default_smart_weight=0.0,
            depressed_cutoff_smart_weight=0.0,
            terminal_streak_smart_weight=0.0,
            anchor_better_smart_weight=0.0,
            candidate_better_smart_weight=0.0,
            thin_history_smart_weight=0.0,
            allow_global_cohort_fallback=False,
            require_target_group_metadata=False,
            cohort_min_wells=8,
            cohort_full_support_wells=30,
            cohort_allowed_primary_phases=("oil",),
            cohort_allowed_groups=("dj", "eagle_ford", "delaware"),
            cohort_allowed_forecast_phases=("oil",),
        ),
        "gated_cohort_v1": SmartCastConfig(
            anchor_impl="scipy",
            use_recovery=False,
            use_cohort=True,
            use_ratio_coupling=False,
            use_terminal_decline=False,
            default_smart_weight=0.0,
            depressed_cutoff_smart_weight=0.0,
            terminal_streak_smart_weight=0.0,
            anchor_better_smart_weight=0.0,
            candidate_better_smart_weight=0.0,
            thin_history_smart_weight=0.0,
            allow_global_cohort_fallback=False,
            require_target_group_metadata=True,
            cohort_min_wells=12,
            cohort_full_support_wells=40,
            cohort_max_weight=0.25,
            cohort_blend_space="log",
            cohort_support_horizon=12,
            cohort_min_forecast_age_support=8,
            cohort_max_log_mad=0.40,
            cohort_similarity_window=12,
            cohort_max_history_log_error=0.35,
            cohort_max_monthly_log_divergence=log(1.35),
            cohort_max_cumulative_log_divergence=log(1.18),
            cohort_endpoint_ratio_low=0.60,
            cohort_endpoint_ratio_high=1.60,
            cohort_endpoint_log_volatility_max=0.45,
        ),
        # Combined conservative major-phase engine. Ratio coupling stays off
        # because it cannot improve the major-phase score and must be judged
        # separately under an all-phase metric.
        "major_conservative": SmartCastConfig(
            anchor_impl="scipy",
            use_recovery=True,
            use_cohort=True,
            use_ratio_coupling=False,
            default_smart_weight=0.0,
            depressed_cutoff_smart_weight=0.0,
            terminal_streak_smart_weight=0.35,
            anchor_better_smart_weight=0.0,
            candidate_better_smart_weight=0.40,
            thin_history_smart_weight=0.10,
            anchor_better_ratio=0.95,
            candidate_better_ratio=0.80,
            allow_global_cohort_fallback=False,
            require_target_group_metadata=True,
            cohort_min_wells=8,
            cohort_full_support_wells=30,
        ),
        # Same major-phase router with secondary-phase coupling enabled. It is
        # retained for all-phase experiments, not headline profile selection.
        "full_conservative": SmartCastConfig(
            anchor_impl="scipy",
            use_recovery=True,
            use_cohort=True,
            use_ratio_coupling=True,
            default_smart_weight=0.0,
            depressed_cutoff_smart_weight=0.0,
            terminal_streak_smart_weight=0.35,
            anchor_better_smart_weight=0.0,
            candidate_better_smart_weight=0.40,
            thin_history_smart_weight=0.10,
            anchor_better_ratio=0.95,
            candidate_better_ratio=0.80,
            allow_global_cohort_fallback=False,
            require_target_group_metadata=True,
            cohort_min_wells=8,
            cohort_full_support_wells=30,
            ratio_independent_blend=0.97,
        ),
        # Reproduce Claude's v1.2 routing after the anchor and NaN defects are
        # fixed. Included only as a diagnostic arm.
        "v12_fixed": SmartCastConfig(anchor_impl="scipy"),
        # Reproduce the v1.1 regression anchor for diagnosis only.
        "v11_regression": SmartCastConfig(anchor_impl="linearized"),
        "candidates_only": SmartCastConfig(
            anchor_impl="none",
            use_recovery=False,
            use_cohort=False,
            use_ratio_coupling=False,
        ),
    }


def profile_names(*, production_only: bool = False) -> tuple[str, ...]:
    names = tuple(smartcast_profiles())
    if not production_only:
        return names
    return tuple(
        name
        for name in names
        if name not in {"v11_regression", "candidates_only"}
    )


def get_profile(name: str) -> SmartCastConfig:
    profiles = smartcast_profiles()
    try:
        return profiles[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown SmartCast profile {name!r}; choose from {', '.join(profiles)}"
        ) from exc


def profile_payload() -> dict[str, dict[str, object]]:
    return {name: asdict(config) for name, config in smartcast_profiles().items()}
