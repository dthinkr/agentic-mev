#!/usr/bin/env python3
"""Paired native development estimates from raw capture or pinned frozen counts.

What this estimates
-------------------
Two native outcomes, measured by ``analysis/strategy_development.py`` from a
verified ``development_acquire`` capture:

``deployments``           successful top-level contract-creation transactions
``deploying_operators``   distinct senders among them

Both are counted **for the frozen strategy cohort only**: the senders the
verified panel recorded for that episode and tier. They are not a measure of
development by AI users at large, by the chain, or by anyone outside the
cohort, and no row here supports such a reading. An internal ``CREATE`` by a
factory emits no top-level transaction and is invisible, as are proxy upgrades
and redeployments.

What it reuses, and what it does not
------------------------------------
Every number comes from the existing math.
``submission_study.match_windows`` performs the matching, with its own
``metrics`` parameter pointed at the two native columns, and
``submission_study.paired_statistics`` performs the clustered paired inference.
``strategy_study.allocation_support`` decides when a core-versus-small
comparison is reportable, from frozen cohort membership rather than realised
activity. ``estimate_strategy.bootstrap_config`` and ``calendar_offsets``
supply the frozen bootstrap and calendar settings. No inference is
reimplemented here; this module assembles a frame and reports.

The ``Q = A x I`` decomposition is **not** produced. ``extension_study``'s
``pair_margins`` and ``decompose`` are written against the metric names
``transactions`` and ``active_operators``; reaching them would mean labelling
development counts as trading counts, which would make the published identity
a statement about the wrong outcome. The result says so explicitly rather than
relabelling.

Coverage, and the flag that must not be reused
----------------------------------------------
``match_windows`` already separates deployment coverage from calendar validity:
a metric in its ``DEPLOYMENT_METRICS`` is gated per row by
``deployment_complete``, both on the treated window and on each candidate
reference. This driver fills that column from the *development* reader's own
``coverage_available`` and from nothing else.

The panel's ``coverage_available`` is trading coverage. It says whether
identified strategy transactions were observed, which has no bearing on whether
the deployment source covered the same window, so it travels as a diagnostic
column and never filters a row. Realised trading is never consulted either: a
cohort with known frozen history that deployed nothing inside a covered window
is a measured zero and belongs in the comparison.

Contamination semantics are the panel's, unchanged: the treated-window policy
is verified to be ``retain_eligible``, the analysis view is re-derived from the
raw screen through ``strategy_study``'s own policy function, and the matcher
sees ``valid``/``invalid_reason`` exactly as the trading path builds them.

When the capture is incomplete
------------------------------
An uncovered window carries no counts. Its cells are dropped by the matcher as
``incomplete_deployment_capture`` and, if nothing survives, the result is an
explicit unavailable with reasons. No absent count becomes a zero.

Frozen input mode verifies a caller-pinned measurement receipt, count table,
prior raw audit and source-code bindings through ``development_inputs``. It
uses this same assembly and inference path without rereading raw capture.
Original measurement provenance is retained; the receipt records the changed
stage boundary. Frozen mode is explicit and is never a missing-raw fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]


def _sibling(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / f"analysis/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_es = _sibling("estimate_strategy")
_ss = _sibling("strategy_study")
_base = _sibling("submission_study")
_sd = _sibling("strategy_development")
_di = _sibling("development_inputs")

RESULT_NAME = "development_estimates.json"
RECEIPT_NAME = "development_receipt.json"
PUBLISHED = (RESULT_NAME, RECEIPT_NAME)
# The legacy metric names, so submission_study.DEPLOYMENT_METRICS recognises
# them and applies the deployment_complete check.
DEV_METRICS = {"deployments": "deployment_count",
               "deploying_operators": "deploying_operators"}
SOURCE_COLUMNS = {"creation_tx_count": "deployment_count",
                  "deploying_addresses": "deploying_operators"}
PANEL_IDENTITY = ("strategy", "provider", "event_eligible", "start_s", "end_s",
                  "window_start_s", "window_end_s", "qualification_rule",
                  "screen_valid", "screen_invalid_reason", "analysis_valid",
                  "analysis_invalid_reason", "treatment_policy", "coverage_available")
JOIN_KEYS = ("event_id", "design", "window", "offset_days", "tier", "head_cut",
             "onset_shift_minutes")
# The only normalisation applied to a panel column: the builder omits the onset
# shift on its primary arm, and the reader records the same 0 for it.
ONSET_DEFAULT = 0
# Fields a development record must reproduce exactly from the panel cell it names.
MIRRORED = ("window_start_s", "window_end_s", "qualification_rule")
CODE_MODULES = ("estimate_development", "development_inputs", "estimate_strategy", "strategy_development",
                "strategy_study", "submission_study", "development_acquire", "mev_acquire")
TOTAL_TIER = "total"
GROUP_TIERS = ("core", "small")
DECOMPOSITION = {
    "available": False,
    "reason": "not_implemented_for_native_development_metrics",
    "detail": ("extension_study.pair_margins and decompose are written against the "
               "metric names transactions and active_operators. Producing a Q = A x I "
               "identity for deployments would mean labelling development counts as "
               "trading counts, so it is left unavailable rather than relabelled."),
}
OUTCOME_DEFINITION = {
    "deployments": "successful top-level contract-creation transactions",
    "deploying_operators": "distinct senders among those transactions",
    "population": ("the frozen strategy cohort for that episode and tier, and nobody "
                   "else; not development by AI users at large and not a chain total"),
    "invisible": ("internal CREATE by a factory emits no top-level transaction; proxy "
                  "upgrades and redeployments are not observed"),
    "zero": ("zero observed top-level creations inside a covered window, which is a "
             "measured zero and not an absence of development"),
}


class DevelopmentEstimationError(RuntimeError):
    """A refusal this driver raises on purpose."""


# --------------------------------------------------------------------------- #
# frame assembly
# --------------------------------------------------------------------------- #

def _panel_index(counts) -> dict[tuple, dict[str, Any]]:
    """Every authoritative cell, keyed by the full stratum key.

    ``onset_shift_minutes`` is the one column the builder may omit; it is
    normalised to :data:`ONSET_DEFAULT`, which is the same value the reader
    records. Nothing else is defaulted.
    """
    optional = {"onset_shift_minutes"}
    missing = [name for name in PANEL_IDENTITY + JOIN_KEYS
               if name not in counts.columns and name not in optional]
    if missing:
        raise DevelopmentEstimationError(f"the verified panel lacks columns {missing}")
    columns = [name for name in list(JOIN_KEYS) + list(PANEL_IDENTITY)
               if name in counts.columns]
    out: dict[tuple, dict[str, Any]] = {}
    for row in (dict(zip(columns, values)) for values in
                counts[columns].itertuples(index=False, name=None)):
        row.setdefault("onset_shift_minutes", ONSET_DEFAULT)
        if row["onset_shift_minutes"] is None:
            row["onset_shift_minutes"] = ONSET_DEFAULT
        key = tuple(row[name] for name in JOIN_KEYS)
        if key in out:
            raise DevelopmentEstimationError(f"the verified panel repeats the cell {key}")
        out[key] = row
    return out


def _flag(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    inner = value.item() if hasattr(value, "item") else value
    if isinstance(inner, bool):
        return inner
    if isinstance(inner, str) and inner.strip().lower() in ("true", "false"):
        return inner.strip().lower() == "true"
    raise DevelopmentEstimationError(f"{name} is not an explicit boolean: {value!r}")


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or value is None:
        raise DevelopmentEstimationError(f"{name} is not an exact count")
    if isinstance(value, float):
        if not value.is_integer():
            raise DevelopmentEstimationError(f"{name} is fractional")
        value = int(value)
    if not isinstance(value, int) or value < 0:
        raise DevelopmentEstimationError(f"{name} is not a non-negative integer")
    return value


def development_frame(development: dict[str, Any], loaded: dict[str, Any]):
    """Join the development counts onto the panel's authoritative identity.

    The join is exact in both directions, and the equality is checked over the
    whole key set *before* any eligibility filtering: a development output that
    silently dropped an entire episode or window family would otherwise match
    cleanly on whatever remained. A development count never travels without the
    episode identity, validity and contamination view the panel froze for it,
    and a record that disagrees with its own panel cell about the window bounds,
    head cut, onset shift or qualification rule is refused rather than joined.
    """
    import pandas as pd

    panel = _panel_index(loaded["counts"])
    supplied: dict[tuple, dict[str, Any]] = {}
    for record in development["windows"]:
        key = tuple(record[name] for name in JOIN_KEYS)
        if key in supplied:
            raise DevelopmentEstimationError(f"the development output repeats the cell {key}")
        supplied[key] = record
    _require_same_cells(set(panel), set(supplied))

    rows, excluded = [], []
    for key, record in sorted(supplied.items(), key=lambda item: tuple(map(str, item[0]))):
        identity = panel[key]
        for name in MIRRORED:
            if record[name] != identity[name]:
                raise DevelopmentEstimationError(
                    f"development row {key} reports {name}={record[name]!r} but the "
                    f"verified panel declares {identity[name]!r}")
        if not _flag(identity["event_eligible"], "event_eligible"):
            excluded.append({**record, "exclusion": "episode_not_eligible"})
            continue
        if identity["treatment_policy"] != "retain_eligible":
            raise DevelopmentEstimationError(
                f"unsupported treated-window policy {identity['treatment_policy']!r}; "
                "the development estimate uses the panel's frozen policy unchanged")
        # The analysis view is re-derived from the raw screen through the panel's
        # own policy function, never trusted as published.
        screen = "" if identity["screen_invalid_reason"] is None else str(
            identity["screen_invalid_reason"])
        analysis = "" if identity["analysis_invalid_reason"] is None else str(
            identity["analysis_invalid_reason"])
        expected = _ss._policy_reason(screen, identity["window"], identity["offset_days"])
        if analysis != expected:
            raise DevelopmentEstimationError(
                f"analysis_invalid_reason for {key} is {analysis!r} but the frozen "
                f"policy implies {expected!r}")
        if _flag(identity["analysis_valid"], "analysis_valid") != (expected == ""):
            raise DevelopmentEstimationError(f"analysis_valid disagrees with its reason for {key}")
        if _flag(identity["screen_valid"], "screen_valid") != (screen == ""):
            raise DevelopmentEstimationError(f"screen_valid disagrees with its reason for {key}")
        covered = _flag(record["coverage_available"], "development coverage_available")
        counts = {}
        for source, column in SOURCE_COLUMNS.items():
            value = record[source]
            if covered:
                counts[column] = _count(value, column)
            elif value is not None:
                raise DevelopmentEstimationError(
                    f"{key} is not covered yet carries a {source}; an uncovered window "
                    "must report no count at all")
            else:
                counts[column] = float("nan")
        if covered and counts["deploying_operators"] > counts["deployment_count"]:
            raise DevelopmentEstimationError(
                f"{key} reports more distinct deployers than deployment transactions")
        rows.append({
            "event_id": record["event_id"], "design": record["design"],
            "window": record["window"], "offset_days": int(record["offset_days"]),
            "tier": record["tier"],
            "head_cut": int(record["head_cut"]),
            "onset_shift_minutes": int(record["onset_shift_minutes"]),
            "window_start_s": int(record["window_start_s"]),
            "window_end_s": int(record["window_end_s"]),
            "provider": str(identity["provider"]),
            "start_s": int(identity["start_s"]), "end_s": int(identity["end_s"]),
            "valid": _flag(identity["analysis_valid"], "analysis_valid"),
            "invalid_reason": analysis,
            # Development metrics require creation-data coverage, not trading coverage.
            "deployment_complete": covered,
            # Carried for the record, never used to filter a development row.
            "trading_coverage_available": _flag(identity["coverage_available"],
                                                "coverage_available"),
            "cohort_addresses": int(record["cohort_addresses"]),
            "tier_is_known_empty": _flag(record["tier_is_known_empty"],
                                         "tier_is_known_empty"),
            "coverage_reason": record["coverage_reason"],
            **counts,
        })
    if not rows:
        raise DevelopmentEstimationError(
            "no eligible development window survived the join; nothing can be estimated")
    frame = pd.DataFrame(rows)
    _require_two_tiers(frame)
    if (frame.window_end_s <= frame.window_start_s).any():
        raise DevelopmentEstimationError("a window is not a positive half-open interval")
    if (frame.end_s <= frame.start_s).any():
        raise DevelopmentEstimationError("an episode is not a positive half-open interval")
    return frame, excluded


def _require_same_cells(panel: set, supplied: set) -> None:
    """Exact key-set equality, in both directions, before anything is filtered."""
    absent, extra = sorted(panel - supplied), sorted(supplied - panel)
    if absent or extra:
        raise DevelopmentEstimationError(
            "the development output and the verified panel do not describe the same "
            f"cells: {len(absent)} declared by the panel and not measured "
            f"(first {absent[:3]}), {len(extra)} measured and not declared "
            f"(first {extra[:3]})")


def _require_two_tiers(frame) -> None:
    keys = ["event_id", "design", "window", "offset_days", "head_cut", "onset_shift_minutes"]
    grouped = frame.groupby(keys, dropna=False).tier
    if not grouped.nunique().eq(len(_ss.TIERS)).all():
        bad = grouped.nunique()
        raise DevelopmentEstimationError(
            "every window must carry both tiers, including a zero-count tier; "
            f"{int((bad != len(_ss.TIERS)).sum())} cells do not")
    if frame.duplicated(keys + ["tier"]).any():
        raise DevelopmentEstimationError("duplicate development cell after the join")


# --------------------------------------------------------------------------- #
# estimation, all borrowed math
# --------------------------------------------------------------------------- #

def _statistics_on(frame, loaded, settings, label: str) -> dict[str, Any]:
    """Totals on the frame, group statistics on the historical both-tier subset."""
    out: dict[str, Any] = {}
    if frame.empty:
        return out
    support = _ss.allocation_support(frame, loaded["events"], loaded["cohorts"])
    events = support["events"]
    for metric in sorted(set(frame.metric)):
        rows = frame[frame.metric == metric]
        total = {"support": label, "events": sorted(set(rows.event_id)),
                 "n_events": int(rows.event_id.nunique())}
        allocation = {"support": f"{label}+historical_both_tier",
                      "available": bool(events),
                      "reason": None if events else support["reason"],
                      "events": events, "n_events": len(events),
                      "undefined_ratio_events": support["undefined_ratio_events"],
                      "membership_source": support["membership_source"]}
        for band, days in (("primary", settings["primary_cluster_days"]),
                           ("fortnight", settings["sensitivity_cluster_days"])):
            _ss.require_single_stratum(rows, "estimate_development")
            stats = _base.paired_statistics(rows, draws=settings["bootstrap_draws"],
                                            seed=settings["bootstrap_seed"],
                                            cluster_days=days, include_market=False)
            total[band] = {k: v for k, v in stats.items() if k in _ss.TOTAL_KEYS}
            if events:
                scoped = rows[rows.event_id.isin(events)]
                grouped = _base.paired_statistics(
                    scoped, draws=settings["bootstrap_draws"],
                    seed=settings["bootstrap_seed"], cluster_days=days,
                    include_market=False)
                allocation[band] = {k: v for k, v in grouped.items()
                                    if k in _ss.ALLOCATION_KEYS}
            else:
                allocation[band] = None
        out[metric] = {"total": total, "allocation": allocation}
    return out


def minimum_clean_controls(cfg: dict[str, Any]) -> int:
    """The frozen reference requirement, from the configuration and nowhere else."""
    if "minimum_clean_controls" not in cfg:
        raise DevelopmentEstimationError(
            "strategy configuration lacks minimum_clean_controls; the matcher's "
            "reference requirement is a frozen value and is not defaulted here")
    return _es._exact_int(cfg["minimum_clean_controls"], "minimum_clean_controls",
                          minimum=1)


def _calendar_block(frame, loaded, calendar: str, offsets, settings, min_clean: int):
    """One calendar: select the design and primary stratum, match, then report."""
    selected = _ss.select_design(frame, calendar)
    stratum = selected[(selected.window == _ss.PRIMARY_WINDOW)
                       & (selected.head_cut == _es.PRIMARY_HEAD_CUT)
                       & (selected.onset_shift_minutes == _es.PRIMARY_ONSET_SHIFT_MINUTES)]
    if stratum.empty:
        raise DevelopmentEstimationError(
            f"{calendar} carries no {_ss.PRIMARY_WINDOW} window at head cut "
            f"{_es.PRIMARY_HEAD_CUT}")
    matched, dropped = _base.match_windows(
        stratum.reset_index(drop=True), [int(o) for o in offsets], calendar,
        min_clean=min_clean, treatment_offset=0, metrics=DEV_METRICS,
        market_column=None)
    block = {
        "calendar": calendar, "offsets": [int(o) for o in offsets],
        "minimum_clean_controls": min_clean,
        "window": _ss.PRIMARY_WINDOW, "head_cut": _es.PRIMARY_HEAD_CUT,
        "onset_shift_minutes": _es.PRIMARY_ONSET_SHIFT_MINUTES, "treatment_offset": 0,
        "n_input_cells": int(len(stratum)),
        "n_matched_cells": int(len(matched)),
        "matched_events": sorted(set(matched.event_id)) if not matched.empty else [],
        "dropped": _es._dropped_reasons(dropped),
        "statistics": _statistics_on(matched, loaded, settings, "all_eligible_matched"),
        "allocation_support": _ss.allocation_support(matched, loaded["events"],
                                                     loaded["cohorts"]),
        "decomposition": dict(DECOMPOSITION),
    }
    return block, matched


def estimate(indir: Path, *, boundaries: Optional[Path] = None,
             capture_roots: Sequence[Path] = (),
             root: Path = ROOT, study_config: Path = _es.DEFAULT_STRATEGY_CONFIG,
             extension_config: Path = _es.DEFAULT_EXTENSION_CONFIG,
             attested_tables: Sequence[Sequence[Any]] = (),
             frozen_counts: Optional[Path] = None,
             frozen_receipt_sha256: Optional[str] = None,
             source_audit: Optional[Path] = None,
             source_root: Optional[Path] = None) -> dict[str, Any]:
    """Run native inference from explicitly selected raw or pinned frozen inputs."""
    frozen = frozen_counts is not None
    if frozen:
        if boundaries is not None or capture_roots or attested_tables:
            raise DevelopmentEstimationError("frozen counts and raw capture inputs are mutually exclusive")
        if frozen_receipt_sha256 is None or source_audit is None:
            raise DevelopmentEstimationError("frozen counts require a trusted receipt SHA256 and source audit")
    else:
        if frozen_receipt_sha256 is not None or source_audit is not None or source_root is not None:
            raise DevelopmentEstimationError("frozen receipt, source audit and relocation require explicit frozen counts")
        if boundaries is None:
            raise DevelopmentEstimationError("raw capture mode requires boundaries; frozen counts are never a fallback")
    indir = _es._resolve(root, indir)
    study_path = _es._resolve(root, study_config)
    extension_path = _es._resolve(root, extension_config)
    if source_root is not None:
        _ss.source_path(Path(source_root), root=root, source_root=source_root)
        for path in (indir, study_path, extension_path):
            _ss.local_path(path, root=root)
    cfg = _es._read_json(study_path, label="strategy configuration")
    extension_cfg = _es._read_json(extension_path, label="extension configuration")
    settings = _es.bootstrap_config(cfg, extension_cfg)
    offsets = _es.calendar_offsets(cfg)
    min_clean = minimum_clean_controls(cfg)

    stage = None
    if frozen:
        loaded = _ss.load_scoped_inputs(indir, root=root, config_path=study_path,
                                        source_root=source_root)
        development, stage = _di.load_frozen(
            indir, directory=_es._resolve(root, frozen_counts),
            receipt_sha256=frozen_receipt_sha256, source_audit=_es._resolve(root, source_audit),
            loaded=loaded, root=root, source_root=source_root)
    else:
        development = _sd.read_and_count(
            indir, boundaries=Path(boundaries), capture_roots=[Path(r) for r in capture_roots],
            attested_tables=attested_tables, root=root, config_path=study_path)
        loaded = _ss.load_scoped_inputs(indir, root=root, config_path=study_path)
    manifest = loaded["manifest"]
    receipt = build_receipt(indir=indir, manifest=manifest, study_path=study_path,
                            extension_path=extension_path, settings=settings,
                            development=development, min_clean=min_clean, root=root,
                            source_root=source_root)
    if stage is not None:
        receipt["input_stage"] = stage
    if source_root is not None:
        receipt["path_resolution"] = {"recorded_root": str(source_root),
                                      "effective_root": str(Path(root).resolve()),
                                      "origin_fallback": False}
    frame, excluded = development_frame(development, loaded)

    calendars, matched_frames = {}, {}
    for calendar in _es.CALENDARS:
        block, matched = _calendar_block(frame, loaded, calendar, offsets[calendar],
                                         settings, min_clean)
        calendars[calendar] = block
        matched_frames[calendar] = matched

    matched_sets = {name: set(block["matched_events"]) for name, block in calendars.items()}
    common = sorted(set.intersection(*matched_sets.values())) if matched_sets else []
    # Each calendar's already-matched rows are filtered to the common episodes
    # and the statistics run again; this is a re-estimation, not a re-match, so
    # every episode keeps the clean references it was matched against.
    cross = {"role": "common_support_comparison", "matched_in_both": common,
             "n_matched_in_both": len(common),
             "matched_only_in": {name: sorted(events - set(common))
                                 for name, events in matched_sets.items()},
             "calendars": {}}
    for calendar, matched in matched_frames.items():
        subset = matched[matched.event_id.isin(common)] if common else matched.iloc[0:0]
        cross["calendars"][calendar] = {
            "n_matched_cells": int(len(subset)),
            "statistics": _statistics_on(subset, loaded, settings,
                                         "common_to_both_calendars"),
            "decomposition": dict(DECOMPOSITION),
        }

    available = any(block["n_matched_cells"] for block in calendars.values())
    return {
        "receipt": receipt, "strategy": manifest["strategy"],
        "available": available,
        "unavailable_reason": None if available else
        "no development cell survived matching; see development_coverage and dropped",
        "outcome_definition": dict(OUTCOME_DEFINITION),
        "metrics": {name: column for name, column in sorted(DEV_METRICS.items())},
        "primary_stratum": receipt["primary_stratum"], "bootstrap": settings,
        "minimum_clean_controls": receipt["minimum_clean_controls"],
        "calendars": calendars, "cross_calendar": cross,
        "development_coverage": receipt["development_source"]["coverage"],
        "excluded_input_rows": {"n": len(excluded), "by_reason": _by_reason(excluded)},
        "decomposition": dict(DECOMPOSITION),
        "trading_coverage_used_as_a_gate": False,
        "profit_status": "not_measured",
        "causal_claim": "none; counts and ratios are diagnostic",
    }


def _by_reason(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        out[row["exclusion"]] = out.get(row["exclusion"], 0) + 1
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- #
# receipt and atomic publication
# --------------------------------------------------------------------------- #

def code_receipt(root: Path = ROOT, *, source_root: Optional[Path] = None) -> dict[str, str]:
    out = {}
    for name in CODE_MODULES:
        path = Path(root) / "analysis" / f"{name}.py"
        if source_root is not None:
            path = _ss.local_path(path, root=root)
        if not path.exists():
            raise DevelopmentEstimationError(f"a module this driver depends on is missing: {path}")
        out[name] = _es.sha256_file(path)
    return out


def build_receipt(*, indir: Path, manifest: dict[str, Any], study_path: Path,
                  extension_path: Path, settings: dict[str, Any],
                  development: dict[str, Any], min_clean: int,
                  root: Path = ROOT, source_root: Optional[Path] = None) -> dict[str, Any]:
    """Provenance for the estimate, including the capture the counts came from."""
    inputs = {record["file"]: record["sha256"] for record in manifest["outputs"]}
    for name, recorded in sorted(inputs.items()):
        path = Path(indir) / name
        if source_root is not None:
            path = _ss.local_path(path, root=root)
        got = _es.sha256_file(path)
        if got != recorded:
            raise DevelopmentEstimationError(
                f"{name} hashes {got} but the panel manifest says {recorded}; rebuild "
                "the panel rather than estimating against stale inputs")
    coverage = {
        "windows_total": development["windows_total"],
        "windows_available": development["windows_available"],
        "windows_unavailable": development["windows_unavailable"],
        "windows_on_a_known_empty_tier": development["windows_on_a_known_empty_tier"],
        "evidence": development["evidence"],
        "sources": development["sources"],
        "assumptions": development["assumptions"],
        "scope_notes": development["scope_notes"],
    }
    return {
        "driver": "estimate_development", "strategy": manifest["strategy"],
        "outcome_scope": "native_development_counts_for_frozen_strategy_cohorts",
        "panel_outcome_scope": manifest.get("outcome_scope"),
        "input_directory": str(indir),
        "input_manifest_sha256": _es.sha256_file(Path(indir) / "strategy_input_manifest.json"),
        "input_outputs": dict(sorted(inputs.items())),
        "panel_code_hashes": dict(sorted((manifest.get("code_hashes") or {}).items())),
        "panel_source_hashes": manifest.get("source_hashes"),
        "strategy_config": {"path": str(study_path), "sha256": _es.sha256_file(study_path)},
        "extension_config": {"path": str(extension_path),
                             "sha256": _es.sha256_file(extension_path)},
        "driver_code_hashes": code_receipt(root, source_root=source_root),
        "development_source": {"reader": development["reader_version"],
                               "provenance": development["provenance"],
                               "coverage": coverage},
        "bootstrap": dict(settings),
        "primary_stratum": {"window": _ss.PRIMARY_WINDOW,
                            "head_cut": _es.PRIMARY_HEAD_CUT,
                            "onset_shift_minutes": _es.PRIMARY_ONSET_SHIFT_MINUTES},
        "calendars": list(_es.CALENDARS),
        "minimum_clean_controls": {"value": min_clean,
                                   "source": "strategy_config.minimum_clean_controls"},
        "metrics": dict(sorted(DEV_METRICS.items())),
        "outcome_definition": dict(OUTCOME_DEFINITION),
        "decomposition": dict(DECOMPOSITION),
        "profit_status": "not_measured",
        "causal_claim": "none; counts and ratios are diagnostic",
    }


def _comparable(receipt: dict[str, Any]) -> dict[str, Any]:
    return {key: receipt.get(key) for key in
            ("driver", "strategy", "outcome_scope", "input_manifest_sha256",
             "input_outputs", "panel_code_hashes", "strategy_config", "extension_config",
             "driver_code_hashes", "development_source", "bootstrap", "primary_stratum",
             "calendars", "minimum_clean_controls", "metrics", "input_stage", "path_resolution")}


def publish(outdir: Path, files: dict[str, str]) -> dict[str, Any]:
    """Refuse an unknown or drifted directory, then publish with one rename."""
    outdir = Path(outdir)
    receipt_payload = json.loads(files[RECEIPT_NAME])
    if outdir.exists():
        present = {p.name for p in outdir.iterdir()}
        unknown = sorted(present - set(PUBLISHED))
        if unknown:
            raise DevelopmentEstimationError(
                f"{outdir} holds files this driver did not write ({unknown}); refusing "
                "to publish into a directory whose contents are unknown")
        existing = outdir / RECEIPT_NAME
        if present and not existing.exists():
            raise DevelopmentEstimationError(
                f"{outdir} holds {sorted(present)} but no {RECEIPT_NAME}; an orphan "
                "result has no provenance and will not be overwritten")
        if existing.exists():
            try:
                prior = json.loads(existing.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raise DevelopmentEstimationError(
                    f"{existing} cannot be read; move it aside") from None
            if _comparable(prior) != _comparable(receipt_payload):
                differing = sorted(key for key, value in _comparable(receipt_payload).items()
                                   if _comparable(prior).get(key) != value)
                raise DevelopmentEstimationError(
                    f"{outdir} already holds a different run (differs on {differing}). "
                    "One capture and one input set per output directory.")
            for record in prior.get("outputs") or ():
                path = outdir / record["file"]
                if not path.exists() or _es.sha256_file(path) != record["sha256"]:
                    raise DevelopmentEstimationError(
                        f"{path} does not match its own receipt; refusing to overwrite "
                        "a result that drifted from its provenance")
            for name, text in files.items():
                path = outdir / name
                if path.exists() and path.read_text(encoding="utf-8") != text:
                    raise DevelopmentEstimationError(
                        f"{path} would change although the inputs did not; a rerun on "
                        "identical inputs must be byte-identical")
            return receipt_payload
    outdir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".estimate_development-", dir=outdir.parent))
    try:
        for name, text in files.items():
            (staging / name).write_text(text, encoding="utf-8")
        os.replace(staging, outdir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return receipt_payload


def write_results(outdir: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Serialise and publish atomically. Called only when the CLI is run."""
    payload = {k: v for k, v in result.items() if k != "receipt"}
    payload["receipt"] = result["receipt"]
    body = _es.dump_json(payload)
    receipt = {**result["receipt"],
               "outputs": [{"file": RESULT_NAME,
                            "sha256": hashlib.sha256(body.encode()).hexdigest()}]}
    return publish(Path(outdir), {RESULT_NAME: body, RECEIPT_NAME: _es.dump_json(receipt)})


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--indir", required=True, help="verified strategy panel directory")
    parser.add_argument("--boundaries", help="raw-capture mode only")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture-root", action="append", default=[])
    mode.add_argument("--frozen-counts", help="directory containing window_counts.parquet and receipt.json")
    parser.add_argument("--frozen-receipt-sha256", help="trusted measurement receipt digest, not read from the receipt")
    parser.add_argument("--source-audit", help="independent raw audit pinned by the measurement receipt")
    parser.add_argument("--source-root", help="recorded repository origin; frozen mode only, no origin fallback")
    parser.add_argument("--outdir", default=None,
                        help="publish the estimate and its receipt into this directory")
    parser.add_argument("--root", default=None)
    parser.add_argument("--study-config", default=None)
    parser.add_argument("--extension-config", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root) if args.root else ROOT
    try:
        result = estimate(
            Path(args.indir), boundaries=Path(args.boundaries) if args.boundaries else None,
            capture_roots=[Path(r) for r in args.capture_root], root=root,
            frozen_counts=Path(args.frozen_counts) if args.frozen_counts else None,
            frozen_receipt_sha256=args.frozen_receipt_sha256,
            source_audit=Path(args.source_audit) if args.source_audit else None,
            source_root=Path(args.source_root) if args.source_root else None,
            study_config=Path(args.study_config) if args.study_config
            else _es.DEFAULT_STRATEGY_CONFIG,
            extension_config=Path(args.extension_config) if args.extension_config
            else _es.DEFAULT_EXTENSION_CONFIG)
        if args.outdir:
            outdir = Path(args.outdir)
            if args.source_root:
                outdir = _ss.local_path(_es._resolve(root, outdir), root=root)
            write_results(outdir, result)
    except (DevelopmentEstimationError, _di.FrozenDevelopmentError, _es.EstimationError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except _sd.DevelopmentReadError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    except _ss.ScopedStudyError as exc:
        print(f"ScopedStudyError: {exc}", file=sys.stderr)
        return 4
    print(_es.dump_json({k: v for k, v in result.items() if k != "receipt"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
