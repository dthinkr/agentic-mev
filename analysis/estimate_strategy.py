"""Scoped estimation driver for one strategy's verified panel.

Reads a verified panel directory, runs the frozen paired design on it, and
publishes a receipt plus results. Nothing here invents an estimator: matching,
paired inference and the joint decomposition are
``strategy_study``/``submission_study``/``extension_study``, unchanged.

Results are written only when the CLI is actually run. ``estimate()`` returns
frames and statistics and writes nothing, so a test or a dry run cannot leave an
effect file behind. Publication is atomic: everything is staged beside the
output directory and moved into place, so a crash cannot leave a half-written
result next to a receipt that describes a different one.

What is fixed, and where it comes from
-------------------------------------
The configuration is ``config/strategy_study.json``. It is read, never
defaulted and never substituted: a missing file is an error rather than a quiet
fall back to the legacy ``config/study.json``, whose incident registry, blocker
set and registry-blocker policy differ and would silently change the sample.

The primary stratum is ``window="during"``, ``head_cut=50``,
``onset_shift_minutes=0``. The other window families, head cuts and the
onset-shift arm are diagnostics and are not run here.

Both frozen reference calendars are reported, neither designated primary:
``prior_days`` from ``control_offsets_days`` and ``prior_weeks`` from
``weekly_control_offsets_days``, each first on its own eligible support and then
again on the episodes common to both. The common-support pass filters each
calendar's already-matched rows, so every episode keeps the clean references it
was matched against; it is not a re-match.

``bootstrap_draws``, ``bootstrap_seed``, ``primary_cluster_days`` and
``sensitivity_cluster_days`` come from the strategy configuration;
``minimum_valid_bootstrap_fraction`` exists only in ``config/extension.json``
and is read from there. The strategy configuration names the weekly width
``primary_cluster_days`` while the extension helpers read ``cluster_days``, so
:func:`bootstrap_config` performs that translation explicitly rather than
letting a missing key fall back to a default. Weeks stay Monday-anchored because
``submission_study.cluster_labels`` anchors them.

Two supports, never mixed
-------------------------
Totals, including the total decomposition, run on every eligible matched
episode, so a thin cohort still gets one. The group comparison, the composition
share and the core/small decomposition run only on episodes whose *frozen
history* holds both tiers. Eligibility is never realised activity: an episode
whose populated tier traded nothing stays in, and its undefined ratios are
recorded rather than dropped.

No market, deployment or multi-pool stand-in is produced anywhere. Counts in the
output are diagnostic; this module asserts no causal claim and
``profit_status`` stays ``not_measured``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_HEAD_CUT = 50
PRIMARY_ONSET_SHIFT_MINUTES = 0
CALENDARS = ("prior_days", "prior_weeks")
CALENDAR_OFFSET_KEYS = {"prior_days": "control_offsets_days",
                        "prior_weeks": "weekly_control_offsets_days"}
BOOTSTRAP_FRACTION_KEY = "minimum_valid_bootstrap_fraction"
# Match the threshold in submission_study.paired_statistics (`n_boot >= 0.95 *
# draws`). The paired ratios and decomposition require the same minimum fraction
# of valid bootstrap draws; the test checks that both implementations agree.
PAIRED_BOOTSTRAP_FRACTION = 0.95
DEFAULT_STRATEGY_CONFIG = Path("config/strategy_study.json")
DEFAULT_EXTENSION_CONFIG = Path("config/extension.json")
RECEIPT_NAME = "estimation_receipt.json"
RESULT_NAME = "strategy_estimates.json"
PUBLISHED = (RESULT_NAME, RECEIPT_NAME)
TOTAL_TIER = "total"
GROUP_TIERS = ("core", "small")
CODE_MODULES = ("estimate_strategy", "strategy_study", "submission_study",
                "extension_study", "strategy_inputs", "build_strategy_panels",
                "competition_inputs")


def _sibling(name: str):
    if name in sys.modules:
        module = sys.modules[name]
        expected = (ROOT / f"analysis/{name}.py").resolve()
        found = getattr(module, "__file__", None)
        if found and Path(found).resolve() == expected:
            return module
    spec = importlib.util.spec_from_file_location(name, ROOT / f"analysis/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ss = _sibling("strategy_study")


class EstimationError(RuntimeError):
    """A provenance, support or estimand-contract violation. Never downgraded."""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: Any) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise EstimationError(
            f"{label} is missing: {path}. It is a frozen input and is never "
            "substituted or reconstructed here.")
    return json.loads(path.read_text(encoding="utf-8"))


def _exact_int(value: Any, name: str, *, minimum: Optional[int] = None) -> int:
    """An exact integer. ``int(7.9)`` truncating to 7 is refused, not rounded."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EstimationError(f"{name} must be an exact integer, got {value!r}")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise EstimationError(f"{name} must be an exact integer, got {value!r}")
        value = int(value)
    if minimum is not None and value < minimum:
        raise EstimationError(f"{name} must be at least {minimum}, got {value}")
    return int(value)


# --------------------------------------------------------------------------
# JSON that cannot smuggle a NaN or a stringified number
# --------------------------------------------------------------------------


def jsonable(value: Any) -> Any:
    """Plain JSON types. Non-finite floats become ``null``; numpy scalars unwrap.

    ``default=str`` would turn a float64 into ``"nan"`` or a numpy integer into
    ``"5"``, and the default encoder emits bare ``NaN``/``Infinity``, which is
    not JSON and which a reader may silently take as a number. Both are refused
    by converting here and dumping with ``allow_nan=False``.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else float(value)
    item = getattr(value, "item", None)
    if item is not None and getattr(value, "shape", None) == ():
        return jsonable(item())
    tolist = getattr(value, "tolist", None)
    if tolist is not None and not isinstance(value, (list, tuple, dict)):
        return jsonable(tolist())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in (sorted(value) if isinstance(value, (set, frozenset))
                                      else value)]
    raise EstimationError(f"refusing to serialise {type(value).__name__}: {value!r}")


def dump_json(payload: Any) -> str:
    return json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False)


# --------------------------------------------------------------------------
# Frozen settings
# --------------------------------------------------------------------------


def bootstrap_config(study_cfg: dict[str, Any], extension_cfg: dict[str, Any]
                     ) -> dict[str, Any]:
    """Assemble the bootstrap settings the extension helpers read."""
    missing = [key for key in ("bootstrap_draws", "bootstrap_seed",
                               "primary_cluster_days", "sensitivity_cluster_days")
               if key not in study_cfg]
    if missing:
        raise EstimationError(f"strategy configuration lacks {missing}")
    if BOOTSTRAP_FRACTION_KEY not in extension_cfg:
        raise EstimationError(
            f"{BOOTSTRAP_FRACTION_KEY} is not in the extension configuration; it is a "
            "frozen value and must not be invented here")
    draws = _exact_int(study_cfg["bootstrap_draws"], "bootstrap_draws", minimum=1)
    seed = _exact_int(study_cfg["bootstrap_seed"], "bootstrap_seed", minimum=0)
    primary = _exact_int(study_cfg["primary_cluster_days"], "primary_cluster_days",
                         minimum=0)
    fortnight = _exact_int(study_cfg["sensitivity_cluster_days"],
                           "sensitivity_cluster_days", minimum=0)
    for name, days in (("primary_cluster_days", primary),
                       ("sensitivity_cluster_days", fortnight)):
        if days not in (0, 7, 14):
            raise EstimationError(
                f"{name} is {days}; submission_study.cluster_labels supports only "
                "episodes (0), calendar weeks (7) and fortnights (14)")
    fraction = extension_cfg[BOOTSTRAP_FRACTION_KEY]
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) \
            or not 0 < float(fraction) <= 1:
        raise EstimationError(f"{BOOTSTRAP_FRACTION_KEY} must lie in (0, 1]")
    if float(fraction) != PAIRED_BOOTSTRAP_FRACTION:
        raise EstimationError(
            f"{BOOTSTRAP_FRACTION_KEY} is {fraction} in the extension configuration but "
            f"submission_study.paired_statistics hardcodes {PAIRED_BOOTSTRAP_FRACTION}. "
            "The paired ratios and decomposition require the same bootstrap-validity "
            "threshold; reconcile the two before estimating.")
    return {
        "bootstrap_draws": draws, "bootstrap_seed": seed,
        "cluster_days": primary, "primary_cluster_days": primary,
        "sensitivity_cluster_days": fortnight,
        BOOTSTRAP_FRACTION_KEY: float(fraction),
        "cluster_days_source": "strategy_config.primary_cluster_days",
        "bootstrap_fraction_source": "extension_config." + BOOTSTRAP_FRACTION_KEY,
        "week_anchor": "monday_via_submission_study.cluster_labels",
    }


def calendar_offsets(study_cfg: dict[str, Any]) -> dict[str, list]:
    out = {}
    for calendar, key in CALENDAR_OFFSET_KEYS.items():
        if key not in study_cfg:
            raise EstimationError(f"strategy configuration lacks {key}")
        offsets = [_exact_int(value, f"{key}[]", minimum=1) for value in study_cfg[key]]
        if not offsets or len(set(offsets)) != len(offsets):
            raise EstimationError(f"{key} must be distinct positive day offsets")
        out[calendar] = sorted(offsets)
    return out


# --------------------------------------------------------------------------
# Receipt, staging and publication
# --------------------------------------------------------------------------


def code_receipt(root: Path = ROOT) -> dict[str, str]:
    out = {}
    for name in CODE_MODULES:
        path = Path(root) / "analysis" / f"{name}.py"
        if not path.exists():
            raise EstimationError(f"a module this driver depends on is missing: {path}")
        out[name] = sha256_file(path)
    return out


def build_receipt(*, indir: Path, manifest: dict[str, Any], study_path: Path,
                  extension_path: Path, settings: dict[str, Any], strategy: str,
                  root: Path = ROOT, source_root: Optional[Path] = None) -> dict[str, Any]:
    resolution = {}
    if source_root is not None:
        for name in CODE_MODULES:
            _ss.local_path(Path(root) / "analysis" / f"{name}.py", root=root)
        resolution["path_resolution"] = {
            "recorded_root": str(source_root), "effective_root": str(Path(root).resolve()),
            "origin_fallback": False}
    inputs = {record["file"]: record["sha256"] for record in manifest["outputs"]}
    for name, recorded in sorted(inputs.items()):
        got = sha256_file(Path(indir) / name)
        if got != recorded:
            raise EstimationError(
                f"{name} hashes {got} but the panel manifest says {recorded}; the "
                "inputs are stale, rebuild the panel")
    return {
        **resolution, "driver": "estimate_strategy", "strategy": strategy,
        "outcome_scope": manifest["outcome_scope"],
        "input_directory": str(indir),
        "input_manifest_sha256": sha256_file(Path(indir) / "strategy_input_manifest.json"),
        "input_outputs": dict(sorted(inputs.items())),
        "panel_code_hashes": dict(sorted((manifest.get("code_hashes") or {}).items())),
        "panel_source_hashes": manifest.get("source_hashes"),
        "strategy_config": {"path": str(study_path), "sha256": sha256_file(study_path)},
        "extension_config": {"path": str(extension_path),
                             "sha256": sha256_file(extension_path)},
        "driver_code_hashes": code_receipt(root),
        "bootstrap": dict(settings),
        "primary_stratum": {"window": _ss.PRIMARY_WINDOW, "head_cut": PRIMARY_HEAD_CUT,
                            "onset_shift_minutes": PRIMARY_ONSET_SHIFT_MINUTES},
        "calendars": list(CALENDARS),
        "profit_status": "not_measured",
        "causal_claim": "none; counts and ratios are diagnostic",
    }


def _comparable(receipt: dict[str, Any]) -> dict[str, Any]:
    return {key: receipt.get(key) for key in
            ("strategy", "outcome_scope", "input_manifest_sha256", "input_outputs",
             "panel_code_hashes", "strategy_config", "extension_config",
             "driver_code_hashes", "bootstrap", "primary_stratum", "calendars")}


def publish(outdir: Path, files: dict[str, str]) -> dict[str, Any]:
    """Refuse an unknown or drifted directory, then publish atomically.

    An identical rerun is a verified no-op: the recorded hashes must still match
    what is on disk *and* the new bytes must be identical. Different bytes are
    refused even when the inputs agree, because a result that moved without its
    inputs moving is exactly the drift a receipt exists to catch.
    """
    outdir = Path(outdir)
    receipt_payload = json.loads(files[RECEIPT_NAME])
    if outdir.exists():
        present = {p.name for p in outdir.iterdir()}
        unknown = sorted(present - set(PUBLISHED))
        if unknown:
            raise EstimationError(
                f"{outdir} holds files this driver did not write ({unknown}); refusing "
                "to publish into a directory whose contents are unknown")
        existing_receipt = outdir / RECEIPT_NAME
        if present and not existing_receipt.exists():
            raise EstimationError(
                f"{outdir} holds {sorted(present)} but no {RECEIPT_NAME}; an orphan "
                "result has no provenance and will not be overwritten")
        if existing_receipt.exists():
            try:
                prior = json.loads(existing_receipt.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raise EstimationError(
                    f"{existing_receipt} cannot be read; move it aside") from None
            if _comparable(prior) != _comparable(receipt_payload):
                differing = sorted(key for key, value in _comparable(receipt_payload).items()
                                   if _comparable(prior).get(key) != value)
                raise EstimationError(
                    f"{outdir} already holds a different run (differs on {differing}). "
                    "One strategy and one input set per output directory.")
            # Same inputs. The results on disk must still be the ones the
            # receipt describes, and must still be byte-identical.
            for record in prior.get("outputs") or ():
                path = outdir / record["file"]
                if not path.exists():
                    raise EstimationError(
                        f"{path} is named by the existing receipt but missing; the "
                        "directory is inconsistent and will not be overwritten")
                if sha256_file(path) != record["sha256"]:
                    raise EstimationError(
                        f"{path} does not match its own receipt; refusing to overwrite "
                        "a result that drifted from its provenance")
            for name, text in files.items():
                path = outdir / name
                if path.exists() and path.read_text(encoding="utf-8") != text:
                    raise EstimationError(
                        f"{path} would change although the inputs did not; a rerun on "
                        "identical inputs must be byte-identical")
            return receipt_payload
    outdir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".estimate_strategy-", dir=outdir.parent))
    try:
        for name, text in files.items():
            (staging / name).write_text(text, encoding="utf-8")
        # One same-filesystem rename publishes both the result and its receipt.
        # A concurrently populated destination refuses instead of replacing it.
        os.replace(staging, outdir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return receipt_payload


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------


def _statistics_on(frame, loaded, settings, support_label: str, *,
                   allocation_events: Optional[list] = None) -> dict[str, Any]:
    """Total statistics on ``frame``, group statistics on the both-tier subset."""
    out = {}
    if frame.empty:
        return out
    support = _ss.allocation_support(frame, loaded["events"], loaded["cohorts"])
    events = (support["events"] if allocation_events is None
              else sorted(set(support["events"]) & set(allocation_events)))
    for metric in sorted(set(frame.metric)):
        rows = frame[frame.metric == metric]
        total = {"support": support_label, "events": sorted(set(rows.event_id)),
                 "n_events": int(rows.event_id.nunique())}
        allocation = {"support": f"{support_label}+historical_both_tier",
                      "available": bool(events), "reason": None if events
                      else support["reason"] or "no_common_both_tier_episode",
                      "events": events, "n_events": len(events),
                      "undefined_ratio_events": support["undefined_ratio_events"],
                      "membership_source": support["membership_source"]}
        for band, days in (("primary", settings["primary_cluster_days"]),
                           ("fortnight", settings["sensitivity_cluster_days"])):
            stats = _ss.paired_scoped(rows, metric, draws=settings["bootstrap_draws"],
                                      seed=settings["bootstrap_seed"], cluster_days=days)
            total[band] = {k: v for k, v in stats.items() if k in _ss.TOTAL_KEYS}
            if events:
                scoped = rows[rows.event_id.isin(events)]
                grouped = _ss.paired_scoped(scoped, metric,
                                            draws=settings["bootstrap_draws"],
                                            seed=settings["bootstrap_seed"],
                                            cluster_days=days)
                allocation[band] = {k: v for k, v in grouped.items()
                                    if k in _ss.ALLOCATION_KEYS}
            else:
                allocation[band] = None
        out[metric] = {"total": total, "allocation": allocation}
    return out


def _decompose_on(frame, loaded, cfg, settings, *, restrict: bool, tiers,
                  label: str) -> dict[str, Any]:
    """``extension_study.decompose`` through strategy_study, selecting tiers.

    The helper computes every tier; the selection here is which of its records
    belong to this support. No arithmetic is repeated.
    """
    if frame.empty:
        return {"support": label, "available": False,
                "reason": "no_matched_cells", "bands": {}}
    if restrict:
        support = _ss.allocation_support(frame, loaded["events"], loaded["cohorts"])
        if not support["available"]:
            return {"support": label, "available": False,
                    "reason": support["reason"], "bands": {}}
    bands = {}
    for band, days in (("primary", settings["primary_cluster_days"]),
                       ("fortnight", settings["sensitivity_cluster_days"])):
        joint = _ss.decomposition_scoped(
            frame, {**cfg, **settings}, loaded["events"],
            cohorts=loaded["cohorts"], cluster_days=days,
            restrict_to_allocation_support=restrict)
        bands[band] = {
            "events": (joint["allocation_support"]["events"] if restrict
                       else sorted(set(frame.event_id))),
            "components": [record for record in joint["components"]
                           if record["tier"] in tiers],
        }
    return {"support": label, "available": True, "reason": None, "bands": bands,
            "tiers": list(tiers),
            "identity": "Q = A * I, verified inside extension_study.decompose"}


def _calendar_block(loaded, calendar, offsets, settings, cfg):
    scoped = _ss.scoped_results(
        loaded["counts"], {**cfg, **settings}, loaded["events"],
        design=calendar, offsets=offsets, window=_ss.PRIMARY_WINDOW,
        head_cut=PRIMARY_HEAD_CUT, onset_shift_minutes=PRIMARY_ONSET_SHIFT_MINUTES,
        cohorts=loaded["cohorts"], treatment_offset=0,
        draws=settings["bootstrap_draws"], seed=settings["bootstrap_seed"],
        cluster_days=settings["primary_cluster_days"],
        sensitivity_cluster_days=settings["sensitivity_cluster_days"])
    matched = scoped["matched"]
    support = scoped["allocation_support"]
    block = {
        "calendar": calendar, "offsets": list(offsets), "window": scoped["window"],
        "head_cut": scoped["head_cut"],
        "onset_shift_minutes": scoped["onset_shift_minutes"],
        "treatment_offset": scoped["treatment_offset"],
        "statistics": scoped["statistics"], "allocation_support": support,
        "matched_events": sorted(set(matched.event_id)) if not matched.empty else [],
        "n_matched_cells": int(len(matched)),
        "dropped": _dropped_reasons(scoped["dropped"]),
        "excluded_input_rows": _excluded_reasons(scoped["excluded"]),
        # The total decomposition runs on every eligible matched episode, so a
        # thin cohort still gets one; the group decomposition runs on the
        # historical both-tier support only.
        "decomposition_total": _decompose_on(
            matched, loaded, cfg, settings, restrict=False, tiers=(TOTAL_TIER,),
            label="all_eligible_matched"),
        "decomposition_groups": _decompose_on(
            matched, loaded, cfg, settings, restrict=True, tiers=GROUP_TIERS,
            label="historical_both_tier_common"),
    }
    return block, matched


def _dropped_reasons(dropped) -> dict[str, Any]:
    if dropped is None or dropped.empty:
        return {"n": 0, "by_reason": {}, "events": []}
    counts = dropped.groupby("reason").size().to_dict()
    return {"n": int(len(dropped)),
            "by_reason": {str(k): int(v) for k, v in sorted(counts.items())},
            "events": sorted(set(dropped.event_id))}


def _excluded_reasons(excluded) -> dict[str, Any]:
    if excluded is None or excluded.empty:
        return {"n": 0, "by_reason": {}}
    counts = excluded.groupby("exclusion").size().to_dict()
    return {"n": int(len(excluded)),
            "by_reason": {str(k): int(v) for k, v in sorted(counts.items())}}


def estimate(indir: Path, *, root: Path = ROOT,
             study_config: Path = DEFAULT_STRATEGY_CONFIG,
             extension_config: Path = DEFAULT_EXTENSION_CONFIG,
             source_root: Optional[Path] = None) -> dict[str, Any]:
    """Run the frozen design on a verified panel. Returns results, writes nothing."""
    indir = _resolve(root, indir)
    study_path = _resolve(root, study_config)
    extension_path = _resolve(root, extension_config)
    if source_root is not None:
        for path in (indir, study_path, extension_path):
            _ss.local_path(path, root=root)
    cfg = _read_json(study_path, label="strategy configuration")
    extension_cfg = _read_json(extension_path, label="extension configuration")
    settings = bootstrap_config(cfg, extension_cfg)
    offsets = calendar_offsets(cfg)

    loaded = _ss.load_scoped_inputs(indir, root=root, config_path=study_path,
                                    source_root=source_root)
    manifest = loaded["manifest"]
    strategy = manifest["strategy"]
    receipt = build_receipt(indir=indir, manifest=manifest, study_path=study_path,
                            extension_path=extension_path, settings=settings,
                            strategy=strategy, root=root, source_root=source_root)

    calendars, matched_frames = {}, {}
    for calendar in CALENDARS:
        block, matched = _calendar_block(loaded, calendar, offsets[calendar], settings, cfg)
        calendars[calendar] = block
        matched_frames[calendar] = matched

    matched_sets = {name: set(block["matched_events"]) for name, block in calendars.items()}
    common = sorted(set.intersection(*matched_sets.values())) if matched_sets else []
    # Both calendars estimated again on the common episodes. Each calendar's
    # already-matched rows are filtered, so every episode keeps the clean
    # references it was matched against; this is not a re-match.
    cross = {"role": "common_support_comparison",
             "matched_in_both": common, "n_matched_in_both": len(common),
             "matched_only_in": {name: sorted(events - set(common))
                                 for name, events in matched_sets.items()},
             "calendars": {}}
    for calendar, matched in matched_frames.items():
        frame = matched[matched.event_id.isin(common)] if common else matched.iloc[0:0]
        cross["calendars"][calendar] = {
            "n_matched_cells": int(len(frame)),
            "statistics": _statistics_on(frame, loaded, settings,
                                         "common_to_both_calendars"),
            "decomposition_total": _decompose_on(
                frame, loaded, cfg, settings, restrict=False, tiers=(TOTAL_TIER,),
                label="common_to_both_calendars"),
            "decomposition_groups": _decompose_on(
                frame, loaded, cfg, settings, restrict=True, tiers=GROUP_TIERS,
                label="common_to_both_calendars+historical_both_tier"),
        }
    return {
        "receipt": receipt, "strategy": strategy,
        "primary_stratum": receipt["primary_stratum"], "bootstrap": settings,
        "calendars": calendars, "cross_calendar": cross,
        "undefined_ratio_events": {
            name: block["allocation_support"]["undefined_ratio_events"]
            for name, block in calendars.items()},
        "market_outcomes": None, "deployment_outcomes": None,
        "multi_pool_outcomes": None, "profit_status": "not_measured",
        "causal_claim": "none; counts and ratios are diagnostic",
    }


def write_results(outdir: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Serialise and publish atomically. Called only when the CLI is actually run."""
    payload = {k: v for k, v in result.items() if k != "receipt"}
    payload["receipt"] = result["receipt"]
    body = dump_json(payload)
    receipt = {**result["receipt"],
               "outputs": [{"file": RESULT_NAME,
                            "sha256": hashlib.sha256(body.encode()).hexdigest()}]}
    return publish(Path(outdir), {RESULT_NAME: body, RECEIPT_NAME: dump_json(receipt)})


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", required=True, type=Path,
                        help="a verified strategy panel directory")
    parser.add_argument("--outdir", required=True, type=Path,
                        help="one strategy and one input set per directory")
    parser.add_argument("--study-config", type=Path, default=DEFAULT_STRATEGY_CONFIG)
    parser.add_argument("--extension-config", type=Path, default=DEFAULT_EXTENSION_CONFIG)
    parser.add_argument("--source-root", type=Path,
                        help="recorded repository root to map into this checkout, without fallback")
    parser.add_argument("--dry-run", action="store_true",
                        help="estimate in memory and write nothing")
    args = parser.parse_args(argv)
    try:
        result = estimate(args.inputs, study_config=args.study_config,
                          extension_config=args.extension_config, source_root=args.source_root)
        written = None if args.dry_run else write_results(args.outdir, result)
    except (EstimationError, _ss.ScopedStudyError) as exc:
        print(f"estimate_strategy: {exc}", file=sys.stderr)
        return 2
    summary = {
        "strategy": result["strategy"], "primary_stratum": result["primary_stratum"],
        "calendars": {name: {"n_matched_cells": block["n_matched_cells"],
                             "n_matched_events": len(block["matched_events"]),
                             "allocation_available":
                                 block["allocation_support"]["available"],
                             "n_allocation_events":
                                 block["allocation_support"]["n_events"]}
                      for name, block in result["calendars"].items()},
        "common_support_events": result["cross_calendar"]["n_matched_in_both"],
        "wrote": sorted(record["file"] for record in (written or {}).get("outputs", [])),
        "dry_run": bool(args.dry_run), "profit_status": result["profit_status"],
    }
    print(dump_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
