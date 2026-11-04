"""Outage effects on deployment, by outcome definition, calendar period, provider and horizon.

Pre-registered on 2026-10-27 before any estimate was seen:

H1  During a provider outage, deployment by trading-bot operators falls (rate ratio below one).
H2  The fall deepens over calendar time as AI coding tools are adopted: the 2026 rate ratio is
    below the 2025 rate ratio (ratio of rate ratios below one), on the same outcome and window.

Design, fixed in advance and shared with the manuscript's outage test: the liquidation outage panel
(`.build/strategy_panels/liquidation/strategy_window_counts.parquet`, core tier, head cut 50,
retain_eligible, analysis_valid), both reference calendars (nearby days `prior_days`, same weekday
`prior_weeks`), three blocker screens (original = every analysis-valid episode; source-aware = the
matched events of the source-aware calendars; bounded = the original screen minus every window that
overlaps one of the four bounded censoring intervals in BOUNDED_CENSOR), the 4,000-draw Monday-week cluster bootstrap and the
minimum of two clean references per episode, all taken from `build_cross_strategy_summary`.

What varies, each reported separately and never selected among:
  outcomes    creations (count of top-level contract creations in the window),
              creating_senders (distinct addresses that create in the window),
              capped_creations (creations with each sender capped at 3 per window, a lumpiness guard
              added after the first run showed bulk deployers dominating the count ratios),
              new_code_creations (creations whose runtime code hash is seen for the first time,
              defined only where the bytecode table covers the window),
              deploying_addresses_share (creating_senders divided by the number of population
              addresses with any deployment in the span, a participation rate),
              deployments_per_deployer (creations divided by creating_senders, undefined and left
              out of the estimate for windows without a deployer);
  populations verified (addresses verified as MEV operators in the ledgers),
              archive (every address in the creation archive);
  windows     lead6_3, lead3_1, pre1, during, outage_plus_24h (onset to 24 h after resolution),
              post24, recovery_equal;
  splits      all episodes, onset half-years 2025H1/2025H2/2026H1/2026H2, provider.

Outputs `results/outage_deployment_heterogeneity/{summary.json,table.md,receipt.json}`.

(2026-10-27, after the adversarial design review, before the first corrected estimate).
The window the earlier docstring called onset-to-24-hours is onset to 24 hours after resolution; it is
renamed `outage_plus_24h` in every output (its bounds come from the frozen panel and are unchanged).
A fourth screen, `isolated`, is the window design's primary screen: starting from the original
screen it drops every episode whose treated (`during`) window overlaps a record of the cloud table
(`data/cloud_twin_window.csv`) or an incident of the other provider in the registry
(`data/support/incident_registry.csv`), so treatment and references are screened symmetrically; the
number of episodes dropped and the reason are reported under `episodes`. Inference rule: an estimate
reports `rr` and `ci` only when its matched split has at least five episodes, at least four distinct
Monday-week clusters among the treated windows and a bootstrap valid-draw fraction (draws with a
positive reference denominator, same seed and draw count as `boot_rr`) of at least 0.95; otherwise
`rr`, `ci`, `lo`, `hi` and the exclusion percentages are null and the row carries `suppressed_reason`
while `n`, `obs` and `ref` stay visible. A qualifying cell whose `boot_rr` interval is undefined
because a few draws (under five percent) divided by zero takes its percentile interval over the
draws with a positive denominator, replayed with the same seed and cluster order, and says so in `note`. The 2026-over-2025 ratio of rate ratios is reported only
when both sides qualify and at least 95 percent of its own draws are finite. Two outcomes of record
are added beside the existing ones, `deploying_addresses_share` and `deployments_per_deployer`, and
the ratio of rate ratios is computed for creations, creating_senders, capped_creations and
deployments_per_deployer. The receipt hashes every input read, including the cloud table, the
registry and the population files when given.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bcss", ROOT / "analysis/build_cross_strategy_summary.py")
S = importlib.util.module_from_spec(_spec)
sys.modules["bcss"] = S
_spec.loader.exec_module(S)

OUT_DIR = Path("results/outage_deployment_heterogeneity")
CAP = 3  # creations per sender per window counted in capped_creations
WINDOW_RENAMES = {"cumulative24": "outage_plus_24h"}  # frozen-panel label -> reported label
WINDOWS = ("lead6_3", "lead3_1", "pre1", "during", "outage_plus_24h", "post24", "recovery_equal")
OUTCOMES = ("creations", "creating_senders", "capped_creations", "new_code_creations", "deploying_addresses_share", "deployments_per_deployer")
H2_OUTCOMES = ("creations", "creating_senders", "capped_creations", "deployments_per_deployer")
SCREENS = ("isolated", "orig", "sa", "bounded")  # isolated first: the window design's primary screen
PERIODS = (("2025H1", "2025-01-01", "2025-07-01"), ("2025H2", "2025-07-01", "2026-01-01"),
           ("2026H1", "2026-01-01", "2026-07-01"), ("2026H2", "2026-07-01", "2027-01-01"))
YEARS = (("2025", "2025-01-01", "2026-01-01"), ("2026", "2026-01-01", "2027-01-01"))
# Bounded-censoring sensitivity screen, fixed after the primary-source check of 2026-10-27
# (reviews/blocker_sources/aws_unresolved_records_2026-10-27.md), before any estimate under it was seen.
# The source-aware screen censors reference availability for four AWS records through the study end.
# This variant bounds each censoring interval instead: the two Middle East regional losses are treated
# as a level shift after 14 days (present in treated and reference windows alike), the us-east-1
# impairment is closed at the last vendor report of residual trouble, and Bedrock access is closed at
# the end of the day AWS states access was restored.
BOUNDED_CENSOR = {
    "AWS_MULTIPLE_SERVICES_OPERATIONAL_ISSUE_5E6B8_EF2498889B5": ("2026-04-29T00:00:00Z", 14, None),
    "AWS_MULTIPLE_SERVICES_OPERATIONAL_ISSUE_E6426_01AD72B4E1E": ("2026-04-29T00:00:00Z", 14, None),
    "AWS_EC2_OPERATIONAL_ISSUE_FFA94_A4DC515D30F": ("2026-05-08T00:25:00Z", None, "2026-05-13T00:00:00Z"),
    "AWS_BEDROCK_OPERATIONAL_ISSUE_588F5_70C147423A3": ("2026-06-13T01:26:00Z", None, "2026-07-02T00:00:00Z"),
}
CLOUD_TABLE = Path("data/cloud_twin_window.csv")
REGISTRY = Path("data/support/incident_registry.csv")
# inference rule.
MIN_EPISODES, MIN_CLUSTERS, MIN_VALID_FRACTION = 5, 4, 0.95
HYPOTHESES = {
    "H1": "during-outage deployment rate ratio below one",
    "H2": "ratio of the 2026 to the 2025 during-outage rate ratio below one (effect deepens with adoption)",
    "registered": "2026-10-27, before any estimate from this script was seen",
}
specification_2 = {
    "primary_screen": "isolated",
    "window_rename": WINDOW_RENAMES,
    "inference_rule": f"rr and ci only with >= {MIN_EPISODES} episodes, >= {MIN_CLUSTERS} Monday-week clusters and a valid-draw fraction >= {MIN_VALID_FRACTION}",
    "outcomes_added": ["deploying_addresses_share", "deployments_per_deployer"],
    "registered": "2026-10-27, before the first corrected estimate",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def period_label(stamp_s: int) -> str:
    t = pd.Timestamp(stamp_s, unit="s", tz="UTC")
    for label, lo, hi in PERIODS:
        if pd.Timestamp(lo, tz="UTC") <= t < pd.Timestamp(hi, tz="UTC"):
            return label
    return "outside"


def load_windows(root: Path) -> dict[str, pd.DataFrame]:
    w = pd.read_parquet(root / S.WINDOW_COUNTS)
    base = ((w["tier"] == "core") & (w["head_cut"] == 50) & (w["qualification_rule"] == "min_tx_1")
            & (w["treatment_policy"] == "retain_eligible") & (w["analysis_valid"] == True))  # noqa: E712
    cols = ["event_id", "window", "offset_days", "window_start_s", "window_end_s", "start_s", "end_s"]
    out = {}
    for design in ("prior_days", "prior_weeks"):
        wdf = w[base & (w["design"] == design)][cols].drop_duplicates().copy()
        wdf["window"] = wdf["window"].map(lambda v: WINDOW_RENAMES.get(v, v))
        wdf["provider"] = wdf["event_id"].str.split("_").str[0]
        wdf["period"] = [period_label(s) for s in wdf["start_s"]]
        wdf["year"] = wdf["period"].str[:4]
        out[design] = wdf.reset_index(drop=True)
    return out


def bounded_intervals(root: Path) -> list[tuple[int, int]]:
    """The four censoring intervals in epoch seconds, starts taken from the blocker table itself."""
    cloud = pd.read_csv(root / CLOUD_TABLE)
    out = []
    for event_id, (fallback_start, days, end) in BOUNDED_CENSOR.items():
        row = cloud[cloud["id"] == event_id]
        start = pd.Timestamp(row["start"].iloc[0], tz="UTC") if len(row) else pd.Timestamp(fallback_start)
        stop = start + pd.Timedelta(days=days) if days is not None else pd.Timestamp(end)
        out.append((int(start.timestamp()), int(stop.timestamp())))
    return out


def bounded_screen(frame: pd.DataFrame, intervals: list[tuple[int, int]]) -> pd.DataFrame:
    """Drop every window (treated or reference) that overlaps a bounded censoring interval; episodes
    whose treated window is dropped leave the screen, and the two-clean-references rule applies as usual."""
    ws, we = frame["window_start_s"].to_numpy(), frame["window_end_s"].to_numpy()
    hit = np.zeros(len(frame), dtype=bool)
    for lo, hi in intervals:
        hit |= (ws < hi) & (we > lo)
    kept = frame[~hit]
    treated_ok = set(kept.loc[kept["offset_days"] == 0, "event_id"])
    return kept[kept["event_id"].isin(treated_ok)]


def cloud_intervals(root: Path) -> list[tuple[int, int]]:
    """Every record of the cloud table as an epoch-second interval."""
    cloud = pd.read_csv(root / CLOUD_TABLE)
    start = pd.to_datetime(cloud["start"], utc=True, format="ISO8601"); end = pd.to_datetime(cloud["end"], utc=True, format="ISO8601")
    return [(int(a.timestamp()), int(b.timestamp())) for a, b in zip(start, end) if pd.notna(a) and pd.notna(b)]


def registry_intervals(root: Path) -> dict[str, list[tuple[int, int]]]:
    """Every registry record as an epoch-second interval, keyed by provider."""
    reg = pd.read_csv(root / REGISTRY)
    start = pd.to_datetime(reg["start"], utc=True, format="ISO8601"); end = pd.to_datetime(reg["end"], utc=True, format="ISO8601")
    out: dict[str, list[tuple[int, int]]] = {}
    for prov, a, b in zip(reg["provider"].astype(str), start, end):
        if pd.notna(a) and pd.notna(b):
            out.setdefault(prov, []).append((int(a.timestamp()), int(b.timestamp())))
    return out


def _overlaps(s: int, e: int, intervals: list[tuple[int, int]]) -> bool:
    return any(s < hi and e > lo for lo, hi in intervals)


def isolated_screen(frame: pd.DataFrame, cloud: list[tuple[int, int]], registry: dict[str, list[tuple[int, int]]]) -> tuple[pd.DataFrame, dict]:
    """Starting from the original screen, drop every episode whose treated `during` window overlaps a
    cloud record or an incident of the other provider. Returns the kept frame and a drop report."""
    treated = frame[(frame["offset_days"] == 0) & (frame["window"] == "during")].drop_duplicates("event_id")
    dropped: dict[str, list[str]] = {}
    for eid, prov, s, e in zip(treated["event_id"], treated["provider"], treated["window_start_s"], treated["window_end_s"]):
        reasons = []
        if _overlaps(int(s), int(e), cloud):
            reasons.append("cloud")
        others = [iv for p, ivs in registry.items() if p != prov for iv in ivs]
        if _overlaps(int(s), int(e), others):
            reasons.append("other_provider")
        if reasons:
            dropped[eid] = reasons
    kept = frame[~frame["event_id"].isin(dropped)]
    report = {"episodes_before": int(treated["event_id"].nunique()), "episodes_after": int(treated["event_id"].nunique() - len(dropped)),
              "dropped": {"cloud": sum("cloud" in r for r in dropped.values()),
                          "other_provider": sum("other_provider" in r for r in dropped.values()),
                          "both": sum(len(r) == 2 for r in dropped.values()), "total": len(dropped)},
              "dropped_episodes": {k: dropped[k] for k in sorted(dropped)}}
    return kept, report


def verified_addresses(root: Path) -> set[bytes]:
    df_sw, _ = S.load_ledger([root / p for p in S.DEFAULT_SWAP_ROOTS])
    df_liq, _ = S.load_ledger([root / p for p in S.DEFAULT_LIQUIDATION_ROOTS], dedupe=True)
    senders = pd.concat([df_sw["sender"], df_liq["sender"]]).str.lower().unique()
    return {bytes.fromhex(h[2:]) for h in senders if len(h) == 42}


def creation_series(cr: pd.DataFrame, cb: pd.DataFrame) -> pd.DataFrame:
    """Creations sorted by time with a sender code and a first-deployment-of-this-code flag."""
    cr = cr.sort_values(["ts", "hash"]).reset_index(drop=True).copy()
    cb = cb[cb["error"].isna() | (cb["error"] == "")] if "error" in cb else cb
    first_seen = cb.groupby("code_hash")["block_number"].min()
    code = cb[["tx_hash", "code_hash", "block_number", "is_proxy_min"]].drop_duplicates("tx_hash")
    code = code.merge(first_seen.rename("first_block"), left_on="code_hash", right_index=True)
    code["new_code"] = (code["block_number"] == code["first_block"]) & (~code["is_proxy_min"].astype(bool))
    cr["hash_hex"] = cr["hash"].map(lambda h: "0x" + h.hex() if isinstance(h, (bytes, bytearray)) else str(h).lower())
    code["tx_hash"] = code["tx_hash"].astype(str).str.lower()
    cr = cr.merge(code[["tx_hash", "new_code"]], left_on="hash_hex", right_on="tx_hash", how="left")
    cr["has_bytecode"] = cr["tx_hash"].notna()
    cr["new_code"] = cr["new_code"].fillna(False).astype(bool)
    cr["sender_code"] = pd.factorize(cr["from"])[0]
    return cr


def count_windows(wdf: pd.DataFrame, cr: pd.DataFrame) -> pd.DataFrame:
    ts = cr["ts"].to_numpy(dtype=np.int64)
    sender = cr["sender_code"].to_numpy()
    newc = cr["new_code"].to_numpy()
    lo = np.searchsorted(ts, wdf["window_start_s"].to_numpy(), side="left")
    hi = np.searchsorted(ts, wdf["window_end_s"].to_numpy(), side="left")
    out = wdf.copy()
    out["creations"] = hi - lo
    out["creating_senders"] = [len(np.unique(sender[a:b])) if b > a else 0 for a, b in zip(lo, hi)]
    out["new_code_creations"] = [int(newc[a:b].sum()) if b > a else 0 for a, b in zip(lo, hi)]
    # Lumpiness guard: each sender's creations inside a window count at most CAP, so one bulk deployer cannot move a ratio.
    out["capped_creations"] = [int(np.minimum(np.unique(sender[a:b], return_counts=True)[1], CAP).sum()) if b > a else 0
                               for a, b in zip(lo, hi)]
    # outcomes of record: participation rate over the population's deployers in the span,
    # and deployments per deployer (undefined where no address deploys).
    deployers = int(len(np.unique(sender))) if len(sender) else 0
    senders = out["creating_senders"].to_numpy(dtype=float)
    out["deploying_addresses_share"] = senders / deployers if deployers else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        out["deployments_per_deployer"] = np.where(senders > 0, out["creations"].to_numpy(dtype=float) / senders, np.nan)
    return out


def usable(df_win: pd.DataFrame, col: str) -> pd.DataFrame:
    """Rows that carry a value of the outcome (deployments_per_deployer is undefined without a deployer)."""
    return df_win[df_win[col].notna()] if col == "deployments_per_deployer" else df_win


def matched(df_win: pd.DataFrame, col: str) -> pd.DataFrame:
    """The treated windows `boot_rr` uses (at least two clean references, treated order preserved),
    with the reference mean and the Monday-week cluster id of each."""
    treated = df_win[df_win["offset_days"] == 0]
    refs = df_win[df_win["offset_days"] > 0]
    vals = refs[col].to_numpy(dtype=float)
    codes, uniq = pd.factorize(refs["event_id"].to_numpy())
    ok = ~np.isnan(vals)
    n_clean = np.bincount(codes[ok], minlength=len(uniq))
    sums = np.bincount(codes[ok], weights=vals[ok], minlength=len(uniq))
    good = n_clean >= S.BOOTSTRAP_MIN_CLEAN_CONTROLS
    ref_mean = dict(zip(uniq[good], sums[good] / np.maximum(n_clean[good], 1)))
    m = treated[treated["event_id"].isin(ref_mean)].copy()
    m["ref_mean"] = m["event_id"].map(ref_mean).astype(float)
    m["week"] = (m["start_s"].to_numpy(dtype=np.int64) // 86400 + 3) // 7   # Monday-week id (1970-01-05 is week 1)
    return m


def bootstrap_draws(m: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """`boot_rr`'s draws replayed on a matched frame (same seed, draw count and cluster order): the
    resampled ratio of each draw and its reference denominator."""
    codes, uniq = pd.factorize(m["week"].to_numpy())
    c_obs = np.bincount(codes, weights=m[col].to_numpy(dtype=float), minlength=len(uniq))
    c_ref = np.bincount(codes, weights=m["ref_mean"].to_numpy(dtype=float), minlength=len(uniq))
    rng = np.random.default_rng(S.BOOTSTRAP_SEED)
    idx = rng.integers(0, len(uniq), size=(S.BOOTSTRAP_DRAWS, len(uniq)))
    den = c_ref[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return c_obs[idx].sum(axis=1) / den, den


def valid_draw_fraction(m: pd.DataFrame, col: str = "ref_mean") -> float:
    """Share of `boot_rr`'s draws whose resampled reference denominator is positive."""
    if m.empty:
        return 0.0
    _, den = bootstrap_draws(m, col if col in m else "ref_mean")
    return float((den > 0).mean())


def finite_interval(m: pd.DataFrame, col: str) -> dict:
    """Percentile interval over the draws with a positive reference denominator, for a qualifying cell
    whose `boot_rr` interval is undefined because a few draws divided by zero."""
    draws, den = bootstrap_draws(m, col)
    kept = draws[den > 0]
    lo, hi = np.quantile(kept, [0.025, 0.975])
    lo_r, hi_r = round(float(lo), 4), round(float(hi), 4)
    return {"ci": f"[{lo:.2f}, {hi:.2f}]", "lo": lo_r, "hi": hi_r, "excl_decline_pct": round((1 - lo_r) * 100, 1),
            "excl_increase_pct": round((hi_r - 1) * 100, 1), "note": f"interval over the {len(kept)} draws with a positive reference denominator"}


def qualification(m: pd.DataFrame, col: str) -> tuple[dict, str | None]:
    """inference rule on a matched frame: counts and the suppression reason, if any."""
    n, clusters, valid = int(len(m)), int(m["week"].nunique()) if len(m) else 0, valid_draw_fraction(m, col)
    reasons = []
    if n < MIN_EPISODES:
        reasons.append(f"episodes<{MIN_EPISODES}")
    if clusters < MIN_CLUSTERS:
        reasons.append(f"clusters<{MIN_CLUSTERS}")
    if valid < MIN_VALID_FRACTION:
        reasons.append(f"valid_draws<{MIN_VALID_FRACTION}")
    return {"clusters": clusters, "valid_draw_fraction": round(valid, 4)}, ("; ".join(reasons) or None)


def safe_boot(df_win: pd.DataFrame, col: str) -> dict:
    """boot_rr, returning an empty estimate instead of dividing by a zero reference sum."""
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            return S.boot_rr(df_win, col)
    except ZeroDivisionError:
        treated = df_win[df_win["offset_days"] == 0]
        return {"n": int(len(treated)), "obs": int(treated[col].sum()) if len(treated) else 0, "ref": 0.0, "rr": None,
                "ci": None, "lo": None, "hi": None, "excl_decline_pct": None, "excl_increase_pct": None,
                "note": "zero reference count"}


def estimate(df_win: pd.DataFrame, col: str) -> dict:
    """boot_rr under the inference rule, with the cluster count and valid-draw fraction."""
    df_win = usable(df_win, col)
    est = safe_boot(df_win, col)
    m = matched(df_win, col)
    stats, reason = qualification(m, col)
    est.update(stats)
    if col in ("deploying_addresses_share", "deployments_per_deployer") and len(m):
        est["obs"] = round(float(m[col].sum()), 4)  # boot_rr truncates obs to an integer; these outcomes are not counts
    if reason:
        for k in ("rr", "ci", "lo", "hi", "excl_decline_pct", "excl_increase_pct"):
            est[k] = None
    elif est.get("lo") is None or not np.isfinite(est["lo"]) or not np.isfinite(est["hi"]):
        est.update(finite_interval(m, col))  # qualifying cell whose boot_rr quantile was undefined
    est["suppressed_reason"] = reason
    return est


def ratio_of_ratios(frame_a: pd.DataFrame, frame_b: pd.DataFrame, col: str) -> dict:
    """Bootstrap of RR_b / RR_a with independent Monday-week clusters in each frame, reported only when
    both sides pass the inference rule and at least MIN_VALID_FRACTION of the ratio draws are finite."""
    frame_a, frame_b = usable(frame_a, col), usable(frame_b, col)

    def prepared(df):
        m = matched(df, col)
        codes, uniq = pd.factorize(m["week"].to_numpy())
        return (np.bincount(codes, weights=m[col].to_numpy(dtype=float), minlength=len(uniq)),
                np.bincount(codes, weights=m["ref_mean"].to_numpy(dtype=float), minlength=len(uniq)), m)
    oa, ra, ma = prepared(frame_a)
    ob, rb, mb = prepared(frame_b)
    qa, reason_a = qualification(ma, col)
    qb, reason_b = qualification(mb, col)
    out = {"n_a": int(len(ma)), "n_b": int(len(mb)), "clusters_a": qa["clusters"], "clusters_b": qb["clusters"],
           "valid_draw_fraction_a": qa["valid_draw_fraction"], "valid_draw_fraction_b": qb["valid_draw_fraction"],
           "rr_a": None, "rr_b": None, "ratio": None, "ci": None, "lo": None, "hi": None, "valid_draws": 0, "valid_draw_fraction": 0.0}
    reasons = [f"2025: {reason_a}"] if reason_a else []
    reasons += [f"2026: {reason_b}"] if reason_b else []
    if len(ma) == 0 or len(mb) == 0 or ra.sum() == 0 or rb.sum() == 0:
        out["suppressed_reason"] = "; ".join(reasons) or "zero reference count"
        return out
    rng = np.random.default_rng(S.BOOTSTRAP_SEED)
    ia = rng.integers(0, len(oa), size=(S.BOOTSTRAP_DRAWS, len(oa)))
    ib = rng.integers(0, len(ob), size=(S.BOOTSTRAP_DRAWS, len(ob)))
    with np.errstate(divide="ignore", invalid="ignore"):
        da = oa[ia].sum(axis=1) / ra[ia].sum(axis=1)
        db = ob[ib].sum(axis=1) / rb[ib].sum(axis=1)
        draws = db / da
    draws = draws[np.isfinite(draws)]
    fraction = len(draws) / S.BOOTSTRAP_DRAWS
    out.update({"valid_draws": int(len(draws)), "valid_draw_fraction": round(fraction, 4)})
    if fraction < MIN_VALID_FRACTION:
        reasons.append(f"ratio: valid_draws<{MIN_VALID_FRACTION}")
    if not reason_a:
        out["rr_a"] = round(float(oa.sum() / ra.sum()), 3)
    if not reason_b:
        out["rr_b"] = round(float(ob.sum() / rb.sum()), 3)
    if reasons:
        out["suppressed_reason"] = "; ".join(reasons)
        return out
    lo, hi = np.quantile(draws, [0.025, 0.975])
    out.update({"ratio": round(float((ob.sum() / rb.sum()) / (oa.sum() / ra.sum())), 3),
                "ci": f"[{lo:.2f}, {hi:.2f}]", "lo": round(float(lo), 4), "hi": round(float(hi), 4), "suppressed_reason": None})
    return out


def load_populations(root: Path, cr_all: pd.DataFrame, populations: Path | None, verified_file: Path | None = None) -> dict[str, pd.DataFrame]:
    """Default: the ledger-verified addresses and the whole archive. With ``--populations``, a JSON
    object mapping a population name to a file holding a JSON list of 0x addresses; each population
    is the creations of those addresses in the supplied creation table. ``verified_file`` reuses a
    previously written address list (the dose panel's ``verified_addresses.json``) where the ledgers
    are not mounted."""
    if populations is None:
        if verified_file is not None:
            verified = {bytes.fromhex(h[2:].lower()) for h in json.loads((root / verified_file).read_text())}
        else:
            verified = verified_addresses(root)
        return {"verified": cr_all[cr_all["from"].isin(verified)], "archive": cr_all}
    spec = json.loads(populations.read_text())
    out = {}
    for name, path in spec.items():
        addrs = {bytes.fromhex(h[2:].lower()) for h in json.loads((root / path).read_text())}
        out[name] = cr_all[cr_all["from"].isin(addrs)]
    return out


def build(root: Path, creations: Path | None = None, populations: Path | None = None, verified_file: Path | None = None) -> dict:
    screen = json.loads((root / S.SOURCE_SCREEN).read_text())
    calendars = screen["results"]["trading"]["source_aware"]["calendars"]
    sa_eids = {"prior_days": set(calendars["prior_days"]["matched_events"]),
               "prior_weeks": set(calendars["prior_weeks"]["matched_events"])}
    windows = load_windows(root)
    censor = bounded_intervals(root)
    cloud, registry = cloud_intervals(root), registry_intervals(root)
    isolated = {d: isolated_screen(w, cloud, registry) for d, w in windows.items()}
    cr_all = pd.read_parquet(root / (creations or S.CREATIONS_ARCHIVE))
    cb = pd.read_parquet(root / S.BYTECODE_TABLE)
    populations_frames = load_populations(root, cr_all, populations, verified_file)
    series = {name: creation_series(frame, cb) for name, frame in populations_frames.items()}
    union = pd.concat(series.values()) if series else cr_all.iloc[0:0]
    covered = union[union["has_bytecode"]] if "has_bytecode" in union else union.iloc[0:0]
    bytecode_span = (int(covered["ts"].min()), int(covered["ts"].max())) if len(covered) else None

    results, rows = {}, []
    for pop, cr in series.items():
        counted = {d: count_windows(wdf, cr) for d, wdf in windows.items()}
        for design, dkey in (("prior_days", "pd"), ("prior_weeks", "pw")):
            frame_all = counted[design]
            isolated_eids = set(isolated[design][0]["event_id"])
            for screen_key in SCREENS:
                if screen_key == "bounded":
                    frame = bounded_screen(frame_all, censor)
                elif screen_key == "sa":
                    frame = frame_all[frame_all["event_id"].isin(sa_eids[design])]
                elif screen_key == "isolated":
                    frame = frame_all[frame_all["event_id"].isin(isolated_eids)]
                else:
                    frame = frame_all
                splits = {"all": frame}
                for label, _, _ in PERIODS:
                    splits[label] = frame[frame["period"] == label]
                for label, _, _ in YEARS:
                    splits[label] = frame[frame["year"] == label]
                for prov in sorted(frame["provider"].unique()):
                    splits[prov] = frame[frame["provider"] == prov]
                for window in WINDOWS:
                    for outcome in OUTCOMES:
                        for split, sub in splits.items():
                            wsub = sub[sub["window"] == window]
                            if outcome == "new_code_creations" and bytecode_span is not None:
                                inside = (wsub["window_start_s"] >= bytecode_span[0]) & (wsub["window_end_s"] <= bytecode_span[1])
                                keep = set(wsub.loc[(wsub["offset_days"] == 0) & inside, "event_id"])
                                wsub = wsub[wsub["event_id"].isin(keep) & inside]
                            est = estimate(wsub, outcome)
                            key = f"{pop}|{screen_key}_{dkey}|{window}|{outcome}|{split}"
                            results[key] = est
                            rows.append({"population": pop, "screen": screen_key, "calendar": dkey, "window": window,
                                         "outcome": outcome, "split": split, **est})
                    # H2: 2026 against 2025 on the same window and outcome
                    for outcome in H2_OUTCOMES:
                        a = splits["2025"][splits["2025"]["window"] == window]
                        b = splits["2026"][splits["2026"]["window"] == window]
                        results[f"{pop}|{screen_key}_{dkey}|{window}|{outcome}|2026_over_2025"] = ratio_of_ratios(a, b, outcome)
    table = pd.DataFrame(rows).to_dict(orient="records")
    for r in table:  # the frame round trip floats every obs once a share column exists and turns None into NaN
        if r["outcome"] not in ("deploying_addresses_share", "deployments_per_deployer") and isinstance(r.get("obs"), float) and r["obs"] == int(r["obs"]):
            r["obs"] = int(r["obs"])
        if isinstance(r.get("suppressed_reason"), float):
            r["suppressed_reason"] = None
    episodes = {d: {"all": int(w[w["offset_days"] == 0]["event_id"].nunique()),
                    "isolated": isolated[d][1]["episodes_after"],
                    "isolated_dropped": isolated[d][1]["dropped"],
                    "isolated_dropped_episodes": isolated[d][1]["dropped_episodes"],
                    "sa": int(len(sa_eids[d])),
                    "bounded": int(bounded_screen(w, censor).loc[lambda f: f["offset_days"] == 0, "event_id"].nunique()),
                    "by_period": w[w["offset_days"] == 0].drop_duplicates("event_id")["period"].value_counts().to_dict(),
                    "by_provider": w[w["offset_days"] == 0].drop_duplicates("event_id")["provider"].value_counts().to_dict()}
                for d, w in windows.items()}
    return {
        "hypotheses": HYPOTHESES,
        "specification_2": specification_2,
        "design": {"windows": list(WINDOWS), "window_renames": WINDOW_RENAMES, "screens": list(SCREENS), "outcomes": list(OUTCOMES),
                   "h2_outcomes": list(H2_OUTCOMES), "periods": [p[0] for p in PERIODS], "bootstrap_draws": S.BOOTSTRAP_DRAWS,
                   "bounded_censor_intervals_utc": [[pd.Timestamp(a, unit="s", tz="UTC").isoformat(), pd.Timestamp(b, unit="s", tz="UTC").isoformat()] for a, b in censor],
                   "bootstrap_seed": S.BOOTSTRAP_SEED, "min_clean_controls": S.BOOTSTRAP_MIN_CLEAN_CONTROLS,
                   "inference_rule": {"min_episodes": MIN_EPISODES, "min_clusters": MIN_CLUSTERS, "min_valid_draw_fraction": MIN_VALID_FRACTION},
                   "isolated_screen": {"cloud_records": len(cloud), "registry_records": int(sum(len(v) for v in registry.values())),
                                       "rule": "drop every episode whose treated during window overlaps a cloud record or an other-provider registry incident"},
                   "bytecode_span_s": bytecode_span,
                   "creation_table": str(creations or S.CREATIONS_ARCHIVE),
                   "populations": {name: int(frame["from"].nunique()) for name, frame in series.items()},
                   "creations": {name: int(len(frame)) for name, frame in series.items()},
                   "deployers_in_span": {name: int(frame["from"].nunique()) for name, frame in series.items()},
                   "share_denominator": "population addresses with any deployment in the span of the creation table"},
        "episodes": episodes,
        "results": results,
        "table": table,
    }


def _fmt(v) -> str:
    return "n.a." if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def _text(v) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def markdown_table(summary: dict) -> str:
    rows = [r for r in summary["table"] if r["window"] in ("during", "recovery_equal")
            and r["split"] in ("all", "2025H1", "2025H2", "2026H1", "2026H2", "2025", "2026")]
    lines = ["| Population | Screen/calendar | Window | Outcome | Split | n | Obs / ref | RR | 95% CI | Clusters | Suppressed |",
             "|:--|:--|:--|:--|:--|--:|--:|--:|:--|--:|:--|"]
    for r in rows:
        lines.append(f"| {r['population']} | {r['screen']}_{r['calendar']} | {r['window']} | {r['outcome']} | {r['split']} "
                     f"| {r['n']} | {r['obs']} / {r['ref']} | {_fmt(r.get('rr'))} | {_fmt(r.get('ci'))} | {r.get('clusters', '')} | {_text(r.get('suppressed_reason'))} |")
    lines += ["", "| Population | Screen/calendar | Window | Outcome | RR 2025 | RR 2026 | 2026/2025 | 95% CI | Suppressed |",
              "|:--|:--|:--|:--|--:|--:|--:|:--|:--|"]
    for key, v in summary["results"].items():
        if key.endswith("|2026_over_2025"):
            pop, sc, window, outcome, _ = key.split("|")
            lines.append(f"| {pop} | {sc} | {window} | {outcome} | {_fmt(v.get('rr_a'))} | {_fmt(v.get('rr_b'))} | {_fmt(v.get('ratio'))} | {_fmt(v.get('ci'))} | {_text(v.get('suppressed_reason'))} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--creations", type=Path, default=None,
                        help="creation table in the archive schema (block_number, ts, hash, from, status); default the operator archive")
    parser.add_argument("--populations", type=Path, default=None,
                        help="JSON object: population name -> file with a JSON list of addresses; default verified + archive")
    parser.add_argument("--out-dir", type=Path, default=None, help=f"default {OUT_DIR}")
    parser.add_argument("--verified-addresses", type=Path, default=None, help="reuse a previously written address list instead of reading the ledgers")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    summary = build(root, args.creations, args.populations, args.verified_addresses)
    out = root / (args.out_dir or OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "table.md").write_text(markdown_table(summary))
    inputs = [S.WINDOW_COUNTS, args.creations or S.CREATIONS_ARCHIVE, S.BYTECODE_TABLE, S.SOURCE_SCREEN, CLOUD_TABLE, REGISTRY]
    if args.populations:
        inputs.append(args.populations)
        inputs.extend(Path(p) for p in json.loads(args.populations.read_text()).values())
    if args.verified_addresses:
        inputs.append(args.verified_addresses)
    receipt = {"script": "analysis/outage_deployment_heterogeneity.py", "script_sha256": sha(Path(__file__).resolve()),
               "inputs": {str(p): sha(root / p) for p in inputs},
               "summary_sha256": sha(out / "summary.json"), "hypotheses": HYPOTHESES, "specification_2": specification_2,
               "screens": list(SCREENS), "windows": list(WINDOWS), "outcomes": list(OUTCOMES)}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out / 'summary.json'} sha256={receipt['summary_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
