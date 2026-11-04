"""Daily dose-response panel: outage hours per day against deployment, with lags, leads and a period interaction.

Pre-registered on 2026-10-27 before any estimate was seen. Motivation: AI provider incidents occur
roughly every two days, so multi-day post-outage windows are almost never free of another incident
and the window design cannot see persistence. A daily panel uses every day and every incident
instead: the exposure of day d is the number of hours in which a developer-relevant incident of at
least 60 minutes was open (union over overlapping incidents, per provider), and the outcome is that
day's deployment count.

H1  Same-day outage hours reduce deployment (coefficient on hours below zero).
H2  The reduction is larger in 2026 than in 2025 (interaction of hours with the 2026 indicator below zero).
H3  The reduction persists into the following days (sum of the lag-1 to lag-3 coefficients below zero);
    leads of one and two days are placebo terms and should be near zero.

Model: Poisson pseudo-maximum-likelihood with day-of-week and calendar-month fixed effects, cloud
blocker hours as a control, HC1 and Monday-week-cluster standard errors. Effects are reported as the
multiplicative change for eight outage hours, exp(8 * beta). Outcomes: creations, distinct creating
addresses, creations capped at three per address and day. Populations: verified MEV addresses, the
whole creation archive. Outputs results/outage_daily_dose_panel/{summary.json,table.md,receipt.json}
and results/outage_daily_dose_panel/verified_addresses.json (the address set used, for reuse).

(2026-10-27, after the adversarial design review, before the first corrected estimate).
The primary test is the spec `primary`: contemporaneous hours plus lags 1 to 3, no leads and no cloud
control, fitted for every population and outcome. H1 is adjudicated on the cumulative effect of an
eight-hour disruption on day 0: the sum over days 0 to 3 of the predicted count under the exposure
path (8, 0, 0, 0) for hours and the lags shifted accordingly minus the predicted count under zero
exposure, averaged over the panel days (so the baseline is the panel-average predicted count with the
fitted fixed effects), reported as a count and as a ratio to the baseline four-day predicted total.
Its interval is the delta method over the full coefficient vector with the Monday-week cluster
covariance; a 28-day calendar-block bootstrap (consecutive panel-day blocks resampled with
replacement, 2000 draws, seed 20261017) gives percentile intervals for the hours coefficient and the
cumulative effect beside the cluster ones. H2 is period heterogeneity, not adoption: the primary and
same-day specs are refitted on the common months January to October of 2025 and of 2026
(`period_jan_oct_2025`, `period_jan_oct_2026`, the 2026 period ending with the panel) and the 2026
minus 2025 contrast of the hours coefficient and of the cumulative ratio is reported. The exposure
taxonomy is selected with `--exposure`: `registry` (default, the rule above, which reproduces the
2026-10-27 outputs) or `core`, which keeps a record only when it is developer-relevant, its surface is
`api` or `coding_agent`, its duration is between 60 minutes and 24 hours inclusive and its title does
not describe an access, policy, compliance, analytics, deprecation, rate-limit, billing or
single-tenant (FedRAMP) change. Excluded records and kept hours per provider and year are written under
`exposure`; the administrative records (developer-relevant, at least 60 minutes, title matching the
administrative pattern, any surface) form the separate daily series `admin_hours`, reported in the
descriptive block and never used as a regressor. The cloud control stays in the secondary specs and
its date coverage is written to the summary. Every fit reports `clusters`, the number of Monday weeks
with positive exposure in its sample. The full-year subsample fits are kept only under `--exposure
registry` and are labelled legacy. The receipt records the exposure rule, the spec columns and the
hash of every input read.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bcss", ROOT / "analysis/build_cross_strategy_summary.py")
S = importlib.util.module_from_spec(_spec)
sys.modules["bcss"] = S
_spec.loader.exec_module(S)

OUT_DIR = Path("results/outage_daily_dose_panel")
REGISTRY = Path("data/support/incident_registry.csv")
CLOUD = Path("data/cloud_twin_window.csv")
PANEL_START, PANEL_END = "2025-01-01", "2026-10-13"   # [start, end) in UTC days
MIN_MINUTES = 60
CAP = 3
DOSE_HOURS = 8
LAGS, LEADS = (1, 2, 3), (1, 2)
HYPOTHESES = {
    "H1": "same-day outage hours reduce deployment: beta_hours < 0",
    "H2": "larger reduction in 2026: beta_hours_x_2026 < 0",
    "H3": "persistence: sum of lag 1-3 betas < 0; leads 1-2 are placebos near zero",
    "registered": "2026-10-27, before any estimate from this script was seen",
}
# constants.
EXPOSURE_RULES = ("registry", "core")
CORE_SURFACES = ("api", "coding_agent")
CORE_MAX_MINUTES = 24 * 60
ADMIN_PATTERN = re.compile(
    r"suspend|suspension|access to|fedramp|compliance|analytics|workspace analytics|policy|deprecat|retire"
    r"|rate limit change|billing", re.IGNORECASE)
PRIMARY_COLS = ["hours"] + [f"lag{k}" for k in LAGS]
PERIODS = (("period_jan_oct_2025", "2025-01-01", "2025-11-01"), ("period_jan_oct_2026", "2026-01-01", PANEL_END))
BLOCK_DAYS, BLOCK_DRAWS, BLOCK_SEED = 28, 2000, 20261017
specification_2 = {
    "H1_primary": "cumulative effect of an 8-hour disruption over days 0-3 below zero (spec primary, outcome creating_senders, "
                  "population mev_deployers when supplied)",
    "H2_primary": "period heterogeneity: the 2026 minus 2025 contrast of the hours coefficient and of the cumulative ratio "
                  "on the common months January to October",
    "primary_spec": PRIMARY_COLS,
    "cumulative_method": "average over panel days of the per-day difference in predicted counts between the exposure path "
                         "(8,0,0,0) and zero exposure; delta method over the coefficient vector with the Monday-week cluster "
                         "covariance; 28-day block bootstrap percentile intervals beside it",
    "registered": "2026-10-27, before the first corrected estimate",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def day_index() -> pd.DatetimeIndex:
    return pd.date_range(PANEL_START, PANEL_END, freq="D", tz="UTC", inclusive="left")


def hours_per_day(intervals: list[tuple[pd.Timestamp, pd.Timestamp]], days: pd.DatetimeIndex) -> np.ndarray:
    """Hours of each UTC day covered by the union of the intervals."""
    if not intervals:
        return np.zeros(len(days))
    ivs = sorted((a, b) for a, b in intervals if b > a)
    merged = []
    for a, b in ivs:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = np.zeros(len(days))
    naive = days.tz_localize(None)
    starts = naive.to_numpy(dtype="datetime64[ns]"); ends = (naive + pd.Timedelta(days=1)).to_numpy(dtype="datetime64[ns]")
    for a, b in merged:
        a64, b64 = np.datetime64(a.tz_convert("UTC").tz_localize(None)), np.datetime64(b.tz_convert("UTC").tz_localize(None))
        ov = np.minimum(ends, b64) - np.maximum(starts, a64)
        out += np.maximum(ov / np.timedelta64(1, "h"), 0)
    return out


def classify(reg: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Mark every registry record: `base` (today's rule), `kept` (under the chosen rule), `admin`
    (administrative title) and `reasons` (why it is not kept, in a fixed order)."""
    if rule not in EXPOSURE_RULES:
        raise ValueError(f"unknown exposure rule {rule!r}; choose one of {EXPOSURE_RULES}")
    reg = reg.copy()
    reg["dur"] = reg["dur"].astype(float)
    dev = reg["dev_relevant"].astype(str).str.lower() == "true"
    long_enough = reg["dur"] >= MIN_MINUTES
    admin_title = reg["title"].astype(str).str.contains(ADMIN_PATTERN, regex=True)
    surface_ok = reg["surface"].astype(str).isin(CORE_SURFACES)
    short_enough = reg["dur"] <= CORE_MAX_MINUTES
    checks = [("not_dev_relevant", ~dev), ("under_60_minutes", ~long_enough)]
    if rule == "core":
        checks += [("surface_not_api_or_coding_agent", ~surface_ok), ("over_24_hours", ~short_enough),
                   ("title_administrative", admin_title)]
    reg["base"] = dev & long_enough
    reg["admin"] = dev & long_enough & admin_title
    failing = np.zeros(len(reg), dtype=bool)
    reasons = [[] for _ in range(len(reg))]
    for name, mask in checks:
        failing |= mask.to_numpy()
        for i in np.flatnonzero(mask.to_numpy()):
            reasons[i].append(name)
    reg["kept"] = ~failing
    reg["reasons"] = reasons
    return reg


def exposure(root: Path, days: pd.DatetimeIndex, rule: str = "registry") -> tuple[pd.DataFrame, dict]:
    """Daily exposure frame and the taxonomy report (kept hours, excluded records, administrative series)."""
    reg = classify(pd.read_csv(root / REGISTRY), rule)
    reg["start"] = pd.to_datetime(reg["start"], utc=True, format="ISO8601"); reg["end"] = pd.to_datetime(reg["end"], utc=True, format="ISO8601")
    kept = reg[reg["kept"]]
    frame = pd.DataFrame(index=days)
    for prov in ("anthropic", "openai"):
        sub = kept[kept["provider"] == prov]
        frame[f"hours_{prov}"] = hours_per_day(list(zip(sub["start"], sub["end"])), days)
    frame["hours_any"] = hours_per_day(list(zip(kept["start"], kept["end"])), days)
    cloud = pd.read_csv(root / CLOUD)
    cloud["start"] = pd.to_datetime(cloud["start"], utc=True, format="ISO8601"); cloud["end"] = pd.to_datetime(cloud["end"], utc=True, format="ISO8601")
    frame["cloud_hours"] = hours_per_day(list(zip(cloud["start"], cloud["end"])), days)
    frame["incidents"] = int(len(kept))
    admin = reg[reg["admin"]]
    frame["admin_hours"] = hours_per_day(list(zip(admin["start"], admin["end"])), days)

    def record(r) -> dict:
        return {"provider": r["provider"], "code": str(r["code"]), "title": str(r["title"]), "start": r["start"].isoformat(),
                "hours": round(float(r["dur"]) / 60, 2), "surface": str(r["surface"]), "reasons": list(r["reasons"])}

    years = sorted({int(y) for y in reg["start"].dt.year})
    record_hours = {prov: {str(y): round(float(kept.loc[(kept["provider"] == prov) & (kept["start"].dt.year == y), "dur"].sum()) / 60, 1)
                           for y in years} for prov in ("anthropic", "openai")}
    panel_hours = {prov: {str(y): round(float(frame.loc[frame.index.year == y, f"hours_{prov}"].sum()), 1)
                          for y in sorted(set(days.year))} for prov in ("anthropic", "openai")}
    excluded = reg[~reg["kept"]]
    by_reason = {}
    for name in ("not_dev_relevant", "under_60_minutes", "surface_not_api_or_coding_agent", "over_24_hours", "title_administrative"):
        hit = excluded[excluded["reasons"].map(lambda rs, n=name: n in rs)]
        if len(hit):
            by_reason[name] = {"count": int(len(hit)), "hours": round(float(hit["dur"].sum()) / 60, 1)}
    removed = excluded[excluded["base"]]  # records today's rule would have counted
    meta = {
        "rule": rule,
        "definition": {"registry": "dev_relevant and duration >= 60 minutes",
                       "core": "dev_relevant, surface in {api, coding_agent}, 60 minutes <= duration <= 24 hours, "
                               "title not matching the administrative pattern"}[rule],
        "administrative_pattern": ADMIN_PATTERN.pattern,
        "base_rule": {"records": int(reg["base"].sum()), "not_dev_relevant": int((~(reg["dev_relevant"].astype(str).str.lower() == "true")).sum()),
                      "under_60_minutes": int((reg["dur"] < MIN_MINUTES).sum())},
        "kept": {"records": int(len(kept)), "record_hours": record_hours, "panel_union_hours": panel_hours},
        "excluded": {"records": int(len(excluded)), "hours": round(float(excluded["dur"].sum()) / 60, 1), "by_reason": by_reason,
                     "removed_from_base_rule": {"count": int(len(removed)), "hours": round(float(removed["dur"].sum()) / 60, 1),
                                                "titles": [record(r) for _, r in removed.sort_values(["start", "provider"]).iterrows()]}},
        "admin": {"records": int(len(admin)), "record_hours": round(float(admin["dur"].sum()) / 60, 1),
                  "in_regressor": rule == "registry",
                  "titles": [record(r) for _, r in admin.sort_values(["start", "provider"]).iterrows()]},
        "cloud_control": {"records": int(len(cloud)),
                          "first_record": cloud["start"].min().isoformat() if len(cloud) else None,
                          "last_record": cloud["end"].max().isoformat() if len(cloud) else None,
                          "panel_days_covered": int((frame["cloud_hours"] > 0).sum()),
                          "note": "used in the secondary specs only; the primary spec carries no cloud term"},
    }
    return frame, meta


def outcomes(cr: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    stamps = pd.to_datetime(cr["ts"], unit="s", utc=True).dt.floor("D")
    g = cr.assign(day=stamps).groupby("day")
    frame = pd.DataFrame(index=days)
    frame["creations"] = g.size().reindex(days, fill_value=0).to_numpy()
    frame["creating_senders"] = g["from"].nunique().reindex(days, fill_value=0).to_numpy()
    capped = cr.assign(day=stamps).groupby(["day", "from"]).size().clip(upper=CAP).groupby(level=0).sum()
    frame["capped_creations"] = capped.reindex(days, fill_value=0).to_numpy()
    return frame


def design(panel: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=panel.index)
    x["hours"] = panel["hours_any"]
    for k in LAGS:
        x[f"lag{k}"] = panel["hours_any"].shift(k).fillna(0)
    for k in LEADS:
        x[f"lead{k}"] = panel["hours_any"].shift(-k).fillna(0)
    x["is2026"] = (panel.index.year == 2026).astype(float)
    x["hours_x_2026"] = x["hours"] * x["is2026"]
    x["cloud_hours"] = panel["cloud_hours"]
    x["hours_anthropic"] = panel["hours_anthropic"]; x["hours_openai"] = panel["hours_openai"]
    dow = pd.get_dummies(panel.index.dayofweek, prefix="dow", drop_first=True).astype(float); dow.index = panel.index
    month = pd.get_dummies(panel.index.strftime("%Y-%m"), prefix="m", drop_first=True).astype(float); month.index = panel.index
    return pd.concat([x, dow, month], axis=1)


def cumulative_effect(params, cov, Xm: pd.DataFrame, path_cols: list[str]) -> dict:
    """Change in predicted counts over days 0..len(path_cols)-1 for a DOSE_HOURS disruption on day 0,
    averaged over the rows of Xm (each row supplies its fitted fixed effects), minus the zero-exposure
    prediction, with a delta-method interval from `cov` when it is given."""
    cols = list(Xm.columns); b = np.asarray(params, dtype=float)
    X0 = Xm.to_numpy(dtype=float).copy()
    for c in path_cols:
        X0[:, cols.index(c)] = 0.0
    mu0 = np.exp(X0 @ b); T = len(mu0)
    total, grad = 0.0, np.zeros(len(b))
    for c in path_cols:
        Xk = X0.copy(); Xk[:, cols.index(c)] = DOSE_HOURS
        muk = np.exp(Xk @ b)
        total += float((muk - mu0).mean())
        grad += (Xk.T @ muk - X0.T @ mu0) / T
    base = len(path_cols) * float(mu0.mean()); gbase = len(path_cols) * (X0.T @ mu0) / T
    ratio = total / base
    gratio = (grad * base - total * gbase) / base ** 2
    out = {"dose_hours": DOSE_HOURS, "days": len(path_cols), "count": round(total, 4), "baseline_count": round(base, 4), "ratio": round(ratio, 5)}
    if cov is not None:
        V = np.asarray(cov, dtype=float)
        se = float(np.sqrt(max(grad @ V @ grad, 0.0))); se_r = float(np.sqrt(max(gratio @ V @ gratio, 0.0)))
        z = ratio / se_r if se_r > 0 else np.nan
        out.update({"se_count_week_cluster": round(se, 4), "ci_count_cluster": f"[{total - 1.96 * se:.3f}, {total + 1.96 * se:.3f}]",
                    "se_ratio_week_cluster": round(se_r, 5), "ci_ratio_cluster": f"[{ratio - 1.96 * se_r:.4f}, {ratio + 1.96 * se_r:.4f}]",
                    "p_ratio_cluster": round(float(2 * stats.norm.sf(abs(z))), 4) if np.isfinite(z) else None,
                    "method": specification_2["cumulative_method"]})
    return out


def fit(y: np.ndarray, X: pd.DataFrame, cols: list[str], weeks: np.ndarray, cumulative: bool = False) -> dict:
    fe = [c for c in X.columns if c.startswith("dow_") or c.startswith("m_")]
    Xm = sm.add_constant(X[cols + fe], has_constant="add")
    out = {"n_days": int(len(y)), "outcome_mean": round(float(np.mean(y)), 3),
           "clusters": int(pd.Series(weeks)[X["hours"].to_numpy() > 0].nunique()) if "hours" in X.columns else 0}
    # inference rule, applied to the fit itself: a sample whose outcome never varies, or on
    # which the Poisson likelihood cannot be evaluated, reports its counts only. The deploy-and-trade
    # table has whole periods with no event for some populations.
    if float(np.sum(y)) == 0 or float(np.var(y)) == 0:
        out["suppressed"] = "outcome has no variation in this sample"
        return out
    model = sm.GLM(y, Xm, family=sm.families.Poisson())
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_hc = model.fit(cov_type="HC1")
            res_cl = model.fit(cov_type="cluster", cov_kwds={"groups": weeks})
    except (ValueError, np.linalg.LinAlgError) as err:  # NaN deviance at the first guess, singular design
        out["suppressed"] = f"poisson fit failed: {str(err).splitlines()[0][:120]}"
        return out
    if not np.all(np.isfinite(res_cl.params.to_numpy())):
        out["suppressed"] = "poisson fit returned non-finite coefficients"
        return out
    for c in cols:
        b = float(res_hc.params[c]); se_h = float(res_hc.bse[c]); se_c = float(res_cl.bse[c])
        out[c] = {"beta": round(b, 5), "se_hc1": round(se_h, 5), "se_week_cluster": round(se_c, 5),
                  f"effect_{DOSE_HOURS}h": round(float(np.exp(DOSE_HOURS * b)), 3),
                  f"ci_{DOSE_HOURS}h_cluster": f"[{np.exp(DOSE_HOURS * (b - 1.96 * se_c)):.3f}, {np.exp(DOSE_HOURS * (b + 1.96 * se_c)):.3f}]",
                  "p_cluster": round(float(res_cl.pvalues[c]), 4)}
    if all(f"lag{k}" in cols for k in LAGS):
        idx = [f"lag{k}" for k in LAGS]
        bsum = float(res_cl.params[idx].sum()); var = float(np.ones(len(idx)) @ res_cl.cov_params().loc[idx, idx].to_numpy() @ np.ones(len(idx)))
        out["lag_sum"] = {"beta": round(bsum, 5), "se_week_cluster": round(var ** 0.5, 5),
                          f"effect_{DOSE_HOURS}h": round(float(np.exp(DOSE_HOURS * bsum)), 3),
                          f"ci_{DOSE_HOURS}h_cluster": f"[{np.exp(DOSE_HOURS * (bsum - 1.96 * var ** 0.5)):.3f}, {np.exp(DOSE_HOURS * (bsum + 1.96 * var ** 0.5)):.3f}]"}
    out["converged"] = bool(res_hc.converged)
    if cumulative and all(c in cols for c in PRIMARY_COLS):
        out[f"cumulative_{DOSE_HOURS}h"] = cumulative_effect(res_cl.params.to_numpy(), res_cl.cov_params().to_numpy(), Xm, PRIMARY_COLS)
    return out


def block_bootstrap(y: np.ndarray, X: pd.DataFrame, cols: list[str], draws: int = BLOCK_DRAWS, seed: int = BLOCK_SEED,
                    block_days: int = BLOCK_DAYS) -> dict:
    """Calendar-block bootstrap of the primary spec: consecutive blocks of `block_days` panel days are
    resampled with replacement, the Poisson model is refitted and the hours coefficient and the
    cumulative effect are recomputed on the resampled days. Percentile intervals are returned."""
    fe = [c for c in X.columns if c.startswith("dow_") or c.startswith("m_")]
    Xm = sm.add_constant(X[cols + fe], has_constant="add")
    A = Xm.to_numpy(dtype=float); names = list(Xm.columns)
    if float(np.sum(y)) == 0 or float(np.var(y)) == 0:
        return {"block_days": block_days, "draws": int(draws), "valid_draws": 0, "failed_draws": int(draws), "seed": seed,
                "suppressed": "outcome has no variation in this sample"}
    blocks = np.arange(len(y)) // block_days; nb = int(blocks.max()) + 1
    members = [np.flatnonzero(blocks == j) for j in range(nb)]
    rng = np.random.default_rng(seed)
    betas, counts, ratios, failed = [], [], [], 0
    for _ in range(draws):
        idx = np.concatenate([members[j] for j in rng.integers(0, nb, size=nb)])
        yb, Ab = y[idx], A[idx]
        keep = (Ab != 0).any(axis=0)
        sub = [n for n, k in zip(names, keep) if k]
        if yb.sum() == 0 or any(c not in sub for c in PRIMARY_COLS):
            failed += 1
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.GLM(yb, Ab[:, keep], family=sm.families.Poisson()).fit()
        except Exception:  # singular or non-convergent draw
            failed += 1
            continue
        params = np.asarray(res.params, dtype=float)
        if not np.all(np.isfinite(params)):
            failed += 1
            continue
        ce = cumulative_effect(params, None, pd.DataFrame(Ab[:, keep], columns=sub), PRIMARY_COLS)
        betas.append(params[sub.index("hours")]); counts.append(ce["count"]); ratios.append(ce["ratio"])
    out = {"block_days": block_days, "blocks": nb, "draws": int(draws), "valid_draws": int(len(betas)), "failed_draws": int(failed), "seed": seed,
           "method": "percentile interval over calendar-block resamples of the primary spec"}
    if betas:
        q = lambda v: np.quantile(np.asarray(v, dtype=float), [0.025, 0.975])  # noqa: E731
        lo_b, hi_b = q(betas); lo_c, hi_c = q(counts); lo_r, hi_r = q(ratios)
        out["hours"] = {"ci_beta_block": f"[{lo_b:.5f}, {hi_b:.5f}]",
                        f"ci_{DOSE_HOURS}h_block": f"[{np.exp(DOSE_HOURS * lo_b):.3f}, {np.exp(DOSE_HOURS * hi_b):.3f}]"}
        out[f"cumulative_{DOSE_HOURS}h"] = {"ci_count_block": f"[{lo_c:.3f}, {hi_c:.3f}]", "ci_ratio_block": f"[{lo_r:.4f}, {hi_r:.4f}]",
                                            "lo_count": round(float(lo_c), 4), "hi_count": round(float(hi_c), 4),
                                            "lo_ratio": round(float(lo_r), 5), "hi_ratio": round(float(hi_r), 5)}
    return out


def period_contrast(fit_a: dict, fit_b: dict) -> dict:
    """2026 minus 2025 contrast of the primary hours coefficient and the cumulative ratio, the two
    period fits treated as independent samples (disjoint days, disjoint Monday weeks)."""
    out = {"n_days_2025": fit_a.get("n_days"), "n_days_2026": fit_b.get("n_days"), "clusters_2025": fit_a.get("clusters"), "clusters_2026": fit_b.get("clusters")}
    if "hours" in fit_a and "hours" in fit_b:
        d = fit_b["hours"]["beta"] - fit_a["hours"]["beta"]
        se = (fit_a["hours"]["se_week_cluster"] ** 2 + fit_b["hours"]["se_week_cluster"] ** 2) ** 0.5
        z = d / se if se > 0 else np.nan
        out["hours"] = {"beta": round(d, 5), "se_week_cluster": round(se, 5), f"effect_{DOSE_HOURS}h": round(float(np.exp(DOSE_HOURS * d)), 3),
                        f"ci_{DOSE_HOURS}h_cluster": f"[{np.exp(DOSE_HOURS * (d - 1.96 * se)):.3f}, {np.exp(DOSE_HOURS * (d + 1.96 * se)):.3f}]",
                        "p_cluster": round(float(2 * stats.norm.sf(abs(z))), 4) if np.isfinite(z) else None}
    ca, cb = fit_a.get(f"cumulative_{DOSE_HOURS}h"), fit_b.get(f"cumulative_{DOSE_HOURS}h")
    if ca and cb and "se_ratio_week_cluster" in ca and "se_ratio_week_cluster" in cb:
        d = cb["ratio"] - ca["ratio"]
        se = (ca["se_ratio_week_cluster"] ** 2 + cb["se_ratio_week_cluster"] ** 2) ** 0.5
        z = d / se if se > 0 else np.nan
        out[f"cumulative_{DOSE_HOURS}h_ratio"] = {"difference": round(d, 5), "se_week_cluster": round(se, 5),
                                                  "ci_cluster": f"[{d - 1.96 * se:.4f}, {d + 1.96 * se:.4f}]",
                                                  "p_cluster": round(float(2 * stats.norm.sf(abs(z))), 4) if np.isfinite(z) else None}
    return out


def verified_addresses(root: Path) -> set[bytes]:
    df_sw, _ = S.load_ledger([root / p for p in S.DEFAULT_SWAP_ROOTS])
    df_liq, _ = S.load_ledger([root / p for p in S.DEFAULT_LIQUIDATION_ROOTS], dedupe=True)
    senders = pd.concat([df_sw["sender"], df_liq["sender"]]).str.lower().unique()
    return {bytes.fromhex(h[2:]) for h in senders if len(h) == 42}


def subsample(X: pd.DataFrame, panel: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Design rows of a calendar subsample with the all-zero columns and, when needed, the first
    month dummy dropped. The full-panel month dummies use January 2025 as the reference; inside a
    subsample every day may fall in one of the retained month dummies, so the first month present must
    be dropped or the constant is collinear with them. A rank-deficient design leaves the point
    estimates identified but makes the robust standard errors platform dependent."""
    Xy = X[mask].loc[:, (X[mask] != 0).any(axis=0)]
    months_present = [c for c in Xy.columns if c.startswith("m_")]
    if len(months_present) == panel.index[mask].strftime("%Y-%m").nunique():
        Xy = Xy.drop(columns=months_present[0])
    return Xy


def build(root: Path, verified_file: Path | None, creations: Path | None = None, populations_spec: Path | None = None,
          out_dir: Path | None = None, exposure_rule: str = "registry", block_draws: int = BLOCK_DRAWS) -> tuple[dict, list[str]]:
    days = day_index()
    expo, expo_meta = exposure(root, days, exposure_rule)
    cr_all = pd.read_parquet(root / (creations or S.CREATIONS_ARCHIVE))
    out_dir = root / (out_dir or OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    if populations_spec is not None:
        # name -> file with a JSON list of addresses; each population is those addresses' creations
        populations = {}
        for name, path in json.loads(populations_spec.read_text()).items():
            addrs = {bytes.fromhex(h[2:].lower()) for h in json.loads((root / path).read_text())}
            populations[name] = cr_all[cr_all["from"].isin(addrs)]
        hexes = []
    else:
        if verified_file and verified_file.exists():
            verified = {bytes.fromhex(h[2:]) for h in json.loads(verified_file.read_text())}
        else:
            verified = verified_addresses(root)
        hexes = sorted("0x" + b.hex() for b in verified)
        (out_dir / "verified_addresses.json").write_text(json.dumps(hexes) + "\n")  # written first so a failed fit still leaves the set
        populations = {"verified": cr_all[cr_all["from"].isin(verified)], "archive": cr_all}
    weeks = (days - pd.to_timedelta(days.dayofweek, unit="D")).strftime("%Y-%m-%d").to_numpy()
    specs = {
        "same_day": ["hours", "cloud_hours"],
        "lags_leads": ["hours"] + [f"lag{k}" for k in LAGS] + [f"lead{k}" for k in LEADS] + ["cloud_hours"],
        "period_interaction": ["hours", "hours_x_2026", "cloud_hours"],
        "by_provider": ["hours_anthropic", "hours_openai", "cloud_hours"],
        "primary": list(PRIMARY_COLS),
    }
    outcome_names = ("creations", "creating_senders", "capped_creations")
    results, legacy_fits = {}, []
    for pop, cr in populations.items():
        panel = pd.concat([outcomes(cr, days), expo], axis=1)
        X = design(panel)
        for outcome in outcome_names:
            y = panel[outcome].to_numpy(dtype=float)
            for spec, cols in specs.items():
                results[f"{pop}|{outcome}|{spec}"] = fit(y, X, cols, weeks, cumulative=(spec == "primary"))
            results[f"{pop}|{outcome}|primary"]["block_bootstrap"] = block_bootstrap(y, X, specs["primary"], draws=block_draws)
        if exposure_rule == "registry":
            for year in ("2025", "2026"):
                mask = panel.index.year == int(year)
                Xy = subsample(X, panel, mask)
                year_cols = [c for c in ("hours", "cloud_hours") if c in Xy.columns]  # no cloud blockers fall in 2025
                for outcome in outcome_names:
                    key = f"{pop}|{outcome}|same_day|{year}"
                    results[key] = fit(panel.loc[mask, outcome].to_numpy(dtype=float), Xy, year_cols, weeks[mask])
                    results[key]["legacy"] = "full-year subsample kept for continuity with the 2026-10-27 outputs; H2 uses period_jan_oct_*"
                    legacy_fits.append(key)
        period_fits = {}
        for label, lo, hi in PERIODS:
            mask = np.asarray((panel.index >= pd.Timestamp(lo, tz="UTC")) & (panel.index < pd.Timestamp(hi, tz="UTC")))
            Xy = subsample(X, panel, mask)
            same_day_cols = [c for c in ("hours", "cloud_hours") if c in Xy.columns]
            primary_cols = [c for c in PRIMARY_COLS if c in Xy.columns]
            for outcome in outcome_names:
                y = panel.loc[mask, outcome].to_numpy(dtype=float)
                results[f"{pop}|{outcome}|same_day|{label}"] = fit(y, Xy, same_day_cols, weeks[mask])
                period_fits[(outcome, label)] = results[f"{pop}|{outcome}|primary|{label}"] = fit(y, Xy, primary_cols, weeks[mask], cumulative=True)
        for outcome in outcome_names:
            results[f"{pop}|{outcome}|primary|period_contrast_2026_minus_2025"] = period_contrast(
                period_fits[(outcome, PERIODS[0][0])], period_fits[(outcome, PERIODS[1][0])])
    descriptive = {"days": int(len(days)), "incidents_used": int(expo["incidents"].iloc[0]),
                   "days_with_outage_hours": int((expo["hours_any"] > 0).sum()),
                   "mean_outage_hours_per_day": round(float(expo["hours_any"].mean()), 3),
                   "outage_hours_2025": round(float(expo.loc[expo.index.year == 2025, "hours_any"].sum()), 1),
                   "outage_hours_2026": round(float(expo.loc[expo.index.year == 2026, "hours_any"].sum()), 1),
                   "verified_addresses": len(hexes), "archive_addresses": int(cr_all["from"].nunique()),
                   "populations": {name: {"addresses": int(cr["from"].nunique()), "creations": int(len(cr))}
                                   for name, cr in populations.items()},
                   "admin_hours": {**expo_meta["admin"],
                                   "days_with_admin_hours": int((expo["admin_hours"] > 0).sum()),
                                   "panel_hours_2025": round(float(expo.loc[expo.index.year == 2025, "admin_hours"].sum()), 1),
                                   "panel_hours_2026": round(float(expo.loc[expo.index.year == 2026, "admin_hours"].sum()), 1),
                                   "note": "descriptive series only, never a regressor"}}
    exposure_block = {k: v for k, v in expo_meta.items() if k not in ("admin", "cloud_control")}
    return {"hypotheses": HYPOTHESES, "specification_2": specification_2,
            "design": {"dose_hours": DOSE_HOURS, "cap": CAP, "lags": list(LAGS), "leads": list(LEADS),
                       "fixed_effects": ["day_of_week", "calendar_month"], "panel": [PANEL_START, PANEL_END],
                       "creation_table": str(creations or S.CREATIONS_ARCHIVE),
                       "exposure": exposure_rule, "specs": specs, "primary_spec": "primary",
                       "periods": {label: [lo, hi] for label, lo, hi in PERIODS},
                       "block_bootstrap": {"block_days": BLOCK_DAYS, "draws": int(block_draws), "seed": BLOCK_SEED},
                       "cluster_definition": "Monday weeks with positive contemporaneous exposure in the fit's sample"},
            "exposure": exposure_block, "cloud_control": expo_meta["cloud_control"],
            "legacy_fits": legacy_fits,
            "descriptive": descriptive, "results": results}, hexes


def markdown(summary: dict) -> str:
    skip = {f"cumulative_{DOSE_HOURS}h", "block_bootstrap", f"cumulative_{DOSE_HOURS}h_ratio"}
    lines = ["| Population | Outcome | Spec | Term | exp(8h·β) | 95% CI (week cluster) | p |", "|:--|:--|:--|:--|--:|:--|--:|"]
    for key, v in summary["results"].items():
        parts = key.split("|"); pop, outcome, spec = parts[0], parts[1], "|".join(parts[2:])
        for term, est in v.items():
            if isinstance(est, dict) and term not in skip and "beta" in est:
                lines.append(f"| {pop} | {outcome} | {spec} | {term} | {est.get(f'effect_{DOSE_HOURS}h')} | {est.get(f'ci_{DOSE_HOURS}h_cluster')} | {est.get('p_cluster', '')} |")
    lines += ["", f"Primary specification (hours + lags 1-3, no cloud control; exposure rule `{summary['design'].get('exposure')}`): "
                  f"cumulative effect of an {DOSE_HOURS}-hour disruption over days 0-3.", "",
              "| Population | Outcome | Sample | β hours | exp(8h·β) | 95% CI (cluster) | 95% CI (block) | Cumulative count | Baseline 4-day count | Ratio | 95% CI ratio (delta) | 95% CI ratio (block) | Clusters |",
              "|:--|:--|:--|--:|--:|:--|:--|--:|--:|--:|:--|:--|--:|"]
    ck = f"cumulative_{DOSE_HOURS}h"
    for key, v in summary["results"].items():
        parts = key.split("|")
        if len(parts) < 3 or parts[2] != "primary" or "period_contrast" in key:
            continue
        pop, outcome = parts[0], parts[1]; sample = parts[3] if len(parts) > 3 else "full"
        h, c, bb = v.get("hours", {}), v.get(ck, {}), v.get("block_bootstrap", {})
        lines.append(f"| {pop} | {outcome} | {sample} | {h.get('beta')} | {h.get(f'effect_{DOSE_HOURS}h')} | {h.get(f'ci_{DOSE_HOURS}h_cluster')} "
                     f"| {bb.get('hours', {}).get(f'ci_{DOSE_HOURS}h_block', 'n.a.')} | {c.get('count')} | {c.get('baseline_count')} | {c.get('ratio')} "
                     f"| {c.get('ci_ratio_cluster')} | {bb.get(ck, {}).get('ci_ratio_block', 'n.a.')} | {v.get('clusters')} |")
    lines += ["", "| Population | Outcome | Contrast | Δβ hours (2026 minus 2025) | exp(8h·Δβ) | 95% CI | p | Δ cumulative ratio | 95% CI | p |",
              "|:--|:--|:--|--:|--:|:--|--:|--:|:--|--:|"]
    for key, v in summary["results"].items():
        if "period_contrast" not in key:
            continue
        parts = key.split("|"); h, r = v.get("hours", {}), v.get(f"{ck}_ratio", {})
        lines.append(f"| {parts[0]} | {parts[1]} | {parts[3]} | {h.get('beta')} | {h.get(f'effect_{DOSE_HOURS}h')} | {h.get(f'ci_{DOSE_HOURS}h_cluster')} | {h.get('p_cluster')} "
                     f"| {r.get('difference')} | {r.get('ci_cluster')} | {r.get('p_cluster')} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--verified-addresses", type=Path, default=None, help="reuse a previously written address list")
    parser.add_argument("--creations", type=Path, default=None,
                        help="creation table in the archive schema (block_number, ts, hash, from, status); default the operator archive")
    parser.add_argument("--populations", type=Path, default=None,
                        help="JSON object: population name -> file with a JSON list of addresses; default verified + archive")
    parser.add_argument("--out-dir", type=Path, default=None, help=f"default {OUT_DIR}")
    parser.add_argument("--exposure", choices=EXPOSURE_RULES, default="registry",
                        help="exposure taxonomy: registry (dev_relevant, >= 60 min; the 2026-10-27 rule) or core")
    parser.add_argument("--block-bootstrap-draws", type=int, default=BLOCK_DRAWS, help=f"28-day block bootstrap draws for the primary spec (default {BLOCK_DRAWS})")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    summary, hexes = build(root, args.verified_addresses, args.creations, args.populations, args.out_dir, args.exposure, args.block_bootstrap_draws)
    out = root / (args.out_dir or OUT_DIR); out.mkdir(parents=True, exist_ok=True)
    if hexes:
        (out / "verified_addresses.json").write_text(json.dumps(hexes) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "table.md").write_text(markdown(summary))
    inputs = [args.creations or S.CREATIONS_ARCHIVE, REGISTRY, CLOUD]
    if args.populations:
        inputs.append(args.populations)
        inputs.extend(Path(p) for p in json.loads(args.populations.read_text()).values())
    elif args.verified_addresses and args.verified_addresses.exists():
        inputs.append(args.verified_addresses)
    receipt = {"script": "analysis/outage_daily_dose_panel.py", "script_sha256": sha(Path(__file__).resolve()),
               "inputs": {str(p): sha(root / p) for p in inputs},
               "verified_addresses_sha256": sha(out / "verified_addresses.json") if hexes else None,
               "summary_sha256": sha(out / "summary.json"), "hypotheses": HYPOTHESES,
               "exposure": args.exposure, "spec": {"primary": "primary", "columns": summary["design"]["specs"]},
               "block_bootstrap": summary["design"]["block_bootstrap"], "specification_2": specification_2}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out / 'summary.json'} sha256={receipt['summary_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
