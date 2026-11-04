#!/usr/bin/env python3
"""Hourly within-day outage design for one chain's searcher panel (Base or Ethereum).

Uses exact Frisch-Waugh-Lovell (Hausman-Hall-Griliches / Wooldridge) concentration of UTC-date fixed
effects in Poisson PML (0.15s per fit instead of dense 680x680 dummy inversion) with date-clustered
sandwich standard errors, plus bulk-proof distinct-operator and capped-creation outcomes from
searcher_deployments.parquet when present.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm

warnings.simplefilter("ignore")
ADMIN = re.compile(r"suspend|suspension|access to|fedramp|compliance|analytics|workspace analytics|policy|deprecat|retire|rate limit change|billing", re.I)
MODEL_ERR = re.compile(r"(claude|opus|sonnet|haiku).*(error|latenc|degrad|unavailab)|(error|latenc|degrad).*(claude|opus|sonnet|haiku)", re.I)


def coverage(hours: pd.DatetimeIndex, starts, ends) -> np.ndarray:
    h0 = hours.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]"); h1 = h0 + np.timedelta64(1, "h")
    c = np.zeros(len(hours))
    for a, b in zip(starts, ends):
        a64 = np.datetime64(pd.Timestamp(a).tz_convert("UTC").tz_localize(None)); b64 = np.datetime64(pd.Timestamp(b).tz_convert("UTC").tz_localize(None))
        c += np.clip((np.minimum(h1, b64) - np.maximum(h0, a64)) / np.timedelta64(1, "h"), 0, 1)
    return np.minimum(c, 1)


def exposures_v2(hours: pd.DatetimeIndex, registry_v2: str, providers: list[str] | None) -> tuple[pd.DataFrame, dict]:
    reg = pd.read_csv(registry_v2, low_memory=False)
    reg["start"] = pd.to_datetime(reg["start_utc"], utc=True, errors="coerce"); reg["end"] = pd.to_datetime(reg["end_utc"], utc=True, errors="coerce")
    reg = reg[reg["start"].notna() & reg["end"].notna()]
    reg["dur"] = (reg["end"] - reg["start"]).dt.total_seconds() / 60
    own = reg["own_model"].astype(str).str.lower().isin(["true", "1", "yes"]); dev = reg["dev_relevant"].astype(str).str.lower() == "true"
    span = (reg["end"] > hours[0]) & (reg["start"] < hours[-1] + pd.Timedelta(hours=1)); length = (reg["dur"] >= 60) & (reg["dur"] <= 1440)
    base = reg[own & dev & span & length]
    groups = {"anthropic": ["anthropic"], "openai": ["openai"], "google": ["google-ai"], "xai": ["xai"], "cn": ["moonshot-kimi", "deepseek", "alibaba-qwen", "zhipu-bigmodel", "volcengine-doubao", "baidu-qianfan", "tencent-hunyuan", "minimax", "cn-others"],
              "other_intl": ["mistral", "cohere", "perplexity", "meta-llama"]}
    if providers:
        groups = {g: [p for p in ps if p in providers] for g, ps in groups.items()}
    X = pd.DataFrame(index=hours); counts = {}
    for g, ps in groups.items():
        sub = base[base["provider"].isin(ps)]
        if len(sub):
            X[g] = coverage(hours, sub["start"], sub["end"]); counts[g] = len(sub)
    X["all"] = X.max(axis=1)
    pre = np.zeros(len(hours)); h0 = hours.tz_localize(None).to_numpy(dtype="datetime64[ns]"); h1 = h0 + np.timedelta64(1, "h")
    for a in base["start"]:
        a64 = np.datetime64(a.tz_convert("UTC").tz_localize(None)); pre += np.clip((np.minimum(h1, a64) - np.maximum(h0, a64 - np.timedelta64(6, "h"))) / np.timedelta64(1, "h"), 0, 1)
    X["pre1_6"] = np.minimum(pre, 1) * (X["all"].to_numpy() == 0)
    return X, counts


def exposures(hours: pd.DatetimeIndex, registry: str, copilot: str | None) -> tuple[pd.DataFrame, dict]:
    reg = pd.read_csv(registry); reg["start"] = pd.to_datetime(reg["start"], utc=True, format="ISO8601"); reg["end"] = pd.to_datetime(reg["end"], utc=True, format="ISO8601")
    reg["dur"] = reg["dur"].astype(float); dev = reg["dev_relevant"].astype(str).str.lower() == "true"
    span = (reg["end"] > hours[0]) & (reg["start"] < hours[-1] + pd.Timedelta(hours=1)); length = (reg["dur"] >= 60) & (reg["dur"] <= 1440)
    core = reg[span & dev & length & reg["surface"].isin(["api", "coding_agent"]) & ~reg["title"].astype(str).str.contains(ADMIN)]
    misf = reg[span & dev & length & (reg["provider"] == "anthropic") & (reg["surface"] == "other") & reg["title"].astype(str).str.contains(MODEL_ERR)]
    X = pd.DataFrame({"core": coverage(hours, core["start"], core["end"]), "misfiled": coverage(hours, misf["start"], misf["end"])}, index=hours)
    counts = {"core": len(core), "misfiled": len(misf)}
    if copilot:
        cp = pd.read_csv(copilot); cp["start"] = pd.to_datetime(cp["start"], utc=True); cp["end"] = pd.to_datetime(cp["end"], utc=True)
        cp = cp[(cp["dev_relevant"].astype(str).str.lower() == "true") & (cp["dur"] >= 60) & (cp["dur"] <= 1440) & (cp["end"] > hours[0]) & (cp["start"] < hours[-1])]
        X["copilot"] = coverage(hours, cp["start"], cp["end"]); counts["copilot"] = len(cp)
    X["union"] = X.max(axis=1)
    pre = np.zeros(len(hours)); h0 = hours.tz_localize(None).to_numpy(dtype="datetime64[ns]"); h1 = h0 + np.timedelta64(1, "h")
    for a in core["start"]:
        a64 = np.datetime64(a.tz_convert("UTC").tz_localize(None)); pre += np.clip((np.minimum(h1, a64) - np.maximum(h0, a64 - np.timedelta64(6, "h"))) / np.timedelta64(1, "h"), 0, 1)
    X["pre1_6"] = np.minimum(pre, 1) * (X["union"].to_numpy() == 0)
    return X, counts


def fit_hourly(y: np.ndarray, X: pd.DataFrame, hours: pd.DatetimeIndex) -> dict:
    date_id = pd.factorize(hours.strftime("%Y-%m-%d"))[0]
    day_sum = np.bincount(date_id, weights=y)
    keep = day_sum[date_id] > 0
    yk = y[keep]
    hs = hours[keep]
    did = pd.factorize(date_id[keep])[0]
    n_d = int(did.max()) + 1
    hdum = pd.get_dummies(hs.hour, prefix="h", drop_first=True).to_numpy(dtype=float)
    Xk = X.loc[keep].to_numpy(dtype=float)
    Z = np.column_stack([Xk, hdum])
    p = Z.shape[1]
    theta = np.zeros(p)
    day_y = np.bincount(did, weights=yk)
    for _ in range(40):
        eta_z = np.clip(Z @ theta, -20, 20)
        exp_z = np.exp(eta_z)
        day_exp = np.bincount(did, weights=exp_z)
        mu = exp_z * (day_y[did] / np.maximum(day_exp[did], 1e-300))
        r = yk - mu
        wz = mu[:, None] * Z
        wz_day = np.zeros((n_d, p))
        np.add.at(wz_day, did, wz)
        Z_tilde = Z - wz_day[did] / np.maximum(day_y[did, None], 1e-300)
        H = Z_tilde.T @ (mu[:, None] * Z_tilde)
        g = Z_tilde.T @ r
        step = np.linalg.lstsq(H + np.eye(p) * 1e-9, g, rcond=None)[0]
        theta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    H_inv = np.linalg.pinv(H)
    score_day = np.zeros((n_d, p))
    np.add.at(score_day, did, Z_tilde * r[:, None])
    scale = n_d / max(n_d - 1, 1)
    V = scale * (H_inv @ (score_day.T @ score_day) @ H_inv)
    se = np.sqrt(np.maximum(np.diag(V), 0))
    out = {}
    for i, c in enumerate(X.columns):
        b, s = float(theta[i]), float(se[i])
        pv = float(2 * norm.sf(abs(b / s))) if s > 0 else 1.0
        out[c] = (float(np.exp(b)), float(np.exp(b - 1.96 * s)), float(np.exp(b + 1.96 * s)), pv)
    return out


def fit_daily(y: np.ndarray, core: np.ndarray, hours: pd.DatetimeIndex) -> dict:
    days = pd.date_range(hours[0].floor("D"), hours[-1].floor("D"), freq="D", tz="UTC")
    yd = pd.Series(y, index=hours).groupby(hours.floor("D")).sum().reindex(days, fill_value=0).to_numpy(dtype=float)
    xd = pd.Series(core, index=hours).groupby(hours.floor("D")).sum().reindex(days, fill_value=0)
    D = pd.DataFrame({"x0": xd.to_numpy()}, index=days)
    for k in (1, 2, 3): D[f"lag{k}"] = xd.shift(k).fillna(0).to_numpy()
    for k in (1, 2): D[f"lead{k}"] = xd.shift(-k).fillna(0).to_numpy()
    fe = pd.concat([pd.get_dummies(days.dayofweek, prefix="dow", drop_first=True).astype(float).set_index(days), pd.get_dummies(days.strftime("%Y-%m"), prefix="m", drop_first=True).astype(float).set_index(days)], axis=1)
    Xd = sm.add_constant(pd.concat([D, fe], axis=1), has_constant="add"); wk = pd.factorize(days.to_period("W-SUN").astype(str))[0]
    try:
        r = sm.GLM(yd, Xd, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": wk}, maxiter=200)
    except Exception:
        r = sm.GLM(yd, Xd, family=sm.families.Poisson()).fit(method="lbfgs", maxiter=2000, cov_type="cluster", cov_kwds={"groups": wk})
    return {c: (float(np.exp(8 * r.params[c])), float(np.exp(8 * (r.params[c] - 1.96 * r.bse[c]))), float(np.exp(8 * (r.params[c] + 1.96 * r.bse[c])))) for c in ["lead2", "lead1", "x0", "lag1", "lag2", "lag3"]}


def series_from_ts(ts: np.ndarray, hours: pd.DatetimeIndex) -> np.ndarray:
    t = pd.to_datetime(ts, unit="s", utc=True).floor("h"); t = t[(t >= hours[0]) & (t <= hours[-1])]
    return pd.Series(t).value_counts().reindex(hours, fill_value=0).to_numpy(dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True); ap.add_argument("--panel", required=True); ap.add_argument("--executors", required=True)
    ap.add_argument("--operators", default=None); ap.add_argument("--registry", default=None); ap.add_argument("--copilot", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--registry-v2", default=None, help="rebuilt registry.csv; replaces --registry/--copilot with per-vendor-group exposures")
    ap.add_argument("--providers", default=None, help="comma list of v2 provider slugs to keep")
    a = ap.parse_args()
    panel = pd.read_parquet(a.panel).sort_values("hour"); hours = pd.to_datetime(panel["hour"], unit="s", utc=True); hours = pd.DatetimeIndex(hours)
    if a.registry_v2:
        X, counts = exposures_v2(hours, a.registry_v2, a.providers.split(",") if a.providers else None)
    else:
        X, counts = exposures(hours, a.registry, a.copilot)
    ex = pd.read_parquet(a.executors)
    outcomes = {}
    sd_path = Path(a.panel).parent / "searcher_deployments.parquet"
    if sd_path.exists():
        import polars as pl
        sd = pl.read_parquet(sd_path, columns=["hour", "tx_from", "becomes_executor", "by_operator"])
        h_int = panel["hour"].to_numpy()
        ex_ops = sd.filter(pl.col("becomes_executor")).group_by("hour").agg(pl.col("tx_from").n_unique().alias("v")).to_pandas().set_index("hour")["v"]
        ex_cap = sd.filter(pl.col("becomes_executor")).group_by(["hour", "tx_from"]).len().with_columns(pl.col("len").clip(upper_bound=3)).group_by("hour").agg(pl.col("len").sum().alias("v")).to_pandas().set_index("hour")["v"]
        all_ops = sd.filter(pl.col("by_operator")).group_by("hour").agg(pl.col("tx_from").n_unique().alias("v")).to_pandas().set_index("hour")["v"]
        outcomes["ops deploying executor"] = ex_ops.reindex(h_int, fill_value=0).to_numpy(dtype=float)
        outcomes["executor creations (capped <=3/op/h)"] = ex_cap.reindex(h_int, fill_value=0).to_numpy(dtype=float)
        outcomes["ops deploying any contract"] = all_ops.reindex(h_int, fill_value=0).to_numpy(dtype=float)
    for col, name in [("create_executor", "executor creations"), ("create_by_operator", "operator creations (all)"), ("create_top", "top-level creations (chain)"),
                      ("dep_arb", "arb executor deployments"), ("dep_sandwich", "sandwich executor deployments"), ("top_by_operator", "top-level creations by operators"),
                      ("exec_tx_arb", "arb executions"), ("exec_tx_sandwich", "sandwich executions"), ("exec_tx_other", "other executor txs"), ("swap_tx_all", "all swap txs"), ("active_to", "active destinations")]:
        if col in panel.columns:
            outcomes[name] = panel[col].fillna(0).to_numpy(dtype=float)
    mev = ex[ex["etype"].isin(["arb", "sandwich"])]
    outcomes["go-live: arb+sandwich executors"] = series_from_ts(mev["first_ts"].to_numpy(), hours)
    outcomes["go-live: all executors"] = series_from_ts(ex["first_ts"].to_numpy(), hours)
    if a.operators:
        ops = pd.read_parquet(a.operators); first = ops.groupby("caller")["first_ts"].min()
        outcomes["first appearance: caller keys"] = series_from_ts(first.to_numpy(), hours)
    if a.registry_v2:
        vendor_cols = [c for c in X.columns if c not in ("all", "pre1_6")]
        specs = {"all": X[["all"]], "by_vendor": X[vendor_cols], "all+pre1-6h": X[["all", "pre1_6"]]}
        print(f"\n## {a.label} (registry v2): {len(hours):,} hours {hours[0]:%Y-%m-%d} → {hours[-1]:%Y-%m-%d}; incidents in span by group: {counts}; covered hours: " + ", ".join(f"{c} {X[c].sum():.0f}" for c in vendor_cols + ["all"]), flush=True)
        print("| outcome | total | per hour | all vendors | " + " | ".join(vendor_cols) + " | pre 1–6h | daily lead1 | day0 (8h) | lag2 |", flush=True)
        print("|---|---|---|---|" + "---|" * len(vendor_cols) + "---|---|---|---|", flush=True)
    else:
        specs = {"core": X[["core"]], "core+misfiled+copilot": X[[c for c in ("core", "misfiled", "copilot") if c in X.columns]], "union+pre1-6h": X[["union", "pre1_6"]]}
        print(f"\n## {a.label}: {len(hours):,} hours {hours[0]:%Y-%m-%d} → {hours[-1]:%Y-%m-%d}; incidents in span: {counts}; covered hours core {X['core'].sum():.0f}, misfiled {X['misfiled'].sum():.0f}, copilot {X.get('copilot', pd.Series(dtype=float)).sum():.0f}, union {X['union'].sum():.0f}", flush=True)
        print("| outcome | total | per hour | core | misfiled | copilot | union | pre 1–6h | daily lead1 | day0 (8h) | lag2 |", flush=True)
        print("|---|---|---|---|---|---|---|---|---|---|---|", flush=True)
    results = {}
    fmt = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
    for name, y in outcomes.items():
        if y.sum() == 0:
            continue
        if a.registry_v2:
            r1 = fit_hourly(y, specs["all"], hours); r2 = fit_hourly(y, specs["by_vendor"], hours); r3 = fit_hourly(y, specs["all+pre1-6h"], hours); rd = fit_daily(y, X["all"].to_numpy(), hours)
            results[name] = {"total": float(y.sum()), "all": r1["all"], "by_vendor": r2, "pre1_6": r3["pre1_6"], "daily": rd}
            print(f"| {name} | {int(y.sum()):,} | {y.mean():.2f} | {fmt(r1['all'])} | " + " | ".join(fmt(r2[c]) for c in vendor_cols) + f" | {fmt(r3['pre1_6'])} | {fmt(rd['lead1'])} | {fmt(rd['x0'])} | {fmt(rd['lag2'])} |", flush=True)
            continue
        r1 = fit_hourly(y, specs["core"], hours); r2 = fit_hourly(y, specs["core+misfiled+copilot"], hours); r3 = fit_hourly(y, specs["union+pre1-6h"], hours); rd = fit_daily(y, X["core"].to_numpy(), hours)
        results[name] = {"total": float(y.sum()), "core": r1["core"], "core_joint": r2.get("core"), "misfiled": r2.get("misfiled"), "copilot": r2.get("copilot"), "union": r3["union"], "pre1_6": r3["pre1_6"], "daily": rd}
        print(f"| {name} | {int(y.sum()):,} | {y.mean():.2f} | {fmt(r1['core'])} | {fmt(r2['misfiled']) if 'misfiled' in r2 else ''} | {fmt(r2['copilot']) if 'copilot' in r2 else ''} | {fmt(r3['union'])} | {fmt(r3['pre1_6'])} | {fmt(rd['lead1'])} | {fmt(rd['x0'])} | {fmt(rd['lag2'])} |", flush=True)
    if a.out:
        json.dump({"label": a.label, "hours": len(hours), "incidents": counts, "results": results}, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
