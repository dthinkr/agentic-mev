#!/usr/bin/env python3
"""Classify searcher executor contracts and build hourly outcome panels (streaming memory-safe version).

Inputs (from 01): data/<chain>/swaptx_*.parquet, data/<chain>/swaplog_*.parquet
Inputs (from 02b): data/<chain>/creations_targeted_*.parquet   (trace-level CREATE/CREATE2 filtered to searcher addresses)

Outputs: data/<chain>/executors.parquet, operators.parquet, sandwiches.parquet, hourly_panel.parquet,
         searcher_deployments.parquet, results/<chain>_classify_summary.json
"""
import argparse, gc, glob, json, os, time
from pathlib import Path
import polars as pl

D = Path(__file__).resolve().parents[1] / "data"; R = Path(__file__).resolve().parents[1] / "results"
MIN_TX = 50; MAX_CALLERS = 3; SAND_SHARE = 0.2


def executors_base(tx_files, cache_path=None):
    if cache_path and cache_path.exists():
        return pl.read_parquet(cache_path)
    parts = []
    for i, f in enumerate(tx_files, 1):
        p = (
            pl.scan_parquet(f)
            .filter(pl.col("to").is_not_null())
            .group_by("to")
            .agg(
                pl.len().alias("n_tx"),
                (pl.col("status") == 0).sum().alias("n_fail"),
                pl.col("ts").min().alias("first_ts"),
                pl.col("ts").max().alias("last_ts"),
                pl.col("from").first().alias("main_caller"),
            )
            .collect(engine="streaming")
            .rechunk()
        )
        parts.append(p)
        if len(parts) >= 12 or i == len(tx_files):
            parts = [
                pl.concat(parts)
                .group_by("to")
                .agg(
                    pl.col("n_tx").sum(),
                    pl.col("n_fail").sum(),
                    pl.col("first_ts").min(),
                    pl.col("last_ts").max(),
                    pl.col("main_caller").first(),
                )
                .rechunk()
            ]
    cand = parts[0].filter(pl.col("n_tx") >= MIN_TX)
    cand_set = cand["to"].implode()
    c_parts = []
    for i, f in enumerate(tx_files, 1):
        c = (
            pl.scan_parquet(f)
            .select("to", "from")
            .filter(pl.col("to").is_in(cand_set))
            .unique()
            .group_by("to")
            .head(MAX_CALLERS + 1)
            .collect(engine="streaming")
            .rechunk()
        )
        c_parts.append(c)
        if len(c_parts) >= 12 or i == len(tx_files):
            c_parts = [pl.concat(c_parts).unique().group_by("to").head(MAX_CALLERS + 1).rechunk()]
    callers_cnt = c_parts[0].group_by("to").agg(pl.len().alias("n_callers")).filter(pl.col("n_callers") <= MAX_CALLERS)
    res = (
        cand.join(callers_cnt, on="to", how="inner")
        .with_columns((pl.col("n_fail") / pl.col("n_tx")).alias("revert_rate"))
        .drop("n_fail")
        .rename({"to": "executor"})
        .rechunk()
    )
    if cache_path:
        res.write_parquet(cache_path)
    return res


def exact_median_from_hist(hist_parts, val_col, out_col):
    h = (
        pl.concat(hist_parts)
        .group_by("to", val_col)
        .agg(pl.col("len").sum())
        .sort(["to", val_col])
        .with_columns(
            pl.col("len").cum_sum().over("to").alias("cum"),
            pl.col("len").sum().over("to").alias("tot"),
        )
    )
    lo = h.filter(pl.col("cum") >= ((pl.col("tot") + 1) // 2)).group_by("to").agg(pl.col(val_col).first().alias("m1"))
    hi = h.filter(pl.col("cum") >= (pl.col("tot") // 2 + 1)).group_by("to").agg(pl.col(val_col).first().alias("m2"))
    return lo.join(hi, on="to").select(pl.col("to").alias("executor"), ((pl.col("m1") + pl.col("m2")) / 2.0).alias(out_col))


def sandwiches_month(lg_file, exec_set):
    s = pl.scan_parquet(lg_file).select("block_number", "transaction_index", "transaction_hash", "pool", "from", "to").collect()
    if s.height == 0:
        return s.select(
            pl.col("block_number"),
            pl.col("pool"),
            pl.col("transaction_hash").alias("victim_tx"),
            pl.col("transaction_hash").alias("front_tx"),
            pl.col("transaction_hash").alias("back_tx"),
            pl.col("to").alias("attacker"),
            pl.col("from").alias("victim_from"),
        )
    cnt = s.group_by(["block_number", "pool"]).len().filter(pl.col("len") >= 3).select("block_number", "pool")
    s = s.join(cnt, on=["block_number", "pool"], how="inner").unique(["transaction_hash", "pool"]).sort(["block_number", "pool", "transaction_index"])
    del cnt
    s = s.with_columns(
        pl.col("to").shift(1).over(["block_number", "pool"]).alias("to_prev"),
        pl.col("to").shift(-1).over(["block_number", "pool"]).alias("to_next"),
        pl.col("from").shift(1).over(["block_number", "pool"]).alias("from_prev"),
        pl.col("from").shift(-1).over(["block_number", "pool"]).alias("from_next"),
        pl.col("transaction_hash").shift(1).over(["block_number", "pool"]).alias("tx_prev"),
        pl.col("transaction_hash").shift(-1).over(["block_number", "pool"]).alias("tx_next"),
    )
    same_exec = ((pl.col("to_prev") == pl.col("to_next")) & pl.col("to_prev").is_in(exec_set)) | (
        (pl.col("from_prev") == pl.col("from_next")) & pl.col("from_prev").is_not_null()
    )
    v = s.filter(same_exec & (pl.col("from") != pl.col("from_prev")) & (pl.col("from") != pl.col("from_next")))
    res = v.select(
        "block_number",
        "pool",
        pl.col("transaction_hash").alias("victim_tx"),
        pl.col("tx_prev").alias("front_tx"),
        pl.col("tx_next").alias("back_tx"),
        pl.coalesce([pl.col("to_prev"), pl.col("from_prev")]).alias("attacker"),
        pl.col("from").alias("victim_from"),
    ).rechunk()
    del s, v
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("chain"); ap.add_argument("--reuse", action="store_true"); a = ap.parse_args(); t0 = time.time(); ch = a.chain
    tx_files = sorted(glob.glob(str(D / ch / "swaptx_*.parquet"))); lg_files = sorted(glob.glob(str(D / ch / "swaplog_*.parquet")))
    ex = executors_base(tx_files, D / ch / "_executors_base_cache.parquet"); exec_set = ex["executor"].implode()
    print(f"{ch}: executors {ex.height} {time.time()-t0:.0f}s", flush=True)
    swf = D / ch / "sandwiches.parquet"
    sw_dir = D / ch / "_sw_parts"
    sw_dir.mkdir(exist_ok=True)
    if swf.exists() and a.reuse:
        sw = pl.read_parquet(swf); print(f"reusing {swf.name}: {sw.height:,} victims", flush=True)
    else:
        old_sw = Path("/content/base22/old/sandwiches.parquet")
        if ch == "base22" and old_sw.exists():
            p0 = sw_dir / "part_0000_old6m.parquet"
            if not p0.exists():
                os.symlink(str(old_sw), str(p0))
            lg_run = [f for f in lg_files if not Path(f).name.startswith("swaplog_20260")]
            print(f"  reused 6m old sandwiches; scanning {len(lg_run)} SQD swaplog files", flush=True)
        else:
            lg_run = lg_files
        for idx, f in enumerate(lg_run, 1):
            p_out = sw_dir / f"part_{idx:04d}.parquet"
            if not p_out.exists():
                df_sw = sandwiches_month(f, exec_set)
                df_sw.write_parquet(p_out)
                del df_sw
                gc.collect()
            if idx % 20 == 0 or idx == len(lg_run):
                print(f"  sandwiches {idx}/{len(lg_run)} files {time.time()-t0:.0f}s", flush=True)
        part_files = sorted(glob.glob(str(sw_dir / "part_*.parquet")))
        pl.scan_parquet(part_files).sink_parquet(swf)
        sw = pl.read_parquet(swf)
        print(f"sandwich victims {sw.height:,} {time.time()-t0:.0f}s", flush=True)
    sw_height = sw.height
    sw_attackers = int(sw["attacker"].n_unique())
    bread = pl.concat([sw.select(pl.col("front_tx").alias("hash")), sw.select(pl.col("back_tx").alias("hash"))]).unique().with_columns(pl.lit(True).alias("is_bread")).rechunk()
    del sw
    gc.collect()
    bs_parts, callers_parts, pe_parts, pa_parts, pool_hist_parts, swap_hist_parts, n_tx = [], [], [], [], [], [], 0
    for idx, f in enumerate(tx_files, 1):
        pa_df = (
            pl.scan_parquet(f)
            .select((pl.col("ts") // 3600 * 3600).alias("hour"), "to")
            .group_by("hour")
            .agg(pl.len().alias("swap_tx_all"), pl.col("to").n_unique().alias("active_to"))
            .collect(engine="streaming")
            .rechunk()
        )
        n_tx += int(pa_df["swap_tx_all"].sum()) if pa_df.height else 0
        pa_parts.append(pa_df)
        te = (
            pl.scan_parquet(f)
            .select("hash", "from", "to", "status", "n_swap", "n_pool", "ts")
            .filter(pl.col("to").is_in(exec_set))
            .collect(engine="streaming")
        )
        if te.height:
            te = te.join(bread, on="hash", how="left").with_columns(
                pl.col("is_bread").fill_null(False),
                (pl.col("ts") // 3600 * 3600).alias("hour"),
            )
            bs_parts.append(te.group_by("to").agg(pl.col("is_bread").sum().alias("n_bread"), pl.len().alias("n")).rechunk())
            callers_parts.append(te.select(pl.col("to").alias("executor"), pl.col("from").alias("caller")).unique().rechunk())
            pe_parts.append(te.group_by("hour", "to").agg(pl.len().alias("exec_tx"), (pl.col("status") == 0).sum().alias("exec_fail"), pl.col("is_bread").sum().alias("bread_tx")).rechunk())
            pool_hist_parts.append(te.group_by("to", "n_pool").len().rechunk())
            swap_hist_parts.append(te.group_by("to", "n_swap").len().rechunk())
        del te
        gc.collect()
        if idx % 10 == 0 or idx == len(tx_files):
            print(f"  swaptx pass {idx}/{len(tx_files)} files {time.time()-t0:.0f}s", flush=True)
    med_pool_df = exact_median_from_hist(pool_hist_parts, "n_pool", "med_pool")
    med_swap_df = exact_median_from_hist(swap_hist_parts, "n_swap", "med_swap")
    ex = ex.join(med_swap_df, on="executor", how="left").join(med_pool_df, on="executor", how="left")
    bs = pl.concat(bs_parts).group_by("to").agg(pl.col("n_bread").sum(), pl.col("n").sum()).with_columns((pl.col("n_bread") / pl.col("n")).alias("bread_share")).rename({"to": "executor"}).drop("n")
    ex = ex.join(bs, on="executor", how="left").with_columns(pl.col("bread_share").fill_null(0.0), pl.col("n_bread").fill_null(0)).with_columns(
        pl.when(pl.col("bread_share") >= SAND_SHARE).then(pl.lit("sandwich")).when(pl.col("med_pool") >= 2).then(pl.lit("arb")).otherwise(pl.lit("other")).alias("etype"))
    types = dict(ex.group_by("etype").len().iter_rows()); print("executor types:", types, flush=True)
    callers = pl.concat(callers_parts).unique(); ops = callers.join(ex.select("executor", "etype", "first_ts"), on="executor"); ops.write_parquet(D / ch / "operators.parquet")
    pe = pl.concat(pe_parts).group_by("hour", "to").agg(pl.col("exec_tx").sum(), pl.col("exec_fail").sum(), pl.col("bread_tx").sum()).join(ex.select("executor", "etype"), left_on="to", right_on="executor").group_by("hour", "etype").agg(pl.col("exec_tx").sum(), pl.col("exec_fail").sum(), pl.col("bread_tx").sum())
    pa = pl.concat(pa_parts).group_by("hour").agg(pl.col("swap_tx_all").sum(), pl.col("active_to").max())
    out = {"chain": ch, "swap_tx": n_tx, "executors": ex.height, "executor_types": types, "sandwich_victims": sw_height, "sandwich_attackers": sw_attackers,
           "operator_addresses": int(ops["caller"].n_unique()), "window_start": int(ex["first_ts"].min()), "window_end": int(ex["last_ts"].max())}
    cr_files = sorted(glob.glob(str(D / ch / "creations_targeted_*.parquet"))); panel_dep = None
    if cr_files:
        cr = pl.concat([pl.read_parquet(f) for f in cr_files], rechunk=True).unique(["tx_hash", "address"])
        hx = lambda c: pl.col(c).bin.encode("hex").str.to_lowercase()
        exh = ex.with_columns(("0x" + hx("executor")).alias("addr_hex")).select("addr_hex", pl.col("etype").alias("new_etype"))
        opsh = ops.with_columns(("0x" + hx("caller")).alias("caller_hex"), ("0x" + hx("executor")).alias("exec_hex"))
        optype = opsh.group_by("caller_hex").agg(pl.col("etype").mode().first().alias("op_etype"))
        cr = cr.with_columns(pl.col("address").str.to_lowercase(), pl.col("tx_from").str.to_lowercase(), pl.col("trace_from").str.to_lowercase(), (pl.col("ts") // 3600 * 3600).alias("hour"))
        cr = cr.join(exh, left_on="address", right_on="addr_hex", how="left").join(optype, left_on="tx_from", right_on="caller_hex", how="left")
        cr = cr.with_columns(pl.col("new_etype").is_not_null().alias("becomes_executor"),
                             (pl.col("tx_from").is_in(opsh["caller_hex"].implode()) | pl.col("trace_from").is_in(opsh["exec_hex"].implode()) | pl.col("tx_from").is_in(opsh["exec_hex"].implode())).alias("by_operator"),
                             pl.coalesce([pl.col("new_etype"), pl.col("op_etype")]).alias("dep_etype"))
        cr = cr.filter(pl.col("status") == 1)
        panel_dep = cr.group_by("hour").agg(pl.len().alias("create_all"), pl.col("becomes_executor").sum().alias("create_executor"), (pl.col("by_operator") & ~pl.col("becomes_executor")).sum().alias("create_by_operator"),
                                           ((pl.col("becomes_executor") | pl.col("by_operator")) & (pl.col("dep_etype") == "sandwich")).sum().alias("dep_sandwich"),
                                           ((pl.col("becomes_executor") | pl.col("by_operator")) & (pl.col("dep_etype") == "arb")).sum().alias("dep_arb"), (pl.col("trace_type") == "top").sum().alias("create_top"))
        out.update({"creations": cr.height, "creations_becoming_executor": int(cr["becomes_executor"].sum()), "creations_by_operator": int(cr["by_operator"].sum()),
                    "executors_with_observed_creation": int(cr.filter(pl.col("becomes_executor"))["address"].n_unique()), "deploying_operators": int(cr.filter(pl.col("by_operator"))["tx_from"].n_unique())})
        cr.filter(pl.col("becomes_executor") | pl.col("by_operator")).write_parquet(D / ch / "searcher_deployments.parquet")
    tl_files = sorted(glob.glob(str(D / ch / "creations_toplevel_*.parquet"))); panel_top = None
    if tl_files:
        optype_b = ops.group_by("caller").agg(pl.col("etype").mode().first().alias("op_etype"))
        tl_parts = []
        for f in tl_files:
            tl_parts.append(
                pl.scan_parquet(f)
                .select("hash", "from", "ts", "status")
                .filter(pl.col("status") == 1)
                .join(optype_b.lazy(), left_on="from", right_on="caller", how="inner")
                .collect(engine="streaming")
                .rechunk()
            )
        tl = pl.concat(tl_parts, rechunk=True).unique("hash").with_columns((pl.col("ts") // 3600 * 3600).alias("hour"))
        panel_top = tl.group_by("hour").agg(pl.len().alias("top_by_operator"), (pl.col("op_etype") == "sandwich").sum().alias("top_sandwich"), (pl.col("op_etype") == "arb").sum().alias("top_arb"),
                                            pl.col("from").n_unique().alias("top_ops"), pl.col("from").filter(pl.col("op_etype") == "arb").n_unique().alias("top_ops_arb"), pl.col("from").filter(pl.col("op_etype") == "sandwich").n_unique().alias("top_ops_sandwich"))
        tl.select("hash", "from", "ts", "op_etype").write_parquet(D / ch / "toplevel_operator_deployments.parquet")
        out.update({"toplevel_creations_by_operator": tl.height, "toplevel_deploying_operators": int(tl["from"].n_unique())})
    h0, h1 = int(pa["hour"].min()), int(pa["hour"].max()); hours = pl.DataFrame({"hour": range(h0, h1 + 3600, 3600)})
    pep = pe.pivot(on="etype", index="hour", values=["exec_tx", "exec_fail", "bread_tx"]).fill_null(0)
    panel = hours.join(pa, on="hour", how="left").join(pep, on="hour", how="left")
    if panel_dep is not None: panel = panel.join(panel_dep, on="hour", how="left")
    if panel_top is not None: panel = panel.join(panel_top, on="hour", how="left")
    panel = panel.fill_null(0).sort("hour"); panel.write_parquet(D / ch / "hourly_panel.parquet")
    ex.write_parquet(D / ch / "executors.parquet")
    (R / f"{ch}_classify_summary.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2)); print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
