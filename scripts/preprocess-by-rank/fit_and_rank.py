"""
Fit a simple (file_mb, n_faces, n_meshes, n_vertices) -> (enc_time, oom_flag)
model on labelled data, score all 168k, filter by hard thresholds, then
rank remaining candidates by predicted enc_time and emit top-20k CSV.

Output:
    out/ranked_all.parquet   — all 168k with predicted enc_time / oom_prob / flags
    out/top_20k.csv          — sha256, local_path (precompute_feat18.py compatible)
    out/summary.md           — candidate counts, ETA estimate, threshold audit
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


FEATURES = ["file_mb", "n_faces", "n_vertices", "n_meshes",
            "n_primitives", "n_nodes"]


def _make_features(df: pd.DataFrame) -> np.ndarray:
    # log1p stabilises the wide dynamic range (n_faces varies 6 orders).
    X = np.column_stack([
        np.log1p(df["file_mb"].astype(float)),
        np.log1p(df["n_faces"].astype(float)),
        np.log1p(df["n_vertices"].astype(float)),
        np.log1p(df["n_meshes"].astype(float)),
        np.log1p(df["n_primitives"].astype(float)),
        np.log1p(df["n_nodes"].astype(float)),
    ])
    return X


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan", default="scripts/preprocess-by-rank/out/scan_168k.parquet")
    p.add_argument("--labels", default="scripts/preprocess-by-rank/out/labels.parquet")
    p.add_argument("--out_dir", default="scripts/preprocess-by-rank/out")
    p.add_argument("--n_target", type=int, default=20000)
    p.add_argument("--n_gpus", type=int, default=8)
    # hard threshold — conservative, derived from the labelled distribution
    # (file_mb<1: OOM 4%; n_faces<10k: OOM 2.2%; n_meshes<=3: OOM 7%).
    # We use a slightly looser set to keep >>20k candidates, then rely on
    # the learned predictor to rank them.
    p.add_argument("--max_file_mb", type=float, default=15.0)
    p.add_argument("--max_n_faces", type=int, default=300_000)
    p.add_argument("--max_n_meshes", type=int, default=30)
    # soft thresholds on predicted values
    p.add_argument("--max_pred_enc_s", type=float, default=20.0)
    p.add_argument("--max_oom_prob", type=float, default=0.15)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    scan = pd.read_parquet(args.scan)
    labels = pd.read_parquet(args.labels)[["sha256", "status", "enc_time_s"]]

    # Drop rows that failed GLB parse — they're unusable anyway.
    scan = scan[scan["parse_error"] == ""].copy()
    print(f"[fit] parse-ok rows: {len(scan)}", flush=True)

    # Drop the 1 extreme json_too_large outlier (file_mb=964 / n_faces=19M)
    # — these will never run on a single GPU.
    extreme = (scan["file_mb"] > 500) | (scan["n_faces"] > 10_000_000)
    if extreme.any():
        print(f"[fit] dropping {extreme.sum()} extreme-size meshes (file_mb>500 or n_faces>10M)", flush=True)
        scan = scan[~extreme].copy()

    merged = scan.merge(labels, on="sha256", how="left")

    train_mask = merged["status"].isin(["ok", "oom", "fail"])
    train = merged[train_mask].copy()
    print(f"[fit] labelled rows: {len(train)}  "
          f"(ok={(train['status']=='ok').sum()}, "
          f"oom={(train['status']=='oom').sum()}, "
          f"fail={(train['status']=='fail').sum()})", flush=True)

    X_train = _make_features(train)
    # Target 1: log(enc_time) for ok rows; skipped for oom/fail
    ok_mask = train["status"] == "ok"
    reg = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=0,
    )
    reg.fit(X_train[ok_mask], np.log1p(train.loc[ok_mask, "enc_time_s"]))
    # Target 2: oom_flag (oom | fail vs ok)
    y_oom = (train["status"] != "ok").astype(int).values
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=0,
    )
    clf.fit(X_train, y_oom)

    # in-sample sanity
    pred_log = reg.predict(X_train[ok_mask])
    pred_enc = np.expm1(pred_log)
    true_enc = train.loc[ok_mask, "enc_time_s"].values
    from scipy.stats import spearmanr  # type: ignore
    rho, _ = spearmanr(true_enc, pred_enc)
    pred_oom = clf.predict_proba(X_train)[:, 1]
    from sklearn.metrics import roc_auc_score  # type: ignore
    auc = roc_auc_score(y_oom, pred_oom)
    print(f"[fit] in-sample spearman(enc_time, pred) = {rho:.3f}", flush=True)
    print(f"[fit] in-sample OOM-clf AUC = {auc:.3f}", flush=True)

    # Score everything
    X_all = _make_features(merged)
    merged["pred_enc_s"] = np.expm1(reg.predict(X_all))
    merged["pred_oom_prob"] = clf.predict_proba(X_all)[:, 1]

    # Hard threshold pass
    hard = (
        (merged["file_mb"] < args.max_file_mb)
        & (merged["n_faces"] < args.max_n_faces)
        & (merged["n_meshes"] < args.max_n_meshes)
    )
    # Soft threshold on predictions
    soft = (merged["pred_enc_s"] < args.max_pred_enc_s) & (merged["pred_oom_prob"] < args.max_oom_prob)
    # Never pick a sha already known-oom/fail
    not_bad = ~merged["status"].isin(["oom", "fail"])

    cand = merged[hard & soft & not_bad].copy()
    print(f"[fit] hard-filter survivors: {int(hard.sum())}", flush=True)
    print(f"[fit] + soft-filter survivors: {int((hard & soft).sum())}", flush=True)
    print(f"[fit] - known-bad: {int((hard & soft & not_bad).sum())}", flush=True)
    if len(cand) < args.n_target:
        print(f"[WARN] only {len(cand)} candidates; need {args.n_target}. "
              f"Relax thresholds or lower --n_target.", flush=True)

    # Global ranking across all 168k (not just filter survivors).
    # rank_key orders rows from easy-to-hard so that taking the first N
    # from the ranked CSV gives a valid increment for any N up to ~168k.
    #
    # Tiering:
    #   tier 0: already-ok rows (zero extra work)               → sorted by pred_enc asc
    #   tier 1: passes hard+soft filters, unlabelled (fresh)    → sorted by pred_enc asc
    #   tier 2: passes hard filter only (pred_oom high or pred_enc long, but
    #           still small/simple)                             → sorted by pred_enc asc
    #   tier 3: passes neither; labelled-oom / extreme          → sorted by pred_enc asc
    merged["is_known_bad"] = merged["status"].isin(["oom", "fail"])
    tier = np.full(len(merged), 3, dtype=int)
    tier[(merged["status"] == "ok").values] = 0
    tier_fresh_mask = (
        (hard.values) & (soft.values)
        & (~merged["is_known_bad"].values)
        & (merged["status"] != "ok").values
    )
    tier[tier_fresh_mask] = 1
    tier_loose_mask = (
        (hard.values)
        & (~merged["is_known_bad"].values)
        & (tier == 3)  # not already placed in 0/1
    )
    tier[tier_loose_mask] = 2
    merged["tier"] = tier

    merged = merged.sort_values(
        ["tier", "pred_enc_s", "pred_oom_prob", "file_mb"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    merged["rank"] = np.arange(len(merged))
    # selected = first n_target rows of the full rank
    merged["selected"] = (merged["rank"] < args.n_target).astype(int)

    # Emit full-rank CSV (precompute_feat18.py only reads sha256 + local_path,
    # but we carry the predicted columns too for downstream analysis tools).
    out_full_csv = os.path.join(args.out_dir, "full_ranked.csv")
    merged[[
        "sha256", "local_path", "rank", "tier", "status",
        "file_mb", "n_faces", "n_vertices", "n_meshes",
        "pred_enc_s", "pred_oom_prob",
    ]].to_csv(out_full_csv, index=False)
    print(f"[fit] wrote {out_full_csv} ({len(merged)} rows, ranked easy→hard)", flush=True)

    # Slice top-N as the launcher-ready CSV (precompute_feat18.py reads
    # sha256+local_path and ignores extra columns, so we could hand it
    # full_ranked.csv directly — but top_20k.csv keeps the first-batch
    # intent explicit).
    out_csv = os.path.join(args.out_dir, "top_20k.csv")
    top = merged.head(args.n_target).copy()
    top[["sha256", "local_path"]].to_csv(out_csv, index=False)
    print(f"[fit] wrote {out_csv} (first {args.n_target} rows of full_ranked)", flush=True)

    # Also keep the parquet for richer downstream analysis.
    out_parquet = os.path.join(args.out_dir, "ranked_all.parquet")
    merged[["sha256", "local_path", "file_mb", "n_faces",
            "n_vertices", "n_meshes", "n_primitives", "n_nodes",
            "status", "enc_time_s",
            "pred_enc_s", "pred_oom_prob",
            "tier", "rank", "selected"]].to_parquet(out_parquet, index=False)
    print(f"[fit] wrote {out_parquet}", flush=True)

    # Summary — ETA is fresh-only; reused rows just re-read existing npz
    # and do not consume GPU time.
    reused = int((top["status"] == "ok").sum())
    fresh = int(args.n_target - reused)
    fresh_pred_sum = float(top.loc[top["status"] != "ok", "pred_enc_s"].sum())
    total_pred_sum = float(top["pred_enc_s"].sum())
    eta_h = fresh_pred_sum / max(args.n_gpus, 1) / 3600.0
    lines = []
    lines.append(f"# full-rank selection summary (n={len(merged)})\n")
    lines.append("## Outputs")
    lines.append("- `full_ranked.csv` — all 168k sorted easy→hard (tier 0: reused ok; "
                 "tier 1: predicted safe&fast; tier 2: predicted safe but slower; "
                 "tier 3: rest). Feed into `precompute_feat18.py --metadata_csv <abs_path>` "
                 "directly; it only reads `sha256,local_path` columns.")
    lines.append("- `top_20k.csv` — convenience slice (first N rows of full_ranked).")
    lines.append("- `ranked_all.parquet` — same ranking, richer columns for downstream analysis.")
    lines.append("")
    lines.append("## Tier sizes")
    for t, n in merged["tier"].value_counts().sort_index().items():
        lines.append(f"- tier {t}: {n}")
    lines.append("")
    lines.append("## First-batch (top-N) summary")
    lines.append(f"- target size:      {args.n_target}")
    lines.append(f"- reused (already-ok from prior run): {reused}")
    lines.append(f"- fresh encodes needed:               {fresh}")
    lines.append(f"- thresholds: file_mb<{args.max_file_mb}, n_faces<{args.max_n_faces}, n_meshes<{args.max_n_meshes}")
    lines.append(f"  pred_enc<{args.max_pred_enc_s}s, pred_oom<{args.max_oom_prob}")
    lines.append(f"- predicted total enc-seconds (all N): {total_pred_sum:.0f}s")
    lines.append(f"- predicted fresh-only enc-seconds:    {fresh_pred_sum:.0f}s")
    lines.append(f"- at {args.n_gpus} GPUs, fresh-only ETA wall-clock: {eta_h:.2f} h")
    lines.append("")
    lines.append(f"- in-sample spearman(enc_time, pred) = {rho:.3f}")
    lines.append(f"- in-sample OOM-clf AUC = {auc:.3f}")
    lines.append("")
    lines.append(f"## Feature distribution in selected top-{args.n_target}")
    for col in ["file_mb", "n_faces", "n_vertices", "n_meshes", "pred_enc_s", "pred_oom_prob"]:
        q = top[col].quantile([0.5, 0.9, 0.99]).round(2).tolist()
        lines.append(f"  - {col:16s}  p50={q[0]:>10.2f}   p90={q[1]:>10.2f}   p99={q[2]:>10.2f}")
    lines.append("")
    # Cumulative ETA at rank-N milestones. Only fresh (status != 'ok') rows
    # cost wall-time; reused tier-0 rows are near-zero (stat + re-read .npz).
    # The table answers "if I run the first N of full_ranked.csv, what ETA?"
    lines.append(f"## Cumulative ETA per incremental milestone ({args.n_gpus} GPUs)")
    lines.append("| rank N    | fresh rows | reused | pred enc (h) | OOM-prob>0.1 rows |")
    lines.append("|-----------|------------|--------|--------------|-------------------|")
    cum = merged.copy().reset_index(drop=True)
    cum["is_fresh"] = (cum["status"] != "ok").astype(int)
    cum["fresh_enc_s"] = cum["pred_enc_s"] * cum["is_fresh"]
    cum["cum_fresh"] = cum["is_fresh"].cumsum()
    cum["cum_reused"] = (1 - cum["is_fresh"]).cumsum()
    cum["cum_enc_h"] = cum["fresh_enc_s"].cumsum() / max(args.n_gpus, 1) / 3600.0
    cum["cum_oom_risky"] = (
        (cum["pred_oom_prob"] > 0.1) & (cum["is_fresh"] == 1)
    ).cumsum()
    for n in [5_000, 10_000, 20_000, 30_000, 50_000, 80_000, 120_000, 160_000]:
        if n >= len(cum):
            break
        row = cum.iloc[n - 1]
        lines.append(
            f"| {n:<9d} | {int(row['cum_fresh']):>10d} | {int(row['cum_reused']):>6d}"
            f" | {row['cum_enc_h']:>12.2f} | {int(row['cum_oom_risky']):>17d} |"
        )
    lines.append("")
    with open(os.path.join(args.out_dir, "summary.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
