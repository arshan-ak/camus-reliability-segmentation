import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

VIEWS = ("2CH", "4CH")
INSTANTS = ("ED", "ES")

def nanmean(x):
    x = pd.to_numeric(x, errors="coerce")
    if len(x) == 0:
        return float("nan")
    return float(np.nanmean(x.values))

def summarize(df):
    return {"n": len(df), "dice": nanmean(df["dice_mean"]), "la": nanmean(df["dice_la"])}

def load_thresholds(val_entropy_csv: str, target_cov: float, grouped: bool):
    df = pd.read_csv(val_entropy_csv)
    df["unc_entropy"] = pd.to_numeric(df["unc_entropy"], errors="coerce")

    if not grouped:
        df = df.sort_values("unc_entropy").reset_index(drop=True)
        k = max(1, int(round(len(df) * target_cov)))
        thr = float(df.loc[k - 1, "unc_entropy"])
        return {("ALL", "ALL"): thr}

    thrs = {}
    for v in VIEWS:
        for inst in INSTANTS:
            g = df[(df["view"] == v) & (df["instant"] == inst)].sort_values("unc_entropy").reset_index(drop=True)
            k = max(1, int(round(len(g) * target_cov)))
            thrs[(v, inst)] = float(g.loc[k - 1, "unc_entropy"])
    return thrs

def apply_threshold(df, thrs, grouped: bool):
    if not grouped:
        thr = thrs[("ALL", "ALL")]
        return df["unc_entropy"] <= thr

    # grouped: per-row threshold
    keep = []
    for _, r in df.iterrows():
        thr = thrs[(r["view"], r["instant"])]
        keep.append(r["unc_entropy"] <= thr)
    return pd.Series(keep, index=df.index)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robust_dir", required=True)
    ap.add_argument("--val_entropy_csv", required=True)
    ap.add_argument("--model_tag", default="model")
    ap.add_argument("--file_prefix", required=True,
                    help="Only include robustness CSVs whose filename starts with this prefix, e.g. baseline_ or laweighted_.")
    ap.add_argument("--grouped", action="store_true")
    ap.add_argument("--targets", default="0.95,0.90,0.80,0.70")
    ap.add_argument("--out_csv", default="runs/robust/threshold_robustness.csv")
    args = ap.parse_args()

    targets = [float(x) for x in args.targets.split(",")]
    root = Path(args.robust_dir)

    files = sorted(root.glob(f"{args.file_prefix}*.csv"))
    if not files:
        raise SystemExit(f"No files matching {args.file_prefix}*.csv in {root}")

    # Keep only per-sample robustness CSVs
    good = []
    for f in files:
        head = f.read_text().splitlines()[0]
        if "unc_entropy" in head and "dice_mean" in head and "dice_la" in head:
            good.append(f)

    rows = []
    for f in good:
        name = f.name
        m = re.search(r"_(\w+)_s([123])\.csv$", name)
        corr = m.group(1) if m else "unknown"
        sev = int(m.group(2)) if m else -1

        df = pd.read_csv(f)
        df["unc_entropy"] = pd.to_numeric(df["unc_entropy"], errors="coerce")
        df["dice_mean"] = pd.to_numeric(df["dice_mean"], errors="coerce")
        df["dice_la"] = pd.to_numeric(df["dice_la"], errors="coerce")

        all_s = summarize(df)

        for t in targets:
            thrs = load_thresholds(args.val_entropy_csv, t, args.grouped)
            keep_mask = apply_threshold(df, thrs, args.grouped)
            kept = df[keep_mask]
            kept_s = summarize(kept)
            cov = len(kept) / len(df)

            rows.append({
                "model": args.model_tag,
                "corruption": corr,
                "severity": sev,
                "strategy": "grouped_thr" if args.grouped else "global_thr",
                "target_val_cov": t,
                "achieved_test_cov": cov,
                "all_dice": all_s["dice"],
                "all_la": all_s["la"],
                "kept_dice": kept_s["dice"],
                "kept_la": kept_s["la"],
                "kept_n": kept_s["n"],
                "file": str(f),
            })

    out = pd.DataFrame(rows).sort_values(["corruption", "severity", "strategy", "target_val_cov", "model"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(out.head(20).to_string(index=False))
    print(f"\n[DONE] wrote {args.out_csv}")

if __name__ == "__main__":
    main()
