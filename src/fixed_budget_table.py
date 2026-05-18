import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np

def nanmean(x):
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(x.values))

def summarize(df):
    return {"n": len(df), "dice": nanmean(df["dice_mean"]), "la": nanmean(df["dice_la"])}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robust_dir", required=True)
    ap.add_argument("--model_tag", default="baseline")
    ap.add_argument("--budgets", default="0.8,0.5")
    ap.add_argument("--out_csv", default="runs/robust/fixed_budget.csv")
    args = ap.parse_args()

    budgets = [float(x) for x in args.budgets.split(",")]

    root = Path(args.robust_dir)
    files = sorted(root.glob("*.csv"))
    good = []
    for f in files:
        head = f.read_text().splitlines()[0]
        if "unc_entropy" in head and "dice_mean" in head and "dice_la" in head:
            good.append(f)

    rows = []
    for f in good:
        name = f.name
        m = re.search(r"_(\w+)_s([123])\.csv$", name)
        if m:
            corr = m.group(1)
            sev = int(m.group(2))
        else:
            corr, sev = "unknown", -1

        df = pd.read_csv(f)
        df["unc_entropy"] = pd.to_numeric(df["unc_entropy"], errors="coerce")
        df["dice_mean"] = pd.to_numeric(df["dice_mean"], errors="coerce")
        df["dice_la"] = pd.to_numeric(df["dice_la"], errors="coerce")

        all_s = summarize(df)
        df_sorted = df.sort_values("unc_entropy").reset_index(drop=True)

        for b in budgets:
            k = max(1, int(round(len(df_sorted) * b)))
            kept = df_sorted.iloc[:k]
            s = summarize(kept)
            rows.append({
                "model": args.model_tag,
                "corruption": corr,
                "severity": sev,
                "strategy": f"top{int(b*100)}",
                "target_val_cov": np.nan,
                "achieved_test_cov": b,
                "all_dice": all_s["dice"],
                "all_la": all_s["la"],
                "kept_dice": s["dice"],
                "kept_la": s["la"],
                "kept_n": s["n"],
                "file": str(f),
            })

    out = pd.DataFrame(rows).sort_values(["corruption","severity","strategy","model"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(out.head(20).to_string(index=False))
    print(f"\n[DONE] wrote {args.out_csv}")

if __name__ == "__main__":
    main()
