import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def nanmean(x):
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(x.values))

def summarize(df):
    return {
        "n": len(df),
        "dice": nanmean(df["dice_mean"]),
        "la": nanmean(df["dice_la"]),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--budgets", default="0.8,0.5")  # keep fractions
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["unc_entropy"] = pd.to_numeric(df["unc_entropy"], errors="coerce")
    df["dice_mean"] = pd.to_numeric(df["dice_mean"], errors="coerce")
    df["dice_la"] = pd.to_numeric(df["dice_la"], errors="coerce")

    print(f"\nFile: {args.csv}")
    s_all = summarize(df)
    print(f"ALL:    n={s_all['n']}  Dice={s_all['dice']:.4f}  LA={s_all['la']:.4f}")

    df_sorted = df.sort_values("unc_entropy").reset_index(drop=True)

    for b in [float(x) for x in args.budgets.split(",")]:
        k = max(1, int(round(len(df_sorted) * b)))
        kept = df_sorted.iloc[:k]
        s = summarize(kept)
        print(f"TOP{int(b*100)}%: n={s['n']}  Dice={s['dice']:.4f}  LA={s['la']:.4f}")

if __name__ == "__main__":
    main()
