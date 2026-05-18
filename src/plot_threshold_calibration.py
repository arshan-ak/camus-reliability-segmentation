import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def norm_corr(c: str) -> str:
    # make names nicer for paper
    if c.startswith("v2_"):
        return c.replace("v2_", "")
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="thr_robust_*.csv from threshold_robustness_sweep.py")
    ap.add_argument("--out_dir", default="runs/robust/calib_plots")
    ap.add_argument("--metric", default="achieved_test_cov",
                    choices=["achieved_test_cov", "kept_la", "kept_dice"],
                    help="What to plot vs severity.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    # normalize names
    df["corruption"] = df["corruption"].astype(str).map(norm_corr)

    # safety: drop unknown severities
    df = df[df["severity"].isin([1,2,3])].copy()

    for corr, g in df.groupby("corruption"):
        plt.figure()
        for t, gg in g.groupby("target_val_cov"):
            # sort by severity
            gg = gg.sort_values("severity")
            plt.plot(gg["severity"].values, gg[args.metric].values, marker="o", label=f"VAL target={t:.2f}")

        plt.xlabel("Severity")
        plt.ylabel(args.metric)
        title = f"{df['model'].iloc[0]} | {df['strategy'].iloc[0]} | {corr}"
        plt.title(title)
        plt.xticks([1,2,3])
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        out_path = out_dir / f"{Path(args.csv).stem}_{corr}_{args.metric}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    print(f"[DONE] wrote plots to {out_dir}")

if __name__ == "__main__":
    main()
