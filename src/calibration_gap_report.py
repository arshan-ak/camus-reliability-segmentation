import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="thr_robust_*.csv from threshold_robustness_sweep.py")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Normalize corruption naming (optional, keeps v2_ readable)
    df["corruption"] = df["corruption"].astype(str).str.replace("^v2_", "", regex=True)

    # Only rows with numeric targets
    df = df[df["target_val_cov"].notna()].copy()

    # Calibration gap
    df["gap"] = df["target_val_cov"] - df["achieved_test_cov"]

    # ---- Summary by corruption ----
    summary = (
        df.groupby(["model", "strategy", "corruption"], as_index=False)
          .agg(
              mean_gap=("gap", "mean"),
              max_gap=("gap", "max"),
              mean_achieved=("achieved_test_cov", "mean"),
              min_achieved=("achieved_test_cov", "min"),
          )
          .sort_values(["mean_gap", "max_gap"], ascending=False)
    )

    # ---- Worst cases overall ----
    worst = (
        df.sort_values("gap", ascending=False)
          .head(args.topk)[
              ["model","strategy","corruption","severity","target_val_cov","achieved_test_cov","gap","kept_n"]
          ]
    )

    print("\n=== WORST GAP CASES (largest coverage collapse) ===")
    print(worst.to_string(index=False))

    print("\n=== GAP SUMMARY BY CORRUPTION (mean/max gap; lower is better) ===")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
