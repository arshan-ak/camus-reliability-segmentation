import argparse
import pandas as pd

def load_summary(csv_path):
    df = pd.read_csv(csv_path)
    df["corruption"] = df["corruption"].astype(str).str.replace("^v2_", "", regex=True)
    df = df[df["target_val_cov"].notna()].copy()
    df["gap"] = df["target_val_cov"] - df["achieved_test_cov"]
    out = (
        df.groupby(["model","strategy","corruption"], as_index=False)
          .agg(
              mean_gap=("gap","mean"),
              max_gap=("gap","max"),
              mean_achieved=("achieved_test_cov","mean"),
              min_achieved=("achieved_test_cov","min"),
          )
    )
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--out_csv", default="runs/robust/calibration_gap_summary_all.csv")
    args = ap.parse_args()

    all_rows = []
    for p in args.csvs:
        all_rows.append(load_summary(p))
    merged = pd.concat(all_rows, ignore_index=True).sort_values(
        ["corruption","model","strategy"]
    )
    merged.to_csv(args.out_csv, index=False)
    print(merged.to_string(index=False))
    print(f"\n[DONE] wrote {args.out_csv}")

if __name__ == "__main__":
    main()

