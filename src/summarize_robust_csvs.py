import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np

def nanmean(x):
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(x.values))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="runs/robust")
    args = ap.parse_args()

    root = Path(args.dir)
    files = sorted(root.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSVs found in {root}")

    rows = []
    pat = re.compile(r"^(baseline|laweighted)_(\w+)_s([123])\.csv$")

    for f in files:
        m = pat.match(f.name)
        if not m:
            continue
        model, corr, sev = m.group(1), m.group(2), int(m.group(3))
        df = pd.read_csv(f)
        all_dice = nanmean(df["dice_mean"])
        all_la = nanmean(df["dice_la"])

        kept = df[df["keep"] == 1]
        cov = len(kept) / len(df)
        kept_dice = nanmean(kept["dice_mean"]) if len(kept) else float("nan")
        kept_la = nanmean(kept["dice_la"]) if len(kept) else float("nan")

        rows.append([model, corr, sev, all_dice, all_la, cov, kept_dice, kept_la, len(kept)])

    out = pd.DataFrame(rows, columns=[
        "model","corruption","severity",
        "all_dice","all_la",
        "coverage","kept_dice","kept_la","kept_n"
    ]).sort_values(["corruption","severity","model"])

    print(out.to_string(index=False))
    out.to_csv(root / "summary_table.csv", index=False)
    print(f"\n[DONE] wrote {root/'summary_table.csv'}")

if __name__ == "__main__":
    main()
