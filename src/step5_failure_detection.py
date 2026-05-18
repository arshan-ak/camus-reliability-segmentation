import argparse
import pandas as pd
import numpy as np

def auc_roc(scores, labels):
    # labels: 1 = failure, 0 = ok
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    order = np.argsort(scores)[::-1]
    labels = labels[order]

    P = labels.sum()
    N = len(labels) - P
    if P == 0 or N == 0:
        return np.nan

    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    tpr = tps / P
    fpr = fps / N
    # trapezoid
    return np.trapz(tpr, fpr)

def auc_pr(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    order = np.argsort(scores)[::-1]
    labels = labels[order]

    P = labels.sum()
    if P == 0:
        return np.nan

    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    prec = tps / (tps + fps + 1e-12)
    rec = tps / P
    return np.trapz(prec, rec)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--unc_col", default="unc_entropy")
    ap.add_argument("--fail_dice_mean", type=float, default=0.70)
    ap.add_argument("--fail_dice_la", type=float, default=0.50)
    ap.add_argument("--by_group", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    for c in ["dice_mean","dice_la", args.unc_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # failure definition
    fail = (df["dice_mean"] < args.fail_dice_mean) | (df["dice_la"] < args.fail_dice_la)
    df["fail"] = fail.astype(int)

    auc = auc_roc(df[args.unc_col].values, df["fail"].values)
    aupr = auc_pr(df[args.unc_col].values, df["fail"].values)

    print(f"Failures: {df['fail'].sum()}/{len(df)}  ({df['fail'].mean():.3f})")
    print(f"AUROC(unc -> fail) = {auc:.4f}")
    print(f"AUPRC(unc -> fail) = {aupr:.4f}")

    if args.by_group:
        print("\nBy group:")
        for (v, inst), g in df.groupby(["view","instant"]):
            auc_g = auc_roc(g[args.unc_col].values, g["fail"].values)
            aupr_g = auc_pr(g[args.unc_col].values, g["fail"].values)
            print(f"{v}_{inst}: n={len(g)} fail={g['fail'].mean():.3f}  AUROC={auc_g:.4f}  AUPRC={aupr_g:.4f}")

if __name__ == "__main__":
    main()
