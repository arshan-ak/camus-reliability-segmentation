import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from monai.data import DataLoader, Dataset
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ScaleIntensityRangePercentilesd, ResizeWithPadOrCropd
)

VIEWS = ("2CH", "4CH")
INSTANTS = ("ED", "ES")
CLASS_NAMES = {1: "LV", 2: "MYO", 3: "LA"}

def read_patient_list(txt_path: Path) -> List[str]:
    return [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]

def build_items(nifti_root: Path, patients: List[str]) -> List[Dict]:
    items: List[Dict] = []
    for p in patients:
        case_dir = nifti_root / p
        for view in VIEWS:
            for inst in INSTANTS:
                stem = f"{p}_{view}_{inst}"
                img = case_dir / f"{stem}.nii.gz"
                gt  = case_dir / f"{stem}_gt.nii.gz"
                if img.exists() and gt.exists():
                    items.append({"image": str(img), "label": str(gt), "case": p, "view": view, "instant": inst})
    if not items:
        raise RuntimeError("No items found. Check paths/filenames.")
    return items

def get_tf(roi: int):
    return Compose([
        LoadImaged(keys=("image","label")),
        EnsureChannelFirstd(keys=("image","label")),
        ScaleIntensityRangePercentilesd(keys="image", lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        ResizeWithPadOrCropd(keys=("image","label"), spatial_size=(roi, roi)),
        EnsureTyped(keys=("image","label"), dtype=(torch.float32, torch.int64)),
    ])

def dice_for_class(pred: np.ndarray, gt: np.ndarray, c: int) -> float:
    p = (pred == c)
    g = (gt == c)
    inter = (p & g).sum()
    denom = p.sum() + g.sum()
    if denom == 0:
        return np.nan
    return float((2.0 * inter) / denom)

def nanmean(vals: List[float]) -> float:
    arr = np.array(vals, dtype=np.float32)
    return float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float("nan")

def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    # simple Spearman: corr(rank(x), rank(y))
    def rank(a):
        tmp = a.argsort()
        r = np.empty_like(tmp, dtype=np.float32)
        r[tmp] = np.arange(len(a), dtype=np.float32)
        return r
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = (np.sqrt((rx**2).sum()) * np.sqrt((ry**2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camus_root", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--roi", type=int, default=256)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    ap.add_argument("--out_dir", type=str, default="runs/step1_entropy")
    args = ap.parse_args()

    camus_root = Path(args.camus_root)
    nifti_root = camus_root / "CAMUS_public" / "database_nifti"
    split_root = camus_root / "CAMUS_public" / "database_split"
    split_file = split_root / ("subgroup_testing.txt" if args.split == "test" else "subgroup_validation.txt")

    patients = read_patient_list(split_file)
    items = build_items(nifti_root, patients)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.split}_entropy_metrics.csv"

    tf = get_tf(args.roi)
    ds = Dataset(items, tf)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=2, in_channels=1, out_channels=args.num_classes,
        channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=1
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = []
    eps = 1e-8
    logC = float(np.log(args.num_classes))

    for idx, batch in enumerate(loader):
        meta = items[idx]
        tag = f"{meta['case']}_{meta['view']}_{meta['instant']}"

        x = batch["image"].to(device)  # (1,1,H,W)
        y = batch["label"].to(device)  # (1,1,H,W)

        logits = model(x)
        prob = torch.softmax(logits, dim=1)  # (1,C,H,W)

        pred = torch.argmax(prob, dim=1)  # (1,H,W)

        # numpy
        pr = pred[0].detach().cpu().numpy().astype(np.int32)
        gt = y[0, 0].detach().cpu().numpy().astype(np.int32)

        # Dice per class (skip background)
        dice_pc = [dice_for_class(pr, gt, c) for c in range(1, args.num_classes)]
        dice_mean = nanmean(dice_pc)

        # Entropy map: H = -sum p log p, normalized by log(C) -> [0,1]
        p = prob[0].detach().cpu().numpy().astype(np.float32)  # (C,H,W)
        ent = -(p * np.log(p + eps)).sum(axis=0) / logC         # (H,W)

        # scalar uncertainty: mean entropy over predicted foreground
        fg = (pr != 0)
        if fg.any():
            unc = float(ent[fg].mean())
        else:
            unc = float(ent.mean())

        rows.append((tag, meta["view"], meta["instant"], dice_mean, *dice_pc, unc))

    # write CSV
    with out_csv.open("w") as f:
        f.write("tag,view,instant,dice_mean,dice_lv,dice_myo,dice_la,unc_entropy\n")
        for r in rows:
            tag, view, inst, dm, dlv, dmyo, dla, unc = r
            def fmt(v): return "" if (isinstance(v, float) and np.isnan(v)) else f"{v:.6f}"
            f.write(f"{tag},{view},{inst},{fmt(dm)},{fmt(dlv)},{fmt(dmyo)},{fmt(dla)},{fmt(unc)}\n")

    # arrays for plots/corr
    dice_arr = np.array([r[3] for r in rows], dtype=np.float32)
    unc_arr  = np.array([r[7] for r in rows], dtype=np.float32)

    # Spearman correlation (expect negative: higher uncertainty -> lower dice)
    rho = spearman_corr(unc_arr, dice_arr)

    # Scatter plot
    plt.figure()
    plt.scatter(unc_arr, dice_arr, s=10)
    plt.xlabel("Uncertainty (mean normalized entropy, foreground)")
    plt.ylabel("Dice (mean over LV/MYO/LA)")
    plt.title(f"{args.split.upper()} scatter: uncertainty vs Dice (Spearman rho={rho:.3f})")
    plt.grid(True)
    plt.tight_layout()
    scatter_path = out_dir / f"{args.split}_scatter_unc_vs_dice.png"
    plt.savefig(scatter_path, dpi=150)

    # Risk–coverage curve
    order = np.argsort(unc_arr)  # low uncertainty first (most confident)
    coverages = []
    dices = []
    for cov in [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]:
        k = max(1, int(round(len(rows) * cov)))
        idxs = order[:k]
        coverages.append(cov)
        dices.append(float(np.nanmean(dice_arr[idxs])))

    plt.figure()
    plt.plot(coverages, dices, marker="o")
    plt.xlabel("Coverage (fraction of cases kept)")
    plt.ylabel("Mean Dice on kept cases")
    plt.title(f"{args.split.upper()} risk–coverage (sorted by uncertainty)")
    plt.grid(True)
    plt.tight_layout()
    rc_path = out_dir / f"{args.split}_risk_coverage.png"
    plt.savefig(rc_path, dpi=150)

    # Print quick summary
    print(f"[DONE] CSV: {out_csv}")
    print(f"[DONE] Scatter plot: {scatter_path}")
    print(f"[DONE] Risk–coverage plot: {rc_path}")
    print(f"[INFO] Spearman rho(unc, dice) = {rho:.4f} (expect negative for a good reliability signal)")

    # Show top-10 highest uncertainty samples
    worst_unc = sorted(rows, key=lambda r: r[7], reverse=True)[:10]
    print("\nTop-10 highest uncertainty:")
    for r in worst_unc:
        tag, view, inst, dm, dlv, dmyo, dla, unc = r
        print(f"  unc={unc:.4f}  dice={dm:.4f}  {tag}  pc=[{dlv:.3f},{dmyo:.3f},{dla:.3f}]")

if __name__ == "__main__":
    main()
