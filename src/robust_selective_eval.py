import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from monai.data import DataLoader, Dataset
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ScaleIntensityRangePercentilesd,
    ResizeWithPadOrCropd,
)

VIEWS = ("2CH", "4CH")
INSTANTS = ("ED", "ES")


# ---------------------------
# Dataset utilities
# ---------------------------
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
                gt = case_dir / f"{stem}_gt.nii.gz"
                if img.exists() and gt.exists():
                    items.append({"image": str(img), "label": str(gt), "case": p, "view": view, "instant": inst})
    if not items:
        raise RuntimeError(f"No items found under {nifti_root}. Check filenames.")
    return items


def get_tf(roi: int):
    return Compose(
        [
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            ScaleIntensityRangePercentilesd(
                keys="image", lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
            ),
            ResizeWithPadOrCropd(keys=("image", "label"), spatial_size=(roi, roi)),
            EnsureTyped(keys=("image", "label"), dtype=(torch.float32, torch.int64)),
        ]
    )


# ---------------------------
# Metrics (simple + stable)
# ---------------------------
def dice_for_class(pred: np.ndarray, gt: np.ndarray, c: int) -> float:
    p = (pred == c)
    g = (gt == c)
    inter = (p & g).sum()
    denom = p.sum() + g.sum()
    if denom == 0:
        return np.nan
    return float((2.0 * inter) / denom)


def nanmean(vals) -> float:
    a = np.array(list(vals), dtype=np.float32)
    return float(np.nanmean(a)) if np.any(~np.isnan(a)) else float("nan")


# ---------------------------
# Corruptions (input in [0,1])
# ---------------------------
def corruption(x: torch.Tensor, kind: str, sev: int) -> torch.Tensor:
    """
    x: (1,H,W) float in [0,1]
    sev: 1..3
    """
    sev = int(sev)
    if kind == "none":
        return x

    if kind == "gain":  # brightness shift
        offsets = [0.03, 0.07, 0.12]
        return (x + offsets[sev - 1]).clamp(0, 1)

    if kind == "contrast":  # gamma-like (mid-tone)
        gammas = [0.85, 0.70, 0.55]
        g = gammas[sev - 1]
        return (x.clamp(1e-6, 1.0) ** g).clamp(0, 1)

    if kind == "noise":  # Gaussian proxy for speckle variation
        stds = [0.015, 0.03, 0.05]
        return (x + torch.randn_like(x) * stds[sev - 1]).clamp(0, 1)

    if kind == "blur":  # avg blur via conv
        ks = [3, 5, 7][sev - 1]
        pad = ks // 2
        w = torch.ones((1, 1, ks, ks), device=x.device) / float(ks * ks)
        y = F.conv2d(x.unsqueeze(0), w, padding=pad)
        return y[0].clamp(0, 1)

    if kind == "occlusion":  # horizontal shadow band
        H, W = x.shape[-2], x.shape[-1]
        frac = [0.08, 0.14, 0.20][sev - 1]
        band = max(1, int(H * frac))
        y0 = int(H * 0.35)
        y1 = min(H, y0 + band)
        y = x.clone()
        y[:, y0:y1, :] = 0.0
        return y

    raise ValueError(f"Unknown corruption kind: {kind}")


# ---------------------------
# Uncertainty: normalized entropy
# ---------------------------
def entropy_uncertainty(prob: torch.Tensor, pred: torch.Tensor) -> float:
    """
    prob: (C,H,W)
    pred: (H,W) integer labels
    returns mean normalized entropy over predicted foreground
    """
    eps = 1e-8
    C = prob.shape[0]
    logC = float(np.log(C))

    p = prob.detach().cpu().numpy().astype(np.float32)  # (C,H,W)
    ent = -(p * np.log(p + eps)).sum(axis=0) / logC     # (H,W)

    pr = pred.detach().cpu().numpy()
    fg = (pr != 0)
    return float(ent[fg].mean()) if fg.any() else float(ent.mean())


# ---------------------------
# Thresholds from VAL entropy CSV (clean, no corruption)
# ---------------------------
def load_val_thresholds(val_csv: Path, target_cov: float, grouped: bool) -> Dict[Tuple[str, str], float]:
    import pandas as pd

    df = pd.read_csv(val_csv)
    df["unc_entropy"] = pd.to_numeric(df["unc_entropy"], errors="coerce")

    if not grouped:
        df = df.sort_values("unc_entropy").reset_index(drop=True)
        k = max(1, int(round(len(df) * target_cov)))
        thr = float(df.loc[k - 1, "unc_entropy"])
        return {("ALL", "ALL"): thr}

    thrs = {}
    for view in VIEWS:
        for inst in INSTANTS:
            g = df[(df["view"] == view) & (df["instant"] == inst)].sort_values("unc_entropy").reset_index(drop=True)
            if len(g) == 0:
                raise RuntimeError(f"No VAL samples for group {view}_{inst} in {val_csv}")
            k = max(1, int(round(len(g) * target_cov)))
            thrs[(view, inst)] = float(g.loc[k - 1, "unc_entropy"])
    return thrs


# ---------------------------
# Main
# ---------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camus_root", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--roi", type=int, default=256)
    ap.add_argument("--num_classes", type=int, default=4)

    ap.add_argument("--val_entropy_csv", required=True, type=str,
                    help="VAL entropy CSV from step1_entropy_reliability.py (clean images).")
    ap.add_argument("--target_cov", type=float, default=0.80)
    ap.add_argument("--grouped", action="store_true",
                    help="If set, use per-(view,phase) thresholds from VAL; else global threshold.")

    ap.add_argument("--corruption", type=str, default="none",
                    choices=["none", "gain", "contrast", "noise", "blur", "occlusion"])
    ap.add_argument("--severity", type=int, default=1, choices=[1, 2, 3])

    ap.add_argument("--out_csv", type=str, default="runs/robust_eval.csv")
    args = ap.parse_args()

    camus_root = Path(args.camus_root)
    nifti_root = camus_root / "CAMUS_public" / "database_nifti"
    split_root = camus_root / "CAMUS_public" / "database_split"

    test_pat = read_patient_list(split_root / "subgroup_testing.txt")
    items = build_items(nifti_root, test_pat)

    tf = get_tf(args.roi)
    ds = Dataset(items, tf)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=args.num_classes,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=1,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    thrs = load_val_thresholds(Path(args.val_entropy_csv), args.target_cov, args.grouped)

    rows = []
    for i, batch in enumerate(loader):
        meta = items[i]
        tag = f"{meta['case']}_{meta['view']}_{meta['instant']}"

        x = batch["image"][0].to(device)   # (1,H,W)
        y = batch["label"][0, 0].to(device)  # (H,W)

        x_c = corruption(x, args.corruption, args.severity)

        logits = model(x_c.unsqueeze(0))          # (1,C,H,W)
        prob = torch.softmax(logits, dim=1)[0]    # (C,H,W)
        pred = torch.argmax(prob, dim=0)          # (H,W)

        pr = pred.detach().cpu().numpy().astype(np.int32)
        gt = y.detach().cpu().numpy().astype(np.int32)

        dice_pc = [dice_for_class(pr, gt, c) for c in range(1, args.num_classes)]
        dice_mean = nanmean(dice_pc)

        unc = entropy_uncertainty(prob, pred)

        thr = thrs[(meta["view"], meta["instant"])] if args.grouped else thrs[("ALL", "ALL")]
        keep = 1 if (unc <= thr) else 0

        rows.append((tag, meta["view"], meta["instant"], dice_mean, *dice_pc, unc, keep))

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        f.write("tag,view,instant,dice_mean,dice_lv,dice_myo,dice_la,unc_entropy,keep\n")
        for r in rows:
            tag, v, inst, dm, dlv, dmyo, dla, unc, keep = r

            def fmt(x):
                return "" if (isinstance(x, float) and np.isnan(x)) else f"{x:.6f}"

            f.write(f"{tag},{v},{inst},{fmt(dm)},{fmt(dlv)},{fmt(dmyo)},{fmt(dla)},{fmt(unc)},{keep}\n")

    # Summary
    dice_all = [r[3] for r in rows]
    la_all = [r[6] for r in rows]
    keep_mask = np.array([r[8] for r in rows], dtype=np.int32) == 1

    kept = [rows[j] for j in range(len(rows)) if keep_mask[j]]
    dice_kept = [r[3] for r in kept]
    la_kept = [r[6] for r in kept]

    print(f"[DONE] wrote {out}")
    print(
        f"All:  Dice={nanmean(dice_all):.4f}  LA={nanmean(la_all):.4f}  n={len(rows)}  "
        f"corruption={args.corruption} s={args.severity}"
    )
    print(
        f"Kept: Dice={nanmean(dice_kept):.4f}  LA={nanmean(la_kept):.4f}  n={len(kept)}  "
        f"coverage={len(kept)/len(rows):.3f}  thresholding={'grouped' if args.grouped else 'global'}@{args.target_cov:.2f}"
    )


if __name__ == "__main__":
    main()
