import argparse
from pathlib import Path
import numpy as np
import torch

from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ResizeWithPadOrCropd
)

VIEWS = ("2CH", "4CH")
INSTANTS = ("ED", "ES")

def read_patient_list(txt_path: Path):
    return [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]

def build_items(nifti_root: Path, patients):
    items = []
    for p in patients:
        case_dir = nifti_root / p
        for view in VIEWS:
            for inst in INSTANTS:
                stem = f"{p}_{view}_{inst}"
                gt = case_dir / f"{stem}_gt.nii.gz"
                img = case_dir / f"{stem}.nii.gz"
                if img.exists() and gt.exists():
                    items.append({"label": str(gt)})
    if not items:
        raise RuntimeError("No labels found.")
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camus_root", required=True)
    ap.add_argument("--roi", type=int, default=256)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--la_boost", type=float, default=1.5, help="Extra boost factor for LA weight (class 3).")
    ap.add_argument("--max_items", type=int, default=0, help="0 = use all; else subsample for speed.")
    args = ap.parse_args()

    camus_root = Path(args.camus_root)
    nifti_root = camus_root / "CAMUS_public" / "database_nifti"
    split_root = camus_root / "CAMUS_public" / "database_split"
    train_pat = read_patient_list(split_root / "subgroup_training.txt")

    items = build_items(nifti_root, train_pat)
    if args.max_items and args.max_items > 0:
        items = items[:args.max_items]

    tf = Compose([
        LoadImaged(keys=("label",)),
        EnsureChannelFirstd(keys=("label",)),
        ResizeWithPadOrCropd(keys=("label",), spatial_size=(args.roi, args.roi)),
        EnsureTyped(keys=("label",), dtype=(torch.int64,)),
    ])

    ds = Dataset(items, tf)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    counts = np.zeros(args.num_classes, dtype=np.int64)
    total = 0

    for batch in loader:
        y = batch["label"][0, 0].cpu().numpy().astype(np.int64)  # (H,W)
        for c in range(args.num_classes):
            cnt = (y == c).sum()
            counts[c] += int(cnt)
            total += int(cnt)

    freq = counts / max(1, total)

    # inverse frequency weights; avoid div by zero
    w = 1.0 / np.maximum(freq, 1e-12)

    # normalize so mean foreground weight ~= 1
    fg = w[1:]
    w = w / (fg.mean() + 1e-12)

    # boost LA (class 3)
    if args.num_classes >= 4:
        w[3] *= args.la_boost

    print("Pixel freq:", freq)
    print("CE weights (class 0..C-1):", w)
    print("Use as --ce_weights:", ",".join([f"{x:.6f}" for x in w]))

if __name__ == "__main__":
    main()
