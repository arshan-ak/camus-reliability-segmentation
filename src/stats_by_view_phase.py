import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch

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
        raise RuntimeError("No items found.")
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

def nanmean(x: List[float]) -> float:
    arr = np.array(x, dtype=np.float32)
    return float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camus_root", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--roi", type=int, default=256)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    args = ap.parse_args()

    camus_root = Path(args.camus_root)
    nifti_root = camus_root / "CAMUS_public" / "database_nifti"
    split_root = camus_root / "CAMUS_public" / "database_split"
    split_file = split_root / ("subgroup_testing.txt" if args.split == "test" else "subgroup_validation.txt")

    patients = read_patient_list(split_file)
    items = build_items(nifti_root, patients)

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

    # group -> list of per-class dice
    # groups: (view, instant)
    groups: Dict[Tuple[str, str], Dict[int, List[float]]] = {}
    for v in VIEWS:
        for inst in INSTANTS:
            groups[(v, inst)] = {c: [] for c in range(1, args.num_classes)}

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            meta = items[idx]
            v = meta["view"]
            inst = meta["instant"]

            x = batch["image"].to(device)
            y = batch["label"].to(device)

            logits = model(x)
            pr = torch.argmax(logits, dim=1).detach().cpu().numpy()[0]     # (H,W)
            gt = y.detach().cpu().numpy()[0, 0].astype(np.int32)           # (H,W)

            for c in range(1, args.num_classes):
                d = dice_for_class(pr, gt, c)
                groups[(v, inst)][c].append(d)

    # print table
    print(f"\nSplit: {args.split.upper()}  (roi={args.roi})\n")
    header = ["Group", "MeanDice"] + [CLASS_NAMES.get(c, f"C{c}") for c in range(1, args.num_classes)]
    print("{:<12} {:>8} {:>8} {:>8} {:>8}".format(*header))

    # overall aggregation too
    overall = {c: [] for c in range(1, args.num_classes)}
    for (v, inst), dct in groups.items():
        for c, lst in dct.items():
            overall[c].extend(lst)

    def row_from(dct: Dict[int, List[float]]):
        per_class = [nanmean(dct[c]) for c in range(1, args.num_classes)]
        mean_d = nanmean(per_class)
        return mean_d, per_class

    for v in VIEWS:
        for inst in INSTANTS:
            mean_d, per_class = row_from(groups[(v, inst)])
            print("{:<12} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f}".format(f"{v}_{inst}", mean_d, *per_class))

    mean_all, pc_all = row_from(overall)
    print("-" * 44)
    print("{:<12} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f}\n".format("OVERALL", mean_all, *pc_all))

if __name__ == "__main__":
    main()
