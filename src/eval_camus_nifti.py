import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ScaleIntensityRangePercentilesd, ResizeWithPadOrCropd,
    AsDiscrete
)

VIEWS = ("2CH", "4CH")
INSTANTS = ("ED", "ES")

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
        raise RuntimeError(f"No items found under {nifti_root}. Check filenames.")
    return items

def get_val_tf(roi: int):
    return Compose([
        LoadImaged(keys=("image","label")),
        EnsureChannelFirstd(keys=("image","label")),
        ScaleIntensityRangePercentilesd(keys="image", lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        ResizeWithPadOrCropd(keys=("image","label"), spatial_size=(roi, roi)),
        EnsureTyped(keys=("image","label"), dtype=(torch.float32, torch.int64)),
    ])

@torch.no_grad()
def evaluate(model, loader, num_classes: int, device: torch.device):
    model.eval()

    dice_mean = DiceMetric(include_background=False, reduction="mean")
    dice_pc   = DiceMetric(include_background=False, reduction="mean_batch")
    hd95_mean = HausdorffDistanceMetric(include_background=False, percentile=95)

    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_lab  = AsDiscrete(to_onehot=num_classes)

    dice_mean.reset(); dice_pc.reset(); hd95_mean.reset()

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        logits = model(x)
        prob = torch.softmax(logits, dim=1)

        preds = decollate_batch(prob)
        labs  = decollate_batch(y)

        pred_ohe = [post_pred(p) for p in preds]
        lab_ohe  = [post_lab(l)  for l in labs]

        dice_mean(y_pred=pred_ohe, y=lab_ohe)
        dice_pc(y_pred=pred_ohe, y=lab_ohe)
        hd95_mean(y_pred=pred_ohe, y=lab_ohe)

    return {
        "dice_mean": float(dice_mean.aggregate().item()),
        "dice_per_class": dice_pc.aggregate().cpu().numpy().tolist(),
        "hd95_mean": float(hd95_mean.aggregate().item()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camus_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--roi", type=int, default=256)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    camus_root = Path(args.camus_root)
    nifti_root = camus_root / "CAMUS_public" / "database_nifti"
    split_root = camus_root / "CAMUS_public" / "database_split"

    val_pat  = read_patient_list(split_root / "subgroup_validation.txt")
    test_pat = read_patient_list(split_root / "subgroup_testing.txt")

    val_items  = build_items(nifti_root, val_pat)
    test_items = build_items(nifti_root, test_pat)

    tf = get_val_tf(args.roi)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        spatial_dims=2, in_channels=1, out_channels=args.num_classes,
        channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=1
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    pin = device.type == "cuda"
    # DataLoaders: pin_memory=False (avoid pin memory thread)
    val_loader  = DataLoader(Dataset(val_items, tf),  batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    test_loader = DataLoader(Dataset(test_items, tf), batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    val_res  = evaluate(model, val_loader,  args.num_classes, device)
    test_res = evaluate(model, test_loader, args.num_classes, device)

    # Pretty print
    def fmt(res):
        pc = ["%.3f" % x for x in res["dice_per_class"]]
        return f"Dice(mean)={res['dice_mean']:.4f} | Dice(pc)={pc} | HD95={res['hd95_mean']:.2f}"

    print("[VAL ]", fmt(val_res))
    print("[TEST]", fmt(test_res))

if __name__ == "__main__":
    main()
