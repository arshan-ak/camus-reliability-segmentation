import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from monai.config import print_config
from monai.data import DataLoader, Dataset, decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ScaleIntensityRangePercentilesd,
    ResizeWithPadOrCropd,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    RandAdjustContrastd,
    AsDiscrete,
)
from monai.utils import set_determinism


VIEWS = ("2CH", "4CH")
INSTANTS = ("ED", "ES")


def read_patient_list(txt_path: Path) -> List[str]:
    """
    Each line is a patient folder name like: patient0001
    """
    patients = []
    for line in txt_path.read_text().splitlines():
        s = line.strip()
        if s:
            patients.append(s)
    return patients


def build_items(nifti_root: Path, patients: List[str]) -> List[Dict]:
    """
    Build {image,label,case,view,instant} items for ED/ES and 2CH/4CH.
    Expected file naming inside each patient folder:
      patientXXXX_2CH_ED.nii.gz
      patientXXXX_2CH_ED_gt.nii.gz
    """
    items: List[Dict] = []
    for p in patients:
        case_dir = nifti_root / p
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Missing patient folder: {case_dir}")

        for view in VIEWS:
            for inst in INSTANTS:
                stem = f"{p}_{view}_{inst}"
                img = case_dir / f"{stem}.nii.gz"
                gt = case_dir / f"{stem}_gt.nii.gz"

                if img.exists() and gt.exists():
                    items.append(
                        {
                            "image": str(img),
                            "label": str(gt),
                            "case": p,
                            "view": view,
                            "instant": inst,
                        }
                    )

    if not items:
        raise RuntimeError(
            f"No items found. Check that files follow patientXXXX_2CH_ED.nii.gz and *_gt.nii.gz under {nifti_root}."
        )
    return items


def get_transforms(roi: int):
    # Ultrasound: percentile scaling is robust to gain/contrast shifts
    base = [
        LoadImaged(keys=("image", "label")),  # nibabel reader is fine for .nii.gz
        EnsureChannelFirstd(keys=("image", "label")),
        ScaleIntensityRangePercentilesd(
            keys="image", lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        ),
        ResizeWithPadOrCropd(keys=("image", "label"), spatial_size=(roi, roi)),
        EnsureTyped(keys=("image", "label"), dtype=(torch.float32, torch.int64)),
    ]

    train_aug = [
        RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=1),
        RandRotate90d(keys=("image", "label"), prob=0.3, max_k=3),
        RandAffined(
            keys=("image", "label"),
            prob=0.35,
            rotate_range=(0.0, 0.0, np.deg2rad(10.0)),
            translate_range=(10, 10),
            scale_range=(0.12, 0.12),
            padding_mode="border",
        ),
        RandGaussianNoised(keys="image", prob=0.15, mean=0.0, std=0.02),
        RandAdjustContrastd(keys="image", prob=0.20, gamma=(0.7, 1.5)),
    ]

    return Compose(base + train_aug), Compose(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camus_root", type=str, required=True,
                    help="Path containing CAMUS_public/ (i.e., the folder that has database_nifti and database_split).")
    ap.add_argument("--out_dir", type=str, default="runs/camus_unet_nifti_mx350")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2, help="MX350: start 1-2")
    ap.add_argument("--roi", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--num_classes", type=int, default=4, help="Common CAMUS labels: 0..3")
    args = ap.parse_args()

    print_config()
    set_determinism(args.seed)

    camus_root = Path(args.camus_root)
    nifti_root = camus_root / "CAMUS_public" / "database_nifti"
    split_root = camus_root / "CAMUS_public" / "database_split"

    train_pat = read_patient_list(split_root / "subgroup_training.txt")
    val_pat = read_patient_list(split_root / "subgroup_validation.txt")
    test_pat = read_patient_list(split_root / "subgroup_testing.txt")

    train_items = build_items(nifti_root, train_pat)
    val_items = build_items(nifti_root, val_pat)
    test_items = build_items(nifti_root, test_pat)  # not used in training loop yet

    print(f"[INFO] Train items: {len(train_items)} | Val items: {len(val_items)} | Test items: {len(test_items)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.out_dir)

    train_tf, val_tf = get_transforms(args.roi)
    train_ds = Dataset(train_items, transform=train_tf)
    val_ds = Dataset(val_items, transform=val_tf)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    # Small UNet (2GB VRAM friendly)
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=args.num_classes,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=1,
    ).to(device)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    dice_mean = DiceMetric(include_background=False, reduction="mean")
    dice_per_class = DiceMetric(include_background=False, reduction="mean_batch")
    hd95_mean = HausdorffDistanceMetric(include_background=False, percentile=95)

    post_pred = AsDiscrete(argmax=True, to_onehot=args.num_classes)
    post_lab = AsDiscrete(to_onehot=args.num_classes)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0

        for step, batch in enumerate(train_loader, start=1):
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y) / args.accum_steps

            scaler.scale(loss).backward()

            if step % args.accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            running += loss.item() * args.accum_steps

        train_loss = running / max(1, len(train_loader))
        writer.add_scalar("train/loss", train_loss, epoch)

        # ---- Validation ----
        model.eval()
        dice_mean.reset()
        dice_per_class.reset()
        hd95_mean.reset()

        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)

                with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                    logits = model(x)
                    prob = torch.softmax(logits, dim=1)

                preds = decollate_batch(prob)
                labs = decollate_batch(y)

                pred_ohe = [post_pred(p) for p in preds]
                lab_ohe = [post_lab(l) for l in labs]

                dice_mean(y_pred=pred_ohe, y=lab_ohe)
                dice_per_class(y_pred=pred_ohe, y=lab_ohe)
                hd95_mean(y_pred=pred_ohe, y=lab_ohe)

        val_dice = dice_mean.aggregate().item()
        val_dice_pc = dice_per_class.aggregate().cpu().numpy().tolist()  # per-class (excluding background)
        val_hd95 = hd95_mean.aggregate().item()

        writer.add_scalar("val/dice_mean", val_dice, epoch)
        writer.add_scalar("val/hd95_mean", val_hd95, epoch)

        # log per-class dice as separate scalars
        for i, d in enumerate(val_dice_pc, start=1):  # class indices 1..C-1 (background excluded)
            writer.add_scalar(f"val/dice_class_{i}", float(d), epoch)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_dice={val_dice:.4f} | val_hd95={val_hd95:.2f} | "
            f"val_dice_per_class={['%.3f'%x for x in val_dice_pc]}"
        )

        if val_dice > best:
            best = val_dice
            best_epoch = epoch
            ckpt = Path(args.out_dir) / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "best_dice": best,
                    "args": vars(args),
                },
                ckpt,
            )
            print(f"  [BEST] saved: {ckpt}")

    writer.close()
    print(f"[DONE] best_dice={best:.4f} @ epoch={best_epoch}")


if __name__ == "__main__":
    main()
