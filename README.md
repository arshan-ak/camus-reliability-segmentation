# CAMUS Reliability-Aware Ultrasound Segmentation (Code)

This repository contains code for a reliability-aware cardiac ultrasound segmentation study on CAMUS:
- U-Net segmentation (LV/MYO/LA)
- entropy-based uncertainty
- selective prediction (risk–coverage, failure detection)
- corruption robustness + threshold-transfer / calibration-gap analysis

## Dataset
This repo does NOT include CAMUS data.
Download CAMUS and place it locally as:
  <camus_root>/CAMUS_public/database_nifti/
  <camus_root>/CAMUS_public/database_split/

## Environment
Create/activate your environment (example):
  conda create -n camus-monai python=3.10
  conda activate camus-monai
  pip install -r requirements.txt   # or install manually

## Example commands
Train baseline:
  python src/train_camus_nifti_unet.py --camus_root <camus_root> --out_dir runs/baseline --epochs 150 --batch 1 --roi 256 --amp --accum_steps 2

Evaluate:
  python src/eval_camus_nifti.py --camus_root <camus_root> --ckpt runs/baseline/best.pt --roi 256

Compute uncertainty + risk–coverage:
  python src/step1_entropy_reliability.py --camus_root <camus_root> --ckpt runs/baseline/best.pt --roi 256 --split test --out_dir runs/step1_entropy_test

Robust selective eval (example):
  python src/robust_selective_eval.py --camus_root <camus_root> --ckpt runs/baseline/best.pt --val_entropy_csv runs/step1_entropy_val/val_entropy_metrics.csv --target_cov 0.80 --grouped --corruption occlusion --severity 2 --out_csv runs/robust/baseline_occlusion_s2.csv
