#!/usr/bin/env python3
"""Audit split CSVs and create deterministic patient-isolated fold manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--include-test-in-cv", action="store_true",
                    help="Normally keep the original test patients sealed")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    frames = {k: pd.read_csv(getattr(args, k)).assign(source_split=k)
              for k in ["train", "val", "test"]}
    required = {"uuid", "patient_id", "slide_name", "image_path", "x", "y", "label"}
    audit = {"splits": {}, "warnings": []}
    for name, d in frames.items():
        if missing := required - set(d):
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        audit["splits"][name] = {
            "patches": int(len(d)), "patients": int(d.patient_id.nunique()),
            "slides": int(d.slide_name.nunique()),
            "class_0": int((d.label == 0).sum()), "class_1": int((d.label == 1).sum()),
            "missing_cells": int(d.isna().sum().sum()),
            "duplicate_uuid_xy": int(d.duplicated(["uuid", "x", "y"]).sum()),
        }
    patient_sets = {k: set(v.patient_id) for k, v in frames.items()}
    audit["patient_overlap"] = {
        "train_val": sorted(patient_sets["train"] & patient_sets["val"]),
        "train_test": sorted(patient_sets["train"] & patient_sets["test"]),
        "val_test": sorted(patient_sets["val"] & patient_sets["test"]),
    }
    d = pd.concat(frames.values(), ignore_index=True)
    audit["overall"] = {
        "patches": int(len(d)), "patients": int(d.patient_id.nunique()),
        "slides": int(d.slide_name.nunique()), "source_images": int(d.image_path.nunique()),
        "class_0": int((d.label == 0).sum()), "class_1": int((d.label == 1).sum()),
        "duplicate_uuid_xy": int(d.duplicated(["uuid", "x", "y"]).sum()),
    }
    if d.image_path.astype(str).str.startswith("/home/aistudio/").all():
        audit["warnings"].append("All image paths point to /home/aistudio and were not verified locally.")
    audit["warnings"].append(
        "CSV annotation columns contain no frozen-model probability; tumor_ratio is ground truth, not y_score.")

    summary = d.groupby(["patient_id", "source_split"], as_index=False).agg(
        patches=("label", "size"), positives=("label", "sum"),
        slides=("slide_name", "nunique"), source_images=("image_path", "nunique"))
    summary["negatives"] = summary.patches - summary.positives
    summary["positive_rate"] = summary.positives / summary.patches
    summary.to_csv(out / "patient_summary.csv", index=False)

    cv_data = d if args.include_test_in_cv else d[d.source_split.isin(["train", "val"])].copy()
    cv_summary = summary if args.include_test_in_cv else summary[summary.source_split.isin(["train", "val"])].copy()
    # Greedy multi-objective bin packing balances patients, positive patches and
    # negative patches while keeping every patient wholly within one fold.
    if args.folds > len(cv_summary):
        raise ValueError("fold count exceeds patient count")
    rng = np.random.default_rng(args.seed)
    s = cv_summary.sample(frac=1, random_state=args.seed).copy()
    s["imbalance"] = np.maximum(s.positives / max(1, s.positives.sum()),
                                s.negatives / max(1, s.negatives.sum()))
    s = s.sort_values(["imbalance", "patches"], ascending=False)
    totals = np.zeros((args.folds, 3), dtype=float)  # patients, positives, negatives
    target = np.array([len(s), s.positives.sum(), s.negatives.sum()], dtype=float) / args.folds
    assignments = []
    for _, row in s.iterrows():
        costs = []
        item = np.array([1, row.positives, row.negatives], dtype=float)
        for f in range(args.folds):
            trial = totals.copy(); trial[f] += item
            costs.append(np.sum(((trial - target) / np.maximum(target, 1)) ** 2))
        best_cost = min(costs)
        ties = [i for i, c in enumerate(costs) if np.isclose(c, best_cost)]
        fold = int(rng.choice(ties))
        totals[fold] += item
        assignments.append((row.patient_id, fold + 1))
    folds = pd.DataFrame(assignments, columns=["patient_id", "fold"])
    manifest = cv_data.merge(folds, on="patient_id", validate="many_to_one")
    manifest[["uuid", "patient_id", "slide_name", "image_path", "x", "y", "label", "fold"]].to_csv(
        out / "cv_patch_manifest.csv", index=False)
    folds.merge(cv_summary, on="patient_id", validate="one_to_one").sort_values(
        ["fold", "patient_id"]).to_csv(out / "cv_patient_folds.csv", index=False)
    fold_check = folds.merge(cv_summary, on="patient_id").groupby("fold", as_index=False).agg(
        patients=("patient_id", "nunique"), patches=("patches", "sum"),
        positives=("positives", "sum"), negatives=("negatives", "sum"))
    fold_check["positive_rate"] = fold_check.positives / fold_check.patches
    fold_check.to_csv(out / "cv_fold_summary.csv", index=False)
    # Dependency-light PNG chart (Pillow only).
    img = Image.new("RGB", (1500, 650), "white")
    dr = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=22)
    title_font = ImageFont.load_default(size=30)
    dr.text((55, 25), "Development-set patient-level 5-fold balance", fill="black", font=title_font)
    left, bottom, top, width = 90, 565, 100, 1300
    dr.line((left, top, left, bottom), fill="black", width=2)
    dr.line((left, bottom, left + width, bottom), fill="black", width=2)
    max_patches = float(fold_check.patches.max())
    bar_w, gap = 155, 90
    for i, row in fold_check.iterrows():
        x0 = left + 90 + i * (bar_w + gap)
        neg_h = int((row.negatives / max_patches) * (bottom - top - 35))
        pos_h = int((row.positives / max_patches) * (bottom - top - 35))
        dr.rectangle((x0, bottom-neg_h, x0+bar_w, bottom), fill="#4C78A8")
        dr.rectangle((x0, bottom-neg_h-pos_h, x0+bar_w, bottom-neg_h), fill="#E45756")
        dr.text((x0+55, bottom+12), str(int(row.fold)), fill="black", font=font)
        dr.text((x0+18, bottom-neg_h-pos_h-28), f"{int(row.patches)} patches", fill="black", font=font)
        dr.text((x0+30, bottom-neg_h-pos_h+8), f"n={int(row.patients)} pts", fill="white", font=font)
    dr.rectangle((1110, 35, 1135, 60), fill="#4C78A8"); dr.text((1145, 35), "Label 0", fill="black", font=font)
    dr.rectangle((1250, 35, 1275, 60), fill="#E45756"); dr.text((1285, 35), "Label 1", fill="black", font=font)
    img.save(out / "cv_fold_balance.png", dpi=(300, 300))
    audit["cv_note"] = (
        "These files define folds only. They are not cross-validation performance results; "
        "each fold still requires fresh model training and out-of-fold prediction. "
        + ("The original test set was included by explicit request."
           if args.include_test_in_cv else
           "The 4 original test patients remain excluded and sealed."))
    (out / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
