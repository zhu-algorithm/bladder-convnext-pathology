#!/usr/bin/env python3
"""Auditable evaluation for the bladder ConvNeXt project.

The script never substitutes annotation-derived columns (for example
``tumor_ratio``) for model predictions.  A prediction file must contain a
numeric score produced by a frozen model for every evaluated patch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

KEY = ["uuid", "x", "y"]


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def roc_points(y: np.ndarray, score: np.ndarray):
    order = np.argsort(-score, kind="stable")
    yy = y[order]
    # Keep the final cumulative count for every tied score threshold.
    distinct = np.r_[score[order][:-1] != score[order][1:], True]
    tp = np.cumsum(yy)[distinct]
    fp = np.cumsum(1 - yy)[distinct]
    return np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]]


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr = roc_points(y, score)
    return float(np.trapezoid(tpr, fpr))


def pr_points(y: np.ndarray, score: np.ndarray):
    order = np.argsort(-score, kind="stable")
    yy = y[order]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    recall = tp / tp[-1]
    precision = tp / (tp + fp)
    return np.r_[0, recall], np.r_[1, precision]


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    recall, precision = pr_points(y, score)
    return float(np.sum(np.diff(recall) * precision[1:]))


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
    return {
        "n_patches": int(len(y)),
        "roc_auc": roc_auc(y, score),
        "average_precision": average_precision(y, score),
        "threshold": float(threshold),
        "accuracy": safe_div(tp + tn, len(y)),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "ppv": safe_div(tp, tp + fp),
        "npv": safe_div(tn, tn + fn),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def clustered_auc_ci(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Percentile CI after resampling patients, retaining their patches."""
    rng = np.random.default_rng(seed)
    groups = {p: g for p, g in df.groupby("patient_id", sort=False)}
    patients = np.array(list(groups), dtype=object)
    values = []
    rejected = 0
    for _ in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        boot = pd.concat([groups[p] for p in sampled], ignore_index=True)
        if boot["label"].nunique() < 2:
            rejected += 1
            continue
        values.append(roc_auc(boot["label"].to_numpy(), boot["y_score"].to_numpy()))
    if not values:
        raise ValueError("No valid bootstrap replicate contained both classes")
    lo, hi = np.quantile(values, [0.025, 0.975])
    return {
        "method": "patient-cluster percentile bootstrap of patch-level AUC",
        "n_requested": n_boot,
        "n_valid": len(values),
        "n_rejected_single_class": rejected,
        "seed": seed,
        "auc_ci_95_low": float(lo),
        "auc_ci_95_high": float(hi),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="test CSV with label and patient_id")
    ap.add_argument("--predictions", required=True,
                    help="CSV with uuid,x,y,y_score; optional image_path")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Must be fixed without looking at the test outcomes")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260823)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(args.labels)
    pred = pd.read_csv(args.predictions)
    required_labels = set(KEY + ["patient_id", "label"])
    required_pred = set(KEY + ["y_score"])
    if missing := required_labels - set(labels):
        raise ValueError(f"label CSV missing columns: {sorted(missing)}")
    if missing := required_pred - set(pred):
        raise ValueError(f"prediction CSV missing columns: {sorted(missing)}")
    if labels.duplicated(KEY).any() or pred.duplicated(KEY).any():
        raise ValueError("uuid,x,y must uniquely identify every patch")
    d = labels.merge(pred, on=KEY, how="left", validate="one_to_one", suffixes=("", "_pred"))
    if d["y_score"].isna().any():
        raise ValueError(f"predictions missing for {int(d.y_score.isna().sum())} test patches")
    if len(pred) != len(d):
        raise ValueError("prediction file contains rows not present in the label CSV")
    if not d["y_score"].between(0, 1).all():
        raise ValueError("y_score must be a probability in [0,1]")
    if d["label"].nunique() != 2:
        raise ValueError("both classes are required")

    result = metrics(d.label.to_numpy(), d.y_score.to_numpy(), args.threshold)
    result["n_patients"] = int(d.patient_id.nunique())
    result["ci"] = clustered_auc_ci(d, args.bootstrap, args.seed)
    (out / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fpr, tpr = roc_points(d.label.to_numpy(), d.y_score.to_numpy())
    recall, precision = pr_points(d.label.to_numpy(), d.y_score.to_numpy())
    img = Image.new("RGB", (1800, 780), "white"); dr = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=24); title = ImageFont.load_default(size=32)
    def panel(x0, xs, ys, heading, xlab, ylab, note, diagonal=False, baseline=None):
        l, t, w, h = x0+95, 120, 650, 520
        dr.rectangle((l, t, l+w, t+h), outline="black", width=2)
        pts = [(l+int(float(x)*w), t+h-int(float(y)*h)) for x,y in zip(xs,ys)]
        if diagonal: dr.line((l,t+h,l+w,t), fill="#AAAAAA", width=2)
        if baseline is not None: dr.line((l,t+h-int(baseline*h),l+w,t+h-int(baseline*h)), fill="#AAAAAA", width=2)
        if len(pts)>1: dr.line(pts, fill="#2F6B9A", width=5, joint="curve")
        dr.text((l+80, 55), heading, fill="black", font=title)
        dr.text((l+390, t+20), note, fill="#2F6B9A", font=font)
        dr.text((l+230, t+h+50), xlab, fill="black", font=font)
        dr.text((l-80, t-35), ylab, fill="black", font=font)
        for q in [0, .25, .5, .75, 1]:
            dr.text((l+int(q*w)-12,t+h+8),f"{q:g}",fill="black",font=font)
            dr.text((l-48,t+h-int(q*h)-12),f"{q:g}",fill="black",font=font)
    panel(0, fpr, tpr, "Independent test ROC", "False-positive rate", "TPR", f"AUC={result['roc_auc']:.3f}", True)
    panel(900, recall, precision, "Independent test PR", "Recall", "Precision", f"AP={result['average_precision']:.3f}", baseline=float(d.label.mean()))
    img.save(out / "test_roc_pr.png", dpi=(300,300))

    cm = np.array([[result["tn"], result["fp"]], [result["fn"], result["tp"]]])
    img = Image.new("RGB", (850, 750), "white"); dr = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=30); title = ImageFont.load_default(size=36)
    dr.text((95, 35), f"Confusion matrix (threshold={args.threshold:g})", fill="black", font=title)
    for i in range(2):
        for j in range(2):
            v=int(cm[i,j]); shade=int(245-155*v/max(1,int(cm.max())))
            dr.rectangle((230+j*240,160+i*240,470+j*240,400+i*240), fill=(shade,shade,245), outline="white", width=3)
            dr.text((325+j*240,255+i*240),str(v),fill="black",font=title)
    dr.text((300, 120), "Pred 0", fill="black", font=font); dr.text((540, 120), "Pred 1", fill="black", font=font)
    dr.text((80, 260), "True 0", fill="black", font=font); dr.text((80, 500), "True 1", fill="black", font=font)
    img.save(out / "test_confusion_matrix.png", dpi=(300,300))

    d["y_pred"] = (d.y_score >= args.threshold).astype(int)
    d["error_type"] = np.select(
        [(d.label == 1) & (d.y_pred == 1), (d.label == 0) & (d.y_pred == 0),
         (d.label == 0) & (d.y_pred == 1), (d.label == 1) & (d.y_pred == 0)],
        ["TP", "TN", "FP", "FN"], default="UNKNOWN")
    # The most confident correct and most confident wrong patches are candidates
    # only; a pathologist should approve final representative figures.
    d["selection_priority"] = np.select(
        [d.error_type.eq("TP"), d.error_type.eq("TN"), d.error_type.eq("FP"), d.error_type.eq("FN")],
        [d.y_score, 1-d.y_score, d.y_score, 1-d.y_score], default=np.nan)
    d.sort_values(["error_type", "selection_priority"], ascending=[True, False]).groupby(
        "error_type", sort=False).head(10).to_csv(out / "tp_tn_fp_fn_candidates.csv", index=False)


if __name__ == "__main__":
    main()
