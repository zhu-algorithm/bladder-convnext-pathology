from __future__ import annotations

import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader

from .common import load_config, save_json
from .data import PatchDataset
from .metrics import classification_metrics, cluster_bootstrap_auc
from .model import build_model
from .train import predict


def evaluate(cfg, checkpoint, csv_path, outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(checkpoint,map_location=device,weights_only=False); model=build_model(cfg).to(device); model.load_state_dict(ck["model"])
    ds=PatchDataset(csv_path,cfg["patch_size"],False,cfg.get("image_root")); dl=DataLoader(ds,batch_size=cfg["batch_size"],shuffle=False,num_workers=cfg["workers"])
    y,s,idx=predict(model,dl,device); d=ds.df.iloc[idx].copy(); d["y_score"]=s
    threshold=float(ck["threshold"]); d["y_pred"]=(s>=threshold).astype(int)
    m=classification_metrics(y,s,threshold); m["n_patients"]=int(d.patient_id.nunique())
    m["auc_ci_95"]=cluster_bootstrap_auc(d,int(cfg["bootstrap_replicates"]),int(cfg["bootstrap_seed"]))
    save_json(m,out/"test_metrics.json"); d.to_csv(out/"test_predictions.csv",index=False)
    fpr,tpr,_=roc_curve(y,s); pre,rec,_=precision_recall_curve(y,s)
    fig,ax=plt.subplots(1,2,figsize=(10,4)); ax[0].plot(fpr,tpr,label=f"AUC={m['roc_auc']:.3f}"); ax[0].plot([0,1],[0,1],'--',c='.6'); ax[0].set(xlabel="FPR",ylabel="TPR",title="Independent test ROC"); ax[0].legend()
    ax[1].plot(rec,pre,label=f"AP={m['average_precision']:.3f}"); ax[1].axhline(y.mean(),ls='--',c='.6'); ax[1].set(xlabel="Recall",ylabel="Precision",title="Independent test PR"); ax[1].legend(); fig.tight_layout(); fig.savefig(out/"roc_pr.png",dpi=300); plt.close(fig)
    cm=confusion_matrix(y,d.y_pred,labels=[0,1]); fig,ax=plt.subplots(figsize=(4.5,4)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax); ax.set(xlabel="Predicted",ylabel="True",title=f"Threshold={threshold:.3f}"); fig.tight_layout(); fig.savefig(out/"confusion_matrix.png",dpi=300); plt.close(fig)
    d["error_type"]=np.select([(d.label==1)&(d.y_pred==1),(d.label==0)&(d.y_pred==0),(d.label==0)&(d.y_pred==1),(d.label==1)&(d.y_pred==0)],["TP","TN","FP","FN"])
    d["priority"]=np.where(d.error_type.isin(["TP","FP"]),d.y_score,1-d.y_score)
    d.sort_values("priority",ascending=False).groupby("error_type").head(10).to_csv(out/"tp_tn_fp_fn_candidates.csv",index=False)
    return m


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--checkpoint",required=True); ap.add_argument("--csv",required=True); ap.add_argument("--outdir",required=True); a=ap.parse_args()
    evaluate(load_config(a.config),a.checkpoint,a.csv,a.outdir)


if __name__=="__main__": main()
