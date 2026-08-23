from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from .common import load_config, save_json
from .data import PatchDataset
from .metrics import classification_metrics
from .model import build_model
from .train import fit, predict


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--folds",type=int,default=5); a=ap.parse_args(); cfg=load_config(a.config)
    # Development data only: preserve the original test cohort.
    dev=pd.concat([pd.read_csv(cfg["train_csv"]),pd.read_csv(cfg["val_csv"])],ignore_index=True)
    patient=dev.groupby("patient_id").label.mean(); patient_class=(patient>=.5).astype(int)
    patients=patient.index.to_numpy(); splitter=StratifiedGroupKFold(a.folds,shuffle=True,random_state=int(cfg["seed"]))
    out=Path(cfg["run_dir"]).parent/"cross_validation"; out.mkdir(parents=True,exist_ok=True); all_pred=[]; fold_metrics=[]
    dummy=np.zeros(len(patients))
    for fold,(trp,vap) in enumerate(splitter.split(dummy,patient_class.to_numpy(),groups=patients),1):
        train_pat=set(patients[trp]); val_pat=set(patients[vap]); tr=dev[dev.patient_id.isin(train_pat)].copy(); va=dev[dev.patient_id.isin(val_pat)].copy()
        fold_cfg=dict(cfg); fold_cfg["seed"]=int(cfg["seed"])+fold; run=out/f"fold_{fold}"; ck_path=fit(fold_cfg,tr,va,run)
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); ck=torch.load(ck_path,map_location=device,weights_only=False); model=build_model(fold_cfg).to(device); model.load_state_dict(ck["model"])
        ds=PatchDataset(va,fold_cfg["patch_size"],False,fold_cfg.get("image_root")); dl=DataLoader(ds,batch_size=fold_cfg["batch_size"],shuffle=False,num_workers=fold_cfg["workers"])
        y,s,idx=predict(model,dl,device); p=va.iloc[idx].copy(); p["y_score"]=s; p["fold"]=fold; all_pred.append(p)
        m=classification_metrics(y,s,float(ck["threshold"])); m.update({"fold":fold,"n_patients":len(val_pat)}); fold_metrics.append(m)
    pred=pd.concat(all_pred,ignore_index=True); pred.to_csv(out/"oof_predictions.csv",index=False); pd.DataFrame(fold_metrics).to_csv(out/"fold_metrics.csv",index=False)
    summary={k:{"mean":float(np.nanmean([m[k] for m in fold_metrics])),"sd":float(np.nanstd([m[k] for m in fold_metrics],ddof=1))} for k in ["roc_auc","average_precision","sensitivity","specificity","ppv","npv","f1","accuracy"]}
    summary["patients"]=int(pred.patient_id.nunique()); summary["note"]="Five fresh models; original independent test patients excluded."
    save_json(summary,out/"cv_summary.json")


if __name__=="__main__": main()

