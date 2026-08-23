from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import load_config, seed_everything
from .data import PatchDataset
from .metrics import threshold_youden
from .model import build_model


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); ys=[]; ss=[]; ids=[]
    for x,y,i in loader:
        x=x.to(device); prob=model(x).softmax(1)[:,1].cpu().numpy()
        ys.extend(y.numpy()); ss.extend(prob); ids.extend(i.numpy())
    return np.asarray(ys),np.asarray(ss),np.asarray(ids)


def fit(cfg, train_frame=None, val_frame=None, run_dir=None):
    seed_everything(int(cfg["seed"])); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr=PatchDataset(train_frame if train_frame is not None else cfg["train_csv"],cfg["patch_size"],True,cfg.get("image_root"))
    va=PatchDataset(val_frame if val_frame is not None else cfg["val_csv"],cfg["patch_size"],False,cfg.get("image_root"))
    tl=DataLoader(tr,cfg["batch_size"],shuffle=True,num_workers=cfg["workers"],pin_memory=True)
    vl=DataLoader(va,cfg["batch_size"],shuffle=False,num_workers=cfg["workers"],pin_memory=True)
    model=build_model(cfg).to(device); opt=AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    loss_fn=nn.CrossEntropyLoss(); best=-1.; stale=0; out=Path(run_dir or cfg["run_dir"]); out.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,int(cfg["epochs"])+1):
        model.train()
        for x,y,_ in tqdm(tl,desc=f"epoch {epoch}"):
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(x),y); loss.backward(); opt.step()
        y,s,_=predict(model,vl,device); auc=roc_auc_score(y,s)
        if auc>best:
            best=auc; stale=0; threshold=threshold_youden(y,s)
            torch.save({"model":model.state_dict(),"config":cfg,"epoch":epoch,
                        "val_auc":float(auc),"threshold":threshold},out/"best.pt")
        else: stale+=1
        if stale>=int(cfg["patience"]): break
    return out/"best.pt"


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); a=ap.parse_args()
    fit(load_config(a.config))


if __name__=="__main__": main()

