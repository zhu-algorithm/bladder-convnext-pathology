from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch

from .common import load_config
from .data import PatchDataset
from .model import build_model


class GradCAM:
    def __init__(self, model, layer):
        self.model=model; self.a=None; self.g=None
        layer.register_forward_hook(lambda m,i,o:setattr(self,"a",o.detach()))
        layer.register_full_backward_hook(lambda m,gi,go:setattr(self,"g",go[0].detach()))
    def __call__(self,x,cls=1):
        self.model.zero_grad(set_to_none=True); self.model(x)[0,cls].backward()
        w=self.g.mean(dim=(2,3),keepdim=True); cam=(w*self.a).sum(1).relu()[0]
        cam=cam-cam.min(); cam=cam/(cam.max()+1e-8); return cam.cpu().numpy()


def overlay(original, cam, alpha=.45):
    base=np.asarray(original.convert("RGB")).astype(float); heat=Image.fromarray(np.uint8(cam*255)).resize(original.size)
    h=np.asarray(heat).astype(float)/255; color=np.stack([255*h,60*(1-h),40*(1-h)],axis=-1)
    return Image.fromarray(np.uint8(np.clip((1-alpha)*base+alpha*color,0,255)))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--checkpoint",required=True); ap.add_argument("--predictions",required=True); ap.add_argument("--outdir",required=True); ap.add_argument("--per-type",type=int,default=3); a=ap.parse_args(); cfg=load_config(a.config)
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True); d=pd.read_csv(a.predictions)
    if "error_type" not in d: d["error_type"]=np.select([(d.label==1)&(d.y_pred==1),(d.label==0)&(d.y_pred==0),(d.label==0)&(d.y_pred==1),(d.label==1)&(d.y_pred==0)],["TP","TN","FP","FN"])
    d["priority"]=np.where(d.error_type.isin(["TP","FP"]),d.y_score,1-d.y_score); chosen=d.sort_values("priority",ascending=False).groupby("error_type").head(a.per_type)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); ck=torch.load(a.checkpoint,map_location=device,weights_only=False); model=build_model(cfg).to(device); model.load_state_dict(ck["model"]); model.eval()
    # ConvNeXt final convolutional block in timm. If architecture changes, verify this layer.
    cammer=GradCAM(model,model.stages[-1].blocks[-1].conv_dw)
    for _,r in chosen.iterrows():
        one=pd.DataFrame([r]); ds=PatchDataset(one,cfg["patch_size"],False,cfg.get("image_root")); x,_,_=ds[0]; cam=cammer(x.unsqueeze(0).to(device))
        path=ds.resolve(r.image_path)
        with Image.open(path) as im: patch=im.convert("RGB").crop((int(r.x),int(r.y),int(r.x)+cfg["patch_size"],int(r.y)+cfg["patch_size"]))
        overlay(patch,cam).save(out/f"{r.error_type}_{r.patient_id}_{r.uuid}_{r.x}_{r.y}.png")
    chosen.to_csv(out/"selected_cases.csv",index=False)


if __name__=="__main__": main()
