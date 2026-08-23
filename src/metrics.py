from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, roc_auc_score, roc_curve)


def classification_metrics(y, score, threshold):
    y=np.asarray(y); score=np.asarray(score); pred=(score>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    div=lambda a,b: float(a/b) if b else float("nan")
    return {"n_patches":int(len(y)),"roc_auc":float(roc_auc_score(y,score)),
            "average_precision":float(average_precision_score(y,score)),
            "threshold":float(threshold),"accuracy":float(accuracy_score(y,pred)),
            "f1":float(f1_score(y,pred,zero_division=0)),"sensitivity":div(tp,tp+fn),
            "specificity":div(tn,tn+fp),"ppv":div(tp,tp+fp),"npv":div(tn,tn+fn),
            "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}


def threshold_youden(y, score):
    fpr,tpr,thr=roc_curve(y,score)
    valid=np.isfinite(thr)
    return float(thr[valid][np.argmax((tpr-fpr)[valid])])


def cluster_bootstrap_auc(df, n=10000, seed=20260823):
    rng=np.random.default_rng(seed); groups={p:g for p,g in df.groupby("patient_id")}
    patients=np.array(list(groups),dtype=object); vals=[]; rejected=0
    for _ in range(n):
        b=pd.concat([groups[p] for p in rng.choice(patients,len(patients),replace=True)])
        if b.label.nunique()<2: rejected+=1; continue
        vals.append(roc_auc_score(b.label,b.y_score))
    if not vals: raise ValueError("No valid bootstrap replicate")
    lo,hi=np.quantile(vals,[.025,.975])
    return {"method":"patient-cluster percentile bootstrap of patch-level AUC",
            "n_requested":n,"n_valid":len(vals),"n_rejected_single_class":rejected,
            "seed":seed,"low":float(lo),"high":float(hi)}

