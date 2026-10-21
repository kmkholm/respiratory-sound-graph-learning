"""Same-protocol linear baseline: logistic regression on the identical fused features and the identical
patient-level splits used by the GATFormer runs. Writes results/baseline_lr.json."""
import os, sys, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from run import load_corpus, split_patients, metrics

out = {}
def run(name, vec, meta, names, label_col, tr, te):
    y = meta[label_col].map({n: i for i, n in enumerate(names)}).values.astype(int)
    sc = StandardScaler().fit(vec[tr]); clf = LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced").fit(sc.transform(vec[tr]), y[tr])
    p = clf.predict_proba(sc.transform(vec[te])); seg = metrics(y[te], p, len(names), names)
    rp = pd.DataFrame(p).groupby(meta.path.values[te]).mean(); ry = meta.iloc[te].groupby("path")[label_col].first().loc[rp.index].map({n: i for i, n in enumerate(names)}).values
    recm = metrics(ry, rp.values, len(names), names)
    out[name] = dict(segment={k: seg[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")}, recording={k: recm[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")})
    print(name, "seg acc %.4f mF1 %.4f | rec acc %.4f mF1 %.4f" % (seg["accuracy"], seg["macro_f1"], recm["accuracy"], recm["macro_f1"]), flush=True)

for seed in range(5):
    for corpus in ["ASTHMA_V2", "KAUH", "ICBHI"]:
        _, vec, meta = load_corpus(corpus); names = [n for n in FIVE[corpus] if n in set(meta.label5)]
        keep = meta.label5.isin(names).values; vec = vec[keep]; meta = meta[keep].reset_index(drop=True)
        tr, va, te = split_patients(meta, "label5", seed); run(f"percorpus:{corpus}_s{seed}", vec, meta, names, "label5", tr, te)
    parts = [load_corpus(c) for c in ["ASTHMA_V2", "KAUH", "ICBHI"]]
    meta = pd.concat([p[2] for p in parts], ignore_index=True); keep = meta.label4.notna().values
    vec = np.concatenate([p[1] for p in parts])[keep]; meta = meta[keep].reset_index(drop=True)
    tr, va, te = split_patients(meta, "label4", seed); run(f"merged_s{seed}", vec, meta, MERGED4, "label4", tr, te)
json.dump(out, open(os.path.join(RES, "baseline_lr.json"), "w"), indent=1)
