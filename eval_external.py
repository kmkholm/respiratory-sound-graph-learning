"""Evaluate a trained run (merged 4-class model) on an external corpus never used in training, feature selection
or model selection. Default: RespiratoryDatabase@TR (COPD0 -> Healthy, COPD1-4 -> COPD).

usage: python eval_external.py <run_id> [corpus=TR]
Writes results/runs/<run_id>_ext_<corpus>.json (segment-, recording- and patient-level metrics; severity breakdown).
"""
import os, sys, json, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from run import load_corpus, vec_mask, seq_std_np, evaluate, metrics, DEV, RUNS
from model import GATFormer


def main(run_id, corpus="TR"):
    ck = torch.load(os.path.join(RUNS, run_id + ".pt"), weights_only=False); cfg = ck["cfg"]; names = list(ck["classes"]); C = len(names)
    seq, vec, meta = load_corpus(corpus)
    man = pd.read_csv(os.path.join(DATA, "manifest.csv")); sev = dict(zip(man.path, man.get("severity", pd.Series([""] * len(man)))))
    lab_col = cfg["label_col"]; y = meta[lab_col if lab_col in meta else "label5"].map({n: i for i, n in enumerate(names)})
    keep = y.notna().values; meta = meta[keep].reset_index(drop=True); y = y[keep].astype(int).values; vec = vec[keep]; seq = np.asarray(seq)[keep]
    fm, vm = ck["fm"], vec_mask(cfg["blocks"])
    if cfg.get("corpus_norm"):
        # the external corpus is z-scored with its own (label-free) statistics, as in training
        mu, sd = vec[:, vm].mean(0), vec[:, vm].std(0) + 1e-6; V = ((vec[:, vm] - mu) / sd).astype(np.float32)[:, ck["mask"]]
        sq = np.asarray(seq[:, :, fm], dtype=np.float32); S = seq_std_np(seq, fm, sq.mean((0, 1)), sq.std((0, 1)) + 1e-6)
    else:
        V = ((vec[:, vm] - ck["scaler_mean"]) / ck["scaler_scale"]).astype(np.float32)[:, ck["mask"]]
        S = seq_std_np(seq, fm, ck["seq_mean"], ck["seq_std"])
    model = GATFormer(V.shape[1], int(fm.sum()), C, d=cfg["d"], tf_layers=cfg["tf_layers"], use_gat=cfg["use_gat"], use_tf=cfg["use_tf"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    anchors = (torch.as_tensor(ck["anchors"]).to(DEV), None)
    p, z, att, gate = evaluate(model, S, V, np.arange(len(y)), anchors)
    seg = metrics(y, p, C, names)
    rec = pd.DataFrame(p).groupby(meta.path.values).mean(); ry = meta.groupby("path")[lab_col if lab_col in meta else "label5"].first().loc[rec.index].map({n: i for i, n in enumerate(names)}).values
    recm = metrics(ry, rec.values, C, names)
    pat = pd.DataFrame(p).groupby(meta.patient.values).mean(); py = meta.groupby("patient")[lab_col if lab_col in meta else "label5"].first().loc[pat.index].map({n: i for i, n in enumerate(names)}).values
    patm = metrics(py, pat.values, C, names)
    out = dict(run_id=run_id, corpus=corpus, classes=names, n_segments=int(len(y)), n_recordings=int(len(ry)), n_patients=int(len(py)),
               segment=seg, recording=recm, patient=patm)
    if corpus == "TR":
        sv = meta.path.map(sev).values; pred = p.argmax(1)
        out["severity_breakdown"] = {s: dict(n_segments=int((sv == s).sum()), copd_rate=float((pred[sv == s] == names.index("COPD")).mean()),
                                               healthy_rate=float((pred[sv == s] == names.index("Healthy")).mean()),
                                               pred_distribution={names[c]: int((pred[sv == s] == c).sum()) for c in range(C)}) for s in sorted(set(sv))}
    json.dump(out, open(os.path.join(RUNS, f"{run_id}_ext_{corpus}.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("n_segments", "n_recordings", "n_patients")}),
          "\nsegment:", {k: round(seg[k], 4) if isinstance(seg[k], float) else seg[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")},
          "\nrecording:", {k: round(recm[k], 4) if isinstance(recm[k], float) else recm[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")},
          "\npatient:", {k: round(patm[k], 4) if isinstance(patm[k], float) else patm[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")},
          "\nseverity:", json.dumps(out.get("severity_breakdown", {}), indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "TR")
