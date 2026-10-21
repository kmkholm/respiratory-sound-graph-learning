"""Quantitative explainability analysis for a trained run.

1. SHAP (GradientExplainer) over the frame-level input -> attribution per frequency bin per class;
   band-enrichment ratio = share of |SHAP| mass inside the clinically expected band / share of bins in that band
   (ratio 1 = no preference), with bootstrap CI over test segments.
2. Transformer attention-pooling weights vs ICBHI expert annotations: enrichment of attention mass inside
   annotated wheeze/crackle frames relative to their time fraction.
3. Class-level acoustic signature maps: attention-weighted mean log-Mel spectrogram per class (correct predictions).

usage: python xai.py <run_id>
"""
import os, sys, json, glob, numpy as np, pandas as pd, torch, librosa
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from run import load_corpus, frame_mask, vec_mask, seq_std_np, DEV, RUNS, BLOCKS
from model import GATFormer
import shap

EXPECTED_BAND = {"Asthma": (400, 1000), "COPD": (200, 350), "Pneumonia": (300, 800), "Healthy": (100, 300), "Bronchial": (300, 800)}
MEL_F = librosa.mel_frequencies(n_mels=128 + 2, fmin=100, fmax=2200)[1:-1]     # centre frequency of each Mel bin


class Wrapped(torch.nn.Module):
    """model with fixed anchors so that SHAP sees (vec, seq) only"""
    def __init__(self, m, anchors): super().__init__(); self.m, self.anchors = m, anchors
    def forward(self, vec, seq): return self.m(vec, seq, self.anchors, None)[0]


def icbhi_event_frames(path, t0, T=101, seg=2.0):
    """per-frame flags (crackle, wheeze) for a 2 s window starting at t0 from the ICBHI txt annotation."""
    txt = path.replace(".wav", ".txt")
    cr = np.zeros(T, bool); wh = np.zeros(T, bool)
    if not os.path.exists(txt): return None
    ann = np.loadtxt(txt, ndmin=2)
    t = t0 + np.arange(T) * (seg / (T - 1))
    for s, e, c, w in ann:
        m = (t >= s) & (t < e)
        if c: cr |= m
        if w: wh |= m
    return cr, wh


def main(run_id, n_shap=256, n_bg=64):
    ck = torch.load(os.path.join(RUNS, run_id + ".pt"), weights_only=False); cfg = ck["cfg"]; names = ck["classes"]; C = len(names)
    z = np.load(os.path.join(RUNS, run_id + ".npz"), allow_pickle=True)
    exp = cfg["exp"]
    if exp.startswith("percorpus:"):
        corpora = [exp.split(":")[1]]
    else:
        corpora = ["ASTHMA_V2", "KAUH", "ICBHI"]
    parts = [load_corpus(c) for c in corpora]
    meta = pd.concat([p[2] for p in parts], ignore_index=True)
    if cfg["label_col"] == "label4":
        keep = meta.label4.notna().values
    else:
        keep = meta.label5.isin(names).values
    seq = np.concatenate([np.asarray(p[0]) for p in parts])[keep]; vec = np.concatenate([p[1] for p in parts])[keep]; meta = meta[keep].reset_index(drop=True)
    fm, vm = ck["fm"], vec_mask(cfg["blocks"])
    V = ((vec[:, vm] - ck["scaler_mean"]) / ck["scaler_scale"]).astype(np.float32)[:, ck["mask"]]
    S = seq_std_np(seq, fm, ck["seq_mean"], ck["seq_std"])
    model = GATFormer(V.shape[1], int(fm.sum()), C, d=cfg["d"], tf_layers=cfg["tf_layers"], use_gat=cfg["use_gat"], use_tf=cfg["use_tf"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    anchors = torch.as_tensor(ck["anchors"]).to(DEV)
    te = z["idx"]; y = z["y"]; prob = z["prob"]; pred = prob.argmax(1)
    rng = np.random.RandomState(0)
    out = dict(run_id=run_id, classes=names)

    # ---------------- 1. SHAP per frequency bin
    correct = te[pred == y]; ycor = y[pred == y]
    sel = np.concatenate([rng.choice(np.where(ycor == c)[0], min(n_shap // C, (ycor == c).sum()), replace=False) for c in range(C) if (ycor == c).sum() > 0])
    idx = correct[sel]; ys = ycor[sel]
    tr_pool = np.setdiff1d(np.arange(len(meta)), te); bg = rng.choice(tr_pool, n_bg, replace=False)
    wrapped = Wrapped(model, anchors).to(DEV)
    bg_in = [torch.as_tensor(V[bg]).to(DEV), torch.as_tensor(S[bg]).to(DEV)]
    explainer = shap.GradientExplainer(wrapped, bg_in)
    sv_vec, sv_seq = [], []
    for i in range(0, len(idx), 32):
        b = idx[i:i + 32]
        inp = [torch.as_tensor(V[b]).to(DEV), torch.as_tensor(S[b]).to(DEV)]
        sv = explainer.shap_values(inp)                         # list over inputs, each (B, ..., C)
        sv_vec.append(np.asarray(sv[0])); sv_seq.append(np.asarray(sv[1]))
    sv_seq = np.concatenate(sv_seq)                              # (n, T, Dframe, C)
    frame_idx = np.where(fm)[0]
    mel_cols = np.where((frame_idx >= BLOCKS["mel"][0]) & (frame_idx < BLOCKS["mel"][1]))[0]
    band = {}
    boots = {}
    for c, name in enumerate(names):
        m = ys == c
        if m.sum() == 0: continue
        a = np.abs(sv_seq[m][:, :, mel_cols, c]).sum(1)          # (n_c, 128) |SHAP| per Mel bin for the true class
        prof = a / (a.sum(1, keepdims=True) + 1e-12)
        lo, hi = EXPECTED_BAND.get(name, (100, 2000)); inband = (MEL_F >= lo) & (MEL_F < hi)
        share = prof[:, inband].sum(1); base = inband.mean()
        ratio = share / base
        bs = [np.mean(rng.choice(ratio, len(ratio))) for _ in range(1000)]
        band[name] = dict(expected_band_hz=[lo, hi], n=int(m.sum()), attribution_share=float(share.mean()), bin_share=float(base),
                          enrichment=float(ratio.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                          peak_hz=float(MEL_F[prof.mean(0).argmax()]), profile=prof.mean(0).tolist())
    # block-level attribution (which representation matters per class)
    blk = {}
    for bname, (a0, a1) in BLOCKS.items():
        cols = np.where((frame_idx >= a0) & (frame_idx < a1))[0]
        if len(cols) == 0: continue
        blk[bname] = {names[c]: float(np.abs(sv_seq[ys == c][:, :, cols, c]).sum((1, 2)).mean() / max(1, len(cols))) for c in range(C) if (ys == c).sum() > 0}
    out["shap_band"] = band; out["shap_block_per_feature"] = blk; out["mel_centre_hz"] = MEL_F.tolist()

    # ---------------- 2. attention vs ICBHI annotations
    att = z["tf_att"]
    if att.size and "ICBHI" in corpora:
        paths = z["path"]; t0s = z["t0"]
        enr = []; frac = []; cls_enr = {}
        for k in range(len(te)):
            p = paths[k]
            if "audio_and_txt_files" not in p: continue
            ev = icbhi_event_frames(p, float(t0s[k]), T=att.shape[1])
            if ev is None: continue
            adv = ev[0] | ev[1]
            if adv.sum() == 0 or adv.sum() == len(adv): continue
            a = att[k] / att[k].sum()
            e = a[adv].sum() / adv.mean(); enr.append(e); frac.append(adv.mean())
            cls_enr.setdefault(names[y[k]], []).append(e)
        enr = np.array(enr)
        bs = [np.mean(rng.choice(enr, len(enr))) for _ in range(1000)]
        out["attention_vs_annotation"] = dict(n_segments=int(len(enr)), mean_enrichment=float(enr.mean()), median=float(np.median(enr)),
                                              ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                              share_above_1=float((enr > 1).mean()), mean_event_fraction=float(np.mean(frac)),
                                              per_class={k: dict(n=len(v), mean=float(np.mean(v))) for k, v in cls_enr.items()})

    # ---------------- 3. class signature maps (attention-weighted mean log-Mel of correct predictions)
    sig = {}
    mel_a, mel_z = BLOCKS["mel"]
    for c, name in enumerate(names):
        ids = te[(pred == y) & (y == c)]
        if len(ids) == 0: continue
        ids = rng.choice(ids, min(300, len(ids)), replace=False)
        M = np.asarray(seq[ids][:, :, mel_a:mel_z], dtype=np.float32)       # (n, T, 128)
        if att.size:
            w = att[np.isin(te, ids)]; w = w / w.sum(1, keepdims=True)
            sig[name] = (M * w[:, :, None] * M.shape[1]).mean(0).T.tolist()  # (128, T)
        else:
            sig[name] = M.mean(0).T.tolist()
    np.savez_compressed(os.path.join(RUNS, run_id + "_xai.npz"), **{f"sig_{k}": np.array(v) for k, v in sig.items()},
                        shap_profile=np.array([band[n]["profile"] for n in names if n in band]), mel_f=MEL_F)
    json.dump(out, open(os.path.join(RUNS, run_id + "_xai.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k in ("shap_band", "attention_vs_annotation", "shap_block_per_feature")}, indent=1, default=lambda o: o if not isinstance(o, list) or len(o) < 10 else "[...]"))


if __name__ == "__main__":
    main(sys.argv[1])
