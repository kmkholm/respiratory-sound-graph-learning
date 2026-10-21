"""Experiment runner. One run = (data split) -> standardize -> DRG -> GQDO -> GATFormer -> test metrics.

usage: python run.py <exp> [seed] [variant flags]
  exp in: percorpus:<CORPUS>  merged  lodo:<HELDOUT>  abl:<name>
Results -> results/runs/<run_id>.json (+ .npz with test probabilities / labels / patient ids)
"""
import os, sys, json, time, hashlib, numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
                             roc_auc_score, balanced_accuracy_score)
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from graph_gqdo import GQDO, disease_graph, envelope_modulation
from model import GATFormer, FocalLoss, PrototypeBank

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUNS = os.path.join(RES, "runs"); os.makedirs(RUNS, exist_ok=True)
GQ_CACHE = os.path.join(RES, "gqdo_cache"); os.makedirs(GQ_CACHE, exist_ok=True)
BLOCKS = json.load(open(os.path.join(FEAT, "blocks.json")))
D_FRAME = 228


# ----------------------------------------------------------------------------- data
def load_corpus(corpus):
    seq = np.load(os.path.join(FEAT, f"seq_{corpus}.npy"), mmap_mode="r")
    vec = np.load(os.path.join(FEAT, f"vec_{corpus}.npy"))
    meta = pd.read_csv(os.path.join(FEAT, f"meta_{corpus}.csv"))
    meta["corpus"] = corpus
    return seq, vec, meta


def frame_mask(blocks):
    """boolean mask over the 228 frame features for the chosen representation blocks."""
    m = np.zeros(D_FRAME, bool)
    for b in blocks:
        a, z = BLOCKS[b]; m[a:z] = True
    return m


def vec_mask(blocks):
    m = frame_mask(blocks); return np.concatenate([m, m])   # mean part + std part


def split_patients(meta, labels, seed, test_frac=0.2, val_frac=0.1):
    """patient-level split stratified by class at the PATIENT level: every patient's segments fall in exactly one
    partition; each class keeps >=1 patient in the test set and (when it has >=3 patients) in the validation set."""
    rng = np.random.RandomState(seed)
    pt = meta.groupby("patient")[labels].agg(lambda s: s.mode()[0])
    tr, va, te = [], [], []
    for c in sorted(pt.unique()):
        ps = rng.permutation(pt.index[pt == c].values); n = len(ps)
        nte = max(1, int(round(n * test_frac))) if n >= 2 else 0
        nva = max(1, int(round(n * val_frac))) if n - nte >= 2 else 0
        te += list(ps[:nte]); va += list(ps[nte:nte + nva]); tr += list(ps[nte + nva:])
    f = lambda S: np.where(meta.patient.isin(S))[0]
    return f(tr), f(va), f(te)


# ----------------------------------------------------------------------------- training
def make_batches(idx, y, bs, rng, sampler="sqrt"):
    """sampler: 'none' = plain shuffling; 'balanced' = equal class shares per batch;
    'sqrt' = sampling probability proportional to 1/sqrt(class frequency) (milder re-balancing)."""
    if sampler == "none":
        p = rng.permutation(idx); return [p[i:i + bs] for i in range(0, len(p), bs)]
    classes = np.unique(y[idx]); per = {c: idx[y[idx] == c] for c in classes}
    n_batches = int(np.ceil(len(idx) / bs)); out = []
    if sampler == "balanced":
        for _ in range(n_batches):
            b = np.concatenate([rng.choice(per[c], max(1, bs // len(classes)), replace=len(per[c]) < bs // len(classes)) for c in classes])
            out.append(rng.permutation(b)[:bs])
        return out
    freq = np.array([len(per[c]) for c in classes], float); w = 1 / np.sqrt(freq)
    share = w * freq / (w * freq).sum(); natural = freq / freq.sum()
    share = np.minimum(share, 10 * natural); share = share / share.sum()      # no class up-weighted beyond 10x its natural share
    p_i = np.zeros(len(idx))
    for k, c in enumerate(classes): p_i[y[idx] == c] = share[k] / freq[k]
    p_i /= p_i.sum()
    draw = rng.choice(idx, n_batches * bs, replace=True, p=p_i)
    return [draw[i:i + bs] for i in range(0, len(draw), bs)]


def augment(seq, vec, rng):
    """training-only feature-space augmentation: time roll, additive noise, time masking (minority-aware sampler already balances)."""
    B, T, D = seq.shape
    seq = seq.clone()
    for i in range(B):
        if rng.rand() < 0.5: seq[i] = torch.roll(seq[i], int(rng.randint(-10, 11)), 0)
        if rng.rand() < 0.5:
            t0 = rng.randint(0, T - 10); seq[i, t0:t0 + rng.randint(2, 10)] = 0
    seq = seq + 0.05 * torch.randn_like(seq); vec = vec + 0.05 * torch.randn_like(vec)
    return seq, vec


def evaluate(model, seq_t, vec_t, idx, anchors, bs=256):
    model.eval(); probs = []; zs = []; tf_att = []; gates = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            v = torch.as_tensor(vec_t[b]).to(DEV); s = torch.as_tensor(np.asarray(seq_t[b], dtype=np.float32)).to(DEV)
            logits, z, aux = model(v, s, anchors[0], anchors[1], return_aux=True)
            probs.append(torch.softmax(logits, 1).cpu().numpy()); zs.append(z.cpu().numpy())
            if "tf_att" in aux: tf_att.append(aux["tf_att"].cpu().numpy())
            if "gate" in aux: gates.append(aux["gate"].mean(1).cpu().numpy())
    return (np.concatenate(probs), np.concatenate(zs), np.concatenate(tf_att) if tf_att else None,
            np.concatenate(gates) if gates else None)


def metrics(y, p, C, names):
    pred = p.argmax(1)
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=range(C), zero_division=0)
    present = [c for c in range(C) if sup[c] > 0]
    out = dict(accuracy=float(accuracy_score(y, pred)),
               balanced_accuracy=float(balanced_accuracy_score(y, pred)),
               macro_f1=float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
               weighted_f1=float(f1_score(y, pred, labels=present, average="weighted", zero_division=0)),
               per_class={names[c]: dict(precision=float(prec[c]), recall=float(rec[c]), f1=float(f1[c]), support=int(sup[c])) for c in range(C)},
               confusion=confusion_matrix(y, pred, labels=range(C)).tolist(), n_test=int(len(y)))
    try:
        if len(present) == C: out["macro_auc"] = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
        else:
            yy = np.isin(np.arange(C), present); pp = p[:, yy] / p[:, yy].sum(1, keepdims=True)
            out["macro_auc"] = float(roc_auc_score(np.searchsorted(np.array(present), y), pp, multi_class="ovr", average="macro")) if len(present) > 2 else float(roc_auc_score(y == present[1], pp[:, 1]))
    except Exception as e:
        out["macro_auc"] = None
    return out


def train_eval(run_id, seq, vec, meta, tr, va, te, names, cfg):
    """seq/vec/meta: concatenated arrays over the corpora in play. tr/va/te: index arrays. names: class names."""
    t_start = time.time(); rng = np.random.RandomState(cfg["seed"]); torch.manual_seed(cfg["seed"])
    C = len(names); lab = {n: i for i, n in enumerate(names)}
    y = meta[cfg["label_col"]].map(lab).values.astype(int)
    fm, vm = frame_mask(cfg["blocks"]), vec_mask(cfg["blocks"])
    # standardize on training data
    sc_v = StandardScaler().fit(vec[tr][:, vm]); V = sc_v.transform(vec[:, vm]).astype(np.float32)
    g_mean = np.asarray(seq[tr][:, :, fm], dtype=np.float32).mean((0, 1)); g_std = np.asarray(seq[tr][:, :, fm], dtype=np.float32).std((0, 1)) + 1e-6
    N = len(vec); seq_tr_mean = np.broadcast_to(g_mean, (N, len(g_mean))).copy(); seq_tr_std = np.broadcast_to(g_std, (N, len(g_std))).copy()
    if cfg.get("corpus_norm") and "corpus" in meta.columns:
        # label-free corpus-wise standardisation: every corpus (including a held-out one) is z-scored with its own statistics
        for c in meta.corpus.unique():
            idx = np.where(meta.corpus.values == c)[0]
            mu, sd = vec[idx][:, vm].mean(0), vec[idx][:, vm].std(0) + 1e-6; V[idx] = ((vec[idx][:, vm] - mu) / sd).astype(np.float32)
            sq = np.asarray(seq[idx][:, :, fm], dtype=np.float32); seq_tr_mean[idx] = sq.mean((0, 1)); seq_tr_std[idx] = sq.std((0, 1)) + 1e-6
        print(f"[{run_id}] corpus-wise standardisation over {meta.corpus.nunique()} corpora", flush=True)
    # disease relationship graph from the training set
    mel_a, mel_z = BLOCKS["mel"]
    mel_prof = vec[:, mel_a:mel_z] if "mel" in cfg["blocks"] else None
    mod = envelope_modulation(np.asarray(seq[tr][:, :, mel_a:mel_z], dtype=np.float32)) if "mel" in cfg["blocks"] else None
    A_ref, S_ref = disease_graph(V[tr], y[tr], C, None if mel_prof is None else mel_prof[tr], mod)
    # feature selection
    if cfg["fs"] == "none":
        mask = np.ones(V.shape[1], bool); gq_hist = []
    else:
        key = hashlib.md5(f"{cfg['exp']}|{cfg['seed']}|{cfg['fs']}|{cfg['blocks']}|{cfg['label_col']}".encode()).hexdigest()[:10]
        cache = os.path.join(GQ_CACHE, key + ".npz")
        if os.path.exists(cache):
            z = np.load(cache); mask, gq_hist = z["mask"], z["hist"].tolist()
        else:
            gq = GQDO(variant=cfg["fs"], pop=30, iters=50, seed=cfg["seed"])
            mask, _ = gq.run(V[tr], y[tr], V[va], y[va], C, A_ref); gq_hist = gq.history
            np.savez(cache, mask=mask, hist=np.array(gq_hist))
    Vs = V[:, mask]
    print(f"[{run_id}] C={C} train={len(tr)} val={len(va)} test={len(te)} features {mask.sum()}/{len(mask)} ({time.time()-t_start:.0f}s)", flush=True)

    model = GATFormer(Vs.shape[1], int(fm.sum()), C, d=cfg["d"], tf_layers=cfg["tf_layers"], use_gat=cfg["use_gat"], use_tf=cfg["use_tf"]).to(DEV)
    counts = np.bincount(y[tr], minlength=C).astype(float); alpha = (1 / np.maximum(counts, 1)) ** (0.0 if cfg.get("sampler", "sqrt") == "sqrt" else 1.0); alpha = alpha / alpha.sum() * C   # with the sqrt sampler the focal weights stay uniform (re-balancing lives in the sampler only)
    focal = FocalLoss(alpha, gamma=2.0).to(DEV); proto = PrototypeBank(C, cfg["d"]).to(DEV)
    A_t = torch.as_tensor(A_ref, dtype=torch.float32, device=DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    seq_std = lambda b: torch.as_tensor((np.asarray(seq[b][:, :, fm], dtype=np.float32) - seq_tr_mean[b][:, None, :]) / seq_tr_std[b][:, None, :]).to(DEV)
    anchor_idx = rng.choice(tr, min(512, len(tr)), replace=False)
    anchors = (torch.as_tensor(Vs[anchor_idx]).to(DEV), None)
    best, best_state, patience, curve = -1, None, 0, []
    for ep in range(cfg["epochs"]):
        model.train(); tl = 0; nb = 0
        for b in make_batches(tr, y, cfg["bs"], rng, cfg.get("sampler", "sqrt")):
            v = torch.as_tensor(Vs[b]).to(DEV); s = seq_std(b); yb = torch.as_tensor(y[b]).to(DEV)
            s, v = augment(s, v, rng)
            logits, z = model(v, s)
            loss = focal(logits, yb)
            if cfg["use_proto"]:
                proto.update(z.detach(), yb); loss = loss + cfg["lam_p"] * proto.loss(z, yb)
            if cfg["use_graphcons"] and cfg["use_proto"]:
                loss = loss + cfg["lam_g"] * proto.graph_consistency(z, yb, A_t)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item(); nb += 1
        sched.step()
        pv, _, _, _ = evaluate(model, seq_std_np(seq, fm, seq_tr_mean, seq_tr_std), Vs, va, anchors)
        mv = metrics(y[va], pv, C, names); score = mv["macro_f1"]
        curve.append(dict(epoch=ep, train_loss=tl / nb, val_macro_f1=score, val_acc=mv["accuracy"]))
        if score > best: best, best_state, patience = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else: patience += 1
        if ep % 5 == 0 or patience == 0: print(f"  ep {ep:3d} loss {tl/nb:.4f} val macroF1 {score:.4f} acc {mv['accuracy']:.4f}", flush=True)
        if patience >= cfg["patience"]: break
    model.load_state_dict(best_state)
    pt, zt, att, gate = evaluate(model, seq_std_np(seq, fm, seq_tr_mean, seq_tr_std), Vs, te, anchors)
    mt = metrics(y[te], pt, C, names)
    # recording-level evaluation: mean segment probability per recording (standard practice; no label used)
    rec = pd.DataFrame({"path": meta.path.values[te], "y": y[te]}); rec_ids = rec.path.values
    rp = pd.DataFrame(pt).groupby(rec_ids).mean(); ry = rec.groupby("path").y.first().loc[rp.index].values
    mt_rec = metrics(ry, rp.values, C, names)
    n_params = sum(p.numel() for p in model.parameters())
    # latency (single segment)
    model.eval(); v1 = torch.as_tensor(Vs[te[:1]]).to(DEV); s1 = seq_std(te[:1])
    with torch.no_grad():
        for _ in range(5): model(v1, s1, anchors[0], None)
        if DEV.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50): model(v1, s1, anchors[0], None)
        if DEV.type == "cuda": torch.cuda.synchronize()
        lat_ms = (time.time() - t0) / 50 * 1000
    out = dict(run_id=run_id, cfg=cfg, classes=names, n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)),
               patients_train=int(meta.patient.iloc[tr].nunique()), patients_val=int(meta.patient.iloc[va].nunique()), patients_test=int(meta.patient.iloc[te].nunique()),
               class_counts=dict(train=np.bincount(y[tr], minlength=C).tolist(), val=np.bincount(y[va], minlength=C).tolist(), test=np.bincount(y[te], minlength=C).tolist()),
               patient_counts=dict(test={names[c]: int(meta.patient.iloc[te][y[te] == c].nunique()) for c in range(C)}),
               n_selected=int(mask.sum()), n_features=int(len(mask)), gqdo_history=gq_hist, A_ref=A_ref.tolist(), S_ref=S_ref.tolist(),
               best_val_macro_f1=float(best), epochs_run=len(curve), curve=curve, test=mt, test_recording=mt_rec, n_recordings_test=int(len(ry)),
               n_params=int(n_params), latency_ms=float(lat_ms),
               minutes=(time.time() - t_start) / 60)
    json.dump(out, open(os.path.join(RUNS, run_id + ".json"), "w"), indent=1)
    np.savez_compressed(os.path.join(RUNS, run_id + ".npz"), prob=pt, y=y[te], z=zt, idx=te, mask=mask,
                        tf_att=att if att is not None else np.zeros(0), gate=gate if gate is not None else np.zeros(0),
                        patient=meta.patient.values[te].astype(str), path=meta.path.values[te].astype(str), t0=meta.t0.values[te])
    torch.save(dict(state=best_state, cfg=cfg, mask=mask, scaler_mean=sc_v.mean_, scaler_scale=sc_v.scale_, seq_mean=g_mean, seq_std=g_std,
                    anchors=Vs[anchor_idx], classes=names, fm=fm), os.path.join(RUNS, run_id + ".pt"))
    print(f"[{run_id}] TEST segment: acc {mt['accuracy']:.4f} macroF1 {mt['macro_f1']:.4f} AUC {mt['macro_auc']} | recording: acc {mt_rec['accuracy']:.4f} macroF1 {mt_rec['macro_f1']:.4f} AUC {mt_rec['macro_auc']} ({out['minutes']:.1f} min)", flush=True)
    return out


class seq_std_np:
    """lazy standardized view used by evaluate()"""
    def __init__(self, seq, fm, m, s): self.seq, self.fm, self.m, self.s = seq, fm, m, s
    def __getitem__(self, b):
        x = np.asarray(self.seq[b][:, :, self.fm], dtype=np.float32)
        if np.ndim(self.m) == 2: return (x - self.m[b][:, None, :]) / self.s[b][:, None, :]
        return (x - self.m) / self.s


# ----------------------------------------------------------------------------- experiment definitions
DEFAULT = dict(blocks=["mel", "mfcc", "cqt", "cwt"], fs="gqdo", d=128, tf_layers=3, use_gat=True, use_tf=True, use_proto=True,
               use_graphcons=False, lam_p=0.5, lam_g=0.1, lr=1e-4 * 3, bs=128, epochs=40, patience=8, sampler="sqrt")


def run_experiment(exp, seed=0, **over):
    cfg = dict(DEFAULT); cfg.update(over); cfg["seed"] = seed; cfg["exp"] = exp
    if exp.startswith("percorpus:"):
        corpus = exp.split(":")[1]; seq, vec, meta = load_corpus(corpus)
        names = [n for n in FIVE[corpus] if n in set(meta.label5)]
        meta = meta[meta.label5.isin(names)].reset_index(drop=True); keep = meta.index.values
        seq = np.asarray(seq)[keep] if len(keep) < len(vec) else seq; vec = vec[keep]
        cfg["label_col"] = "label5"; tr, va, te = split_patients(meta, "label5", seed)
    elif exp == "randomsplit:MERGED":
        parts = [load_corpus(c) for c in ["ASTHMA_V2", "KAUH", "ICBHI"]]
        meta = pd.concat([p[2] for p in parts], ignore_index=True); keep = meta.label4.notna().values
        seq = np.concatenate([np.asarray(p[0]) for p in parts])[keep]; vec = np.concatenate([p[1] for p in parts])[keep]
        meta = meta[keep].reset_index(drop=True); names = MERGED4; cfg["label_col"] = "label4"
        from sklearn.model_selection import train_test_split
        yy = meta.label4.values; idx = np.arange(len(meta))
        trv, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=yy)
        tr, va = train_test_split(trv, test_size=0.125, random_state=seed, stratify=yy[trv])
    elif exp.startswith("randomsplit:"):
        corpus = exp.split(":")[1]; seq, vec, meta = load_corpus(corpus)
        names = [n for n in FIVE[corpus] if n in set(meta.label5)]
        meta = meta[meta.label5.isin(names)].reset_index(drop=True); keep = meta.index.values
        seq = np.asarray(seq)[keep] if len(keep) < len(vec) else seq; vec = vec[keep]
        cfg["label_col"] = "label5"
        from sklearn.model_selection import train_test_split
        yy = meta.label5.values; idx = np.arange(len(meta))
        trv, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=yy)
        tr, va = train_test_split(trv, test_size=0.125, random_state=seed, stratify=yy[trv])
    elif exp == "merged" or exp.startswith("abl:"):
        parts = [load_corpus(c) for c in ["ASTHMA_V2", "KAUH", "ICBHI"]]
        meta = pd.concat([p[2] for p in parts], ignore_index=True); keep = meta.label4.notna().values
        seq = np.concatenate([np.asarray(p[0]) for p in parts])[keep]; vec = np.concatenate([p[1] for p in parts])[keep]
        meta = meta[keep].reset_index(drop=True); names = MERGED4; cfg["label_col"] = "label4"
        tr, va, te = split_patients(meta, "label4", seed)
    elif exp.startswith("lodo:"):
        held = exp.split(":")[1]; order = ["ASTHMA_V2", "KAUH", "ICBHI"]
        parts = {c: load_corpus(c) for c in order}
        meta = pd.concat([parts[c][2] for c in order], ignore_index=True); keep = meta.label4.notna().values
        seq = np.concatenate([np.asarray(parts[c][0]) for c in order])[keep]; vec = np.concatenate([parts[c][1] for c in order])[keep]
        meta = meta[keep].reset_index(drop=True); names = MERGED4; cfg["label_col"] = "label4"
        # Asthma v2 files traced to the held-out corpus (exact or probable copies) are removed from the training side
        if "source" in meta.columns:
            leak = (meta.corpus == "ASTHMA_V2") & meta.source.fillna("AV2").str.startswith(held)
            if held == "ASTHMA_V2": leak[:] = False
            keep2 = ~leak.values; seq, vec, meta = seq[keep2], vec[keep2], meta[keep2].reset_index(drop=True)
            print(f"[lodo:{held}] removed {int(leak.sum())} Asthma v2 segments traced to {held}", flush=True)
        src = np.where(meta.corpus != held)[0]; te = np.where(meta.corpus == held)[0]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
        a, b = next(gss.split(src, meta.label4.values[src], meta.patient.values[src])); tr, va = src[a], src[b]
    else:
        raise ValueError(exp)
    run_id = f"{exp.replace(':', '_')}_s{seed}" + ("" if not over else "_" + "_".join(f"{k}-{'+'.join(v) if isinstance(v, list) else v}" for k, v in over.items()))
    if os.path.exists(os.path.join(RUNS, run_id + ".json")):
        print("exists:", run_id); return json.load(open(os.path.join(RUNS, run_id + ".json")))
    return train_eval(run_id, seq, vec, meta, tr, va, te, names, cfg)


if __name__ == "__main__":
    exp = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    over = {}
    for kv in sys.argv[3:]:
        k, v = kv.split("=")
        if k == "blocks": over[k] = v.split("+")
        elif v in ("True", "False"): over[k] = v == "True"
        elif k == "fs": over[k] = v
        elif k == "sampler": over[k] = v
        elif k == "corpus_norm": over[k] = v == "True"
        else: over[k] = float(v) if "." in v else int(v)
    run_experiment(exp, seed, **over)
