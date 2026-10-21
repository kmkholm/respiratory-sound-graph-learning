"""Stage 1-2 of the manuscript: preprocessing (resample 16 kHz, 150-2000 Hz zero-phase Butterworth,
peak normalisation, spectral subtraction, 2 s / 1 s segmentation, zero padding) and the four
time-frequency representations (CWT, CQT, log-Mel, MFCC) per segment.

Outputs (data/features):
  seq_<corpus>.npy   float16 (N, T=101, 228)  frame-level fused features  [Mel128 | MFCC20 | CQT48 | CWT32]
  vec_<corpus>.npy   float32 (N, 456)         segment-level pooled features (per-bin mean, std)
  meta_<corpus>.csv  one row per segment: path, patient, label5, label4, seg_idx, t0
"""
import os, sys, json, numpy as np, pandas as pd, soundfile as sf, librosa, pywt
from scipy.signal import butter, sosfiltfilt, stft, istft
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(__file__))
from common import *

N_FFT, HOPF = 1024, 320            # 20 ms hop at 16 kHz -> 101 frames per 2 s segment
N_MEL, N_MFCC = 128, 20
CQT_FMIN, CQT_BINS, CQT_BPO = 125.0, 48, 12   # 125-2000 Hz, 4 octaves
CWT_SR, CWT_SCALES = 4000, 32      # CWT on a 4 kHz copy, 32 log-spaced scales covering 150-2000 Hz
SOS = butter(4, [150, 2000], btype="band", fs=SR, output="sos")
DELTA = 0.02
BLOCK_NAMES = {"mel": (0, 128), "mfcc": (128, 148), "cqt": (148, 196), "cwt": (196, 228)}

_cwt_freqs = np.geomspace(150, 2000, CWT_SCALES)
_cwt_scales = pywt.scale2frequency("cmor1.5-1.0", 1) / (_cwt_freqs / CWT_SR)


def preprocess(path):
    y, sr = sf.read(path, dtype="float32", always_2d=True); y = y.mean(1)
    if len(y) < 0.1 * sr: return None
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    y = sosfiltfilt(SOS, y).astype(np.float32)
    y = y / (np.max(np.abs(y)) + 1e-8)
    # spectral subtraction, noise estimated from the 10 % lowest-energy frames
    f, t, Z = stft(y, fs=SR, nperseg=512, noverlap=384)
    mag, ph = np.abs(Z), np.angle(Z)
    e = (mag ** 2).sum(0); idx = np.argsort(e)[: max(3, len(e) // 10)]
    noise = mag[:, idx].mean(1, keepdims=True)
    mag2 = np.maximum(mag - noise, DELTA * mag)
    _, y = istft(mag2 * np.exp(1j * ph), fs=SR, nperseg=512, noverlap=384)
    y = y[: len(y)].astype(np.float32)
    return y


def segments(y):
    W, H = int(SEG * SR), int(HOP * SR)
    if len(y) < MIN_LEN * SR: return []
    if len(y) < W: return [(0, np.pad(y, (0, W - len(y))))]
    n = 1 + (len(y) - W) // H
    return [(k * H / SR, y[k * H: k * H + W]) for k in range(n)]


def features(seg):
    S = np.abs(librosa.stft(seg, n_fft=N_FFT, hop_length=HOPF)) ** 2
    mel = librosa.feature.melspectrogram(S=S, sr=SR, n_mels=N_MEL, fmin=100, fmax=2200)
    logmel = librosa.power_to_db(mel + 1e-10)                         # (128, 101)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=N_MFCC)               # (20, 101)
    cqt = np.abs(librosa.cqt(seg, sr=SR, hop_length=HOPF, fmin=CQT_FMIN, n_bins=CQT_BINS, bins_per_octave=CQT_BPO))
    cqt = librosa.amplitude_to_db(cqt + 1e-10)                         # (48, 101)
    y4 = librosa.resample(seg, orig_sr=SR, target_sr=CWT_SR, res_type="soxr_hq")
    coef, _ = pywt.cwt(y4, _cwt_scales, "cmor1.5-1.0", sampling_period=1 / CWT_SR)
    sc = np.abs(coef) ** 2                                             # (32, 8000)
    T = logmel.shape[1]
    edges = np.linspace(0, sc.shape[1], T + 1).astype(int)
    cwt = np.stack([sc[:, edges[i]: max(edges[i] + 1, edges[i + 1])].mean(1) for i in range(T)], 1)
    cwt = 10 * np.log10(cwt + 1e-10)                                   # (32, 101)
    X = np.concatenate([logmel, mfcc, cqt, cwt], 0).T.astype(np.float32)  # (101, 228)
    return X


def process_recording(row):
    y = preprocess(row["path"])
    if y is None: return []
    out = []
    for k, (t0, seg) in enumerate(segments(y)):
        X = features(seg)
        out.append((row["path"], row["patient"], row["label5"], row["label4"], k, t0, X))
    return out


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA, "manifest.csv"))
    corpora = sys.argv[1:] or ["KAUH", "ASTHMA_V2", "ICBHI"]
    for corpus in corpora:
        sub = df[df.corpus == corpus].reset_index(drop=True)
        res = Parallel(n_jobs=6, verbose=5)(delayed(process_recording)(r) for _, r in sub.iterrows())
        rows = [x for r in res for x in r]
        seq = np.stack([r[6] for r in rows]).astype(np.float16)
        vec = np.concatenate([seq.astype(np.float32).mean(1), seq.astype(np.float32).std(1)], 1)
        meta = pd.DataFrame([r[:6] for r in rows], columns=["path", "patient", "label5", "label4", "seg_idx", "t0"])
        np.save(os.path.join(FEAT, f"seq_{corpus}.npy"), seq)
        np.save(os.path.join(FEAT, f"vec_{corpus}.npy"), vec)
        meta.to_csv(os.path.join(FEAT, f"meta_{corpus}.csv"), index=False)
        print(corpus, "segments", len(meta), "patients", meta.patient.nunique(), seq.shape, flush=True)
        print(meta.groupby("label5").agg(seg=("path", "count"), rec=("path", "nunique"), pat=("patient", "nunique")), flush=True)
    json.dump(BLOCK_NAMES, open(os.path.join(FEAT, "blocks.json"), "w"))
