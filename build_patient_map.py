"""Unify patient identities across corpora using the waveform-duplicate checks (results/dup_av2_kauh.csv,
results/dup_av2_icbhi.csv: best normalised cross-correlation of every Asthma Detection v2 file against every
KAUH / ICBHI recording).

  ncc > 0.9        exact copy      -> source patient id, source = KAUH / ICBHI
  0.7 < ncc <= 0.9 probable copy (augmented / re-filtered) -> source patient id, source = KAUH~ / ICBHI~
  ncc <= 0.7       untraced        -> AV2 pseudo id, source = AV2

Outputs: data/patient_map.csv and the rewritten patient column in data/features/meta_ASTHMA_V2.csv
(original id kept in patient_av2, provenance in column source).
"""
import os, re, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import *

k = pd.read_csv(os.path.join(RES, "dup_av2_kauh.csv")).set_index("file")
i = pd.read_csv(os.path.join(RES, "dup_av2_icbhi.csv")).set_index("file")
man = pd.read_csv(os.path.join(DATA, "manifest.csv")); av2 = man[man.corpus == "ASTHMA_V2"]
rows = []
for r in av2.itertuples():
    nk, fk = (float(k.loc[r.file, "ncc"]), k.loc[r.file, "kauh_file"]) if r.file in k.index else (0.0, "")
    ni, fi = (float(i.loc[r.file, "ncc"]), i.loc[r.file, "icbhi_file"]) if r.file in i.index else (0.0, "")
    if max(nk, ni) <= 0.7:
        src, sfile, ncc, pid = "AV2", "", max(nk, ni), r.patient
    elif nk >= ni:
        src, sfile, ncc = "KAUH" if nk > 0.9 else "KAUH~", fk, nk; pid = "KAUH_" + re.match(r"[BDE]P(\d+)_", fk).group(1)
    else:
        src, sfile, ncc = "ICBHI" if ni > 0.9 else "ICBHI~", fi, ni; pid = "ICBHI_" + fi.split("_")[0]
    rows.append(dict(file=r.file, label5=r.label5, av2_patient=r.patient, source=src, source_file=sfile, ncc=round(ncc, 3), patient_unified=pid))
pm = pd.DataFrame(rows); pm.to_csv(os.path.join(DATA, "patient_map.csv"), index=False)
print(pm.groupby(["label5", "source"]).size().unstack(fill_value=0))
print("AV2 pseudo-subjects:", pm.av2_patient.nunique(), "| unified subjects:", pm.patient_unified.nunique(),
      "| of which KAUH:", pm.patient_unified.str.startswith("KAUH").sum() and pm[pm.patient_unified.str.startswith("KAUH")].patient_unified.nunique(),
      "ICBHI:", pm[pm.patient_unified.str.startswith("ICBHI")].patient_unified.nunique(), "untraced:", pm[pm.source == "AV2"].patient_unified.nunique())

mp = os.path.join(FEAT, "meta_ASTHMA_V2.csv"); meta = pd.read_csv(mp)
if "patient_av2" not in meta.columns: meta["patient_av2"] = meta.patient
m = dict(zip(pm.file, pm.patient_unified)); s = dict(zip(pm.file, pm.source))
meta["patient"] = meta.path.map(lambda p: m[os.path.basename(p)]); meta["source"] = meta.path.map(lambda p: s[os.path.basename(p)])
meta.to_csv(mp, index=False)
print(meta.groupby(["label5", "source"]).agg(segments=("path", "count"), patients=("patient", "nunique")))
