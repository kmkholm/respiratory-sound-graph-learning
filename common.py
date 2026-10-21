"""Shared paths, manifest construction and label maps.
Set RS_ROOT (project root holding data/ and results/), RS_ASTHMA_V2 and RS_ICBHI (dataset folders) as environment variables,
or place the datasets under data/ASTHMA_V2, data/ICBHI, data/KAUH and data/TR."""
import os, re, glob
import pandas as pd

ROOT = os.environ.get("RS_ROOT", os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FEAT = os.path.join(DATA, "features")
os.makedirs(RES, exist_ok=True); os.makedirs(FEAT, exist_ok=True)

ASTHMA_V2 = os.environ.get("RS_ASTHMA_V2", os.path.join(DATA, "ASTHMA_V2"))   # Kaggle: mohammedtawfikmusaed/asthma-detection-dataset-version-2
ICBHI = os.environ.get("RS_ICBHI", os.path.join(DATA, "ICBHI"))   # Kaggle: vbookshelf/respiratory-sound-database
KAUH = os.path.join(DATA, "KAUH")

# merged 4-class schema shared by all corpora
MERGED4 = ["Healthy", "Asthma", "COPD", "Pneumonia"]
# per-corpus 5-class schemas (as in the manuscript's Tables 6-8)
FIVE = {
    "ASTHMA_V2": ["Healthy", "Asthma", "COPD", "Pneumonia", "Bronchial"],
    "KAUH":      ["Healthy", "Asthma", "COPD", "Pneumonia", "Bronchial"],
    "ICBHI":     ["Healthy", "COPD", "Pneumonia", "URTI", "Bronchiolitis"],
}

SR = 16000
SEG = 2.0      # seconds
HOP = 1.0      # seconds
MIN_LEN = 1.0  # discard shorter recordings


def manifest_asthma_v2():
    rows = []
    cls_map = {"asthma": "Asthma", "copd": "COPD", "healthy": "Healthy", "pneumonia": "Pneumonia", "Bronchial": "Bronchial"}
    for c in sorted(os.listdir(ASTHMA_V2)):
        for f in sorted(glob.glob(os.path.join(ASTHMA_V2, c, "*.wav"))):
            b = os.path.basename(f)
            # file name pattern: P<k><Class><site/type code>[_<serial>].wav ; P<k> is the subject index within the class
            m = re.match(r"P(\d+)", b)
            subj = m.group(1) if m else "0"
            rows.append(dict(corpus="ASTHMA_V2", path=f, file=b, label5=cls_map[c],
                             patient=f"AV2_{cls_map[c]}_P{subj}"))
    return pd.DataFrame(rows)


def manifest_kauh():
    rows = []
    for f in sorted(glob.glob(os.path.join(KAUH, "*.wav"))):
        b = os.path.basename(f)
        m = re.match(r"([BDE])P(\d+)_([^,]+),", b)
        if not m:
            continue
        mode, pid, diag = m.group(1), m.group(2), m.group(3).strip().lower()
        if diag == "n": lab = "Healthy"
        elif diag == "asthma": lab = "Asthma"
        elif diag == "copd": lab = "COPD"
        elif diag == "pneumonia": lab = "Pneumonia"
        elif diag == "bron": lab = "Bronchial"
        else: continue  # heart failure, lung fibrosis, pleural effusion, mixed diagnoses
        rows.append(dict(corpus="KAUH", path=f, file=b, label5=lab, patient=f"KAUH_{pid}", mode=mode))
    return pd.DataFrame(rows)


def manifest_icbhi():
    diag = pd.read_csv(os.path.join(ICBHI, "patient_diagnosis.csv"), header=None, names=["pid", "diag"])
    dmap = dict(zip(diag.pid.astype(str), diag.diag.str.strip()))
    rows = []
    for f in sorted(glob.glob(os.path.join(ICBHI, "audio_and_txt_files", "*.wav"))):
        b = os.path.basename(f); pid = b.split("_")[0]
        lab = dmap[pid]
        if lab not in FIVE["ICBHI"] and lab != "Asthma":
            continue  # Bronchiectasis, LRTI excluded (as in the manuscript's 5-class ICBHI table)
        rows.append(dict(corpus="ICBHI", path=f, file=b, label5=lab, patient=f"ICBHI_{pid}",
                         device=b.replace(".wav", "").split("_")[-1]))
    return pd.DataFrame(rows)


TR = os.path.join(DATA, "TR")


def manifest_tr():
    """RespiratoryDatabase@TR (Altan et al.): 42 subjects x 12 channels, COPD0 (no COPD) ... COPD4. External test only."""
    lab = pd.read_excel(os.path.join(TR, "Labels.xlsx"), header=None, names=["pid", "diag"]).iloc[1:]
    dmap = dict(zip(lab.pid.astype(str).str.strip(), lab.diag.astype(str).str.strip()))
    rows = []
    for f in sorted(glob.glob(os.path.join(TR, "RespiratoryDatabase@TR", "*.wav"))):
        b = os.path.basename(f); pid = b.split("_")[0]; sev = dmap[pid]
        rows.append(dict(corpus="TR", path=f, file=b, label5="Healthy" if sev == "COPD0" else "COPD", patient=f"TR_{pid}", severity=sev))
    return pd.DataFrame(rows)


def full_manifest():
    df = pd.concat([manifest_asthma_v2(), manifest_kauh(), manifest_icbhi(), manifest_tr()], ignore_index=True)
    df["label4"] = df.label5.where(df.label5.isin(MERGED4), None)
    return df


if __name__ == "__main__":
    df = full_manifest()
    print(df.groupby(["corpus", "label5"]).agg(files=("path", "count"), patients=("patient", "nunique")))
    df.to_csv(os.path.join(DATA, "manifest.csv"), index=False)
