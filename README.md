# Graph-guided respiratory sound classification

Research code for multi-class respiratory disease classification from lung-sound recordings. The pipeline fuses four
time-frequency representations, builds a disease relationship graph from the training data, selects features with a
graph-guided quantum dragonfly optimizer, and classifies with a graph-attention + Transformer network regularised by
disease prototypes. Evaluation is strictly patient-level, with recording-level aggregation, repeated splits,
leave-one-corpus-out and an external corpus.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 0 | `common.py` | paths, label maps, manifests for ICBHI 2017, KAUH, Asthma Detection v2 and RespiratoryDatabase@TR |
| 1 | `01_features.py` | resample 16 kHz, 150-2000 Hz zero-phase band-pass, peak normalisation, spectral subtraction, 2 s / 1 s segmentation, log-Mel + MFCC + CQT + CWT per segment |
| 2 | `build_patient_map.py` | traces every Asthma Detection v2 file to its source recording in KAUH / ICBHI by normalised cross-correlation and unifies subject identities (requires the two cross-correlation CSVs, see below) |
| 3 | `graph_gqdo.py` | disease relationship graph and the graph-guided quantum dragonfly optimizer (variants: `none`, `da`, `qda`, `gqdo`) |
| 4 | `model.py` | GATFormer: dense graph attention over a label-free k-NN sample graph (+ training anchors at inference), Transformer branch, gated fusion, focal loss, prototype bank with graph-consistency term |
| 5 | `run.py` | one experiment: patient-level split, standardisation (optionally corpus-wise), graph, feature selection, training with early stopping, segment- and recording-level metrics |
| 6 | `run_all.py` | the fixed experiment grid (per corpus x 5 seeds, random-split comparability runs, merged corpus, leave-one-corpus-out with and without corpus-wise standardisation, 16 ablations, external test, explainability) |
| 7 | `baseline_lr.py` | same-protocol logistic-regression baseline on the identical features and splits |
| 8 | `eval_external.py` | apply a trained merged model to an external corpus without adaptation |
| 9 | `xai.py` | SHAP band-enrichment per class and attention enrichment inside expert-annotated wheeze/crackle intervals (ICBHI) |

## Data

Download the corpora yourself and point the scripts at them:

* ICBHI 2017 Respiratory Sound Database (Kaggle `vbookshelf/respiratory-sound-database`) -> `RS_ICBHI`
* KAUH lung sound dataset (Mendeley Data `jwyy9np4gv`) -> `data/KAUH` (unzipped, flat)
* Asthma Detection Dataset Version 2 (Kaggle `mohammedtawfikmusaed/asthma-detection-dataset-version-2`) -> `RS_ASTHMA_V2`
* RespiratoryDatabase@TR (Mendeley Data `p9z4h98s6j`) -> `data/TR`

The duplicate check that feeds `build_patient_map.py` computes, for every Asthma Detection v2 file, the maximum
normalised cross-correlation against every KAUH and every ICBHI recording (both resampled to a common rate) and writes
`results/dup_av2_kauh.csv` and `results/dup_av2_icbhi.csv` with columns `cls,file,ncc,<source>_file`.

## Run

```bash
pip install -r requirements.txt
python common.py                 # manifest
python 01_features.py            # features for all corpora
python build_patient_map.py      # unified subject identities
python baseline_lr.py
python run_all.py                # full grid, one GPU
python run.py percorpus:KAUH 0   # a single experiment
python run.py lodo:ICBHI 0 corpus_norm=True
```

Results are written as JSON/NPZ per run under `results/runs/`.

## License

MIT.
