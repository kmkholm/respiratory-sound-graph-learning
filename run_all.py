"""Fixed experiment grid (PROGRESS.md). Sequential: one GPU job at a time. Existing results are skipped.
usage: python run_all.py [stage ...]   stages: percorpus random merged lodo ablation external xai report
"""
import os, sys, subprocess, time, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from run import run_experiment, RUNS

ALL = ["mel", "mfcc", "cqt", "cwt"]
SEEDS = [0, 1, 2, 3, 4]
stages = sys.argv[1:] or ["percorpus", "random", "merged", "lodo", "lodo_cn", "ablation", "external", "xai"]
t0 = time.time()
log = open(os.path.join(RES, "run_all.log"), "a", encoding="utf-8")


def say(msg):
    line = f"[{time.strftime('%H:%M:%S')} +{(time.time()-t0)/60:6.1f} min] {msg}"; print(line, flush=True); log.write(line + "\n"); log.flush()


def go(exp, seed=0, **over):
    try:
        r = run_experiment(exp, seed, **over)
        say(f"{r['run_id']}: seg acc {r['test']['accuracy']:.4f} mF1 {r['test']['macro_f1']:.4f} | rec acc {r.get('test_recording', {}).get('accuracy', float('nan')):.4f} mF1 {r.get('test_recording', {}).get('macro_f1', float('nan')):.4f}")
    except Exception as e:
        import traceback; say(f"FAILED {exp} seed {seed} {over}: {e}\n{traceback.format_exc()}")


if "percorpus" in stages:
    for seed in SEEDS:
        for c in ["ASTHMA_V2", "KAUH", "ICBHI"]: go(f"percorpus:{c}", seed)
if "random" in stages:
    for c in ["ASTHMA_V2", "KAUH", "ICBHI"]: go(f"randomsplit:{c}", 0)
if "merged" in stages:
    for seed in SEEDS: go("merged", seed)
if "lodo" in stages:
    for held in ["KAUH", "ICBHI"]: go(f"lodo:{held}", 0)
if "lodo_cn" in stages:   # leave-one-corpus-out with label-free corpus-wise standardisation
    for held in ["KAUH", "ICBHI"]: go(f"lodo:{held}", 0, corpus_norm=True)
    go("merged", 0, corpus_norm=True)
    go("randomsplit:MERGED", 0)
if "ablation" in stages:
    for b in ALL: go(f"abl:rep_{b}", 0, blocks=[b])
    for b in ALL: go(f"abl:loo_{b}", 0, blocks=[x for x in ALL if x != b])
    for fs in ["none", "da", "qda"]: go(f"abl:fs_{fs}", 0, fs=fs)
    go("abl:gat_only", 0, use_tf=False); go("abl:tf_only", 0, use_gat=False)
    go("abl:no_proto", 0, use_proto=False)
if "external" in stages:
    for seed in SEEDS:
        rid = f"merged_s{seed}"
        if os.path.exists(os.path.join(RUNS, rid + ".pt")) and not os.path.exists(os.path.join(RUNS, rid + "_ext_TR.json")):
            say(subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "eval_external.py"), rid, "TR"], capture_output=True, text=True).stdout[-800:])
    for held in ["KAUH", "ICBHI"]:
        rid = f"lodo_{held}_s0"
        if os.path.exists(os.path.join(RUNS, rid + ".pt")) and not os.path.exists(os.path.join(RUNS, rid + "_ext_TR.json")):
            say(subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "eval_external.py"), rid, "TR"], capture_output=True, text=True).stdout[-800:])
if "xai" in stages:
    if os.path.exists(os.path.join(RUNS, "merged_s0.pt")) and not os.path.exists(os.path.join(RUNS, "merged_s0_xai.json")):
        r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "xai.py"), "merged_s0"], capture_output=True, text=True); say(r.stdout[-1500:] + r.stderr[-1500:])
say("ALL DONE")
