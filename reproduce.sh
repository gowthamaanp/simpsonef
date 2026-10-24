#!/usr/bin/env bash
# Rebuild the three SimpsonEF networks from the raw data, then evaluate. All runs use seed 0 (Python, NumPy, PyTorch CPU and GPU
# are seeded; GPU kernels are not bit-exact, so a rerun matches the reported numbers up to normal run-to-run variation); every
# checkpoint stores its full configuration under "config".
#
#   FrameTracer  13-frame window -> 42 points on every frame          (width 24, scanner-style degradations)
#   BeatFinder   BiGRU on FrameTracer's chords -> ED / ES of every beat (+ EF quantiles for the interval)
#   BeatTracer   ED window + ES window -> both tracings -> Simpson's rule (width 32, no degradations, then the EF-unit stage)
#
# EchoNet-Dynamic is expected in data/ (Videos/, FileList.csv, master.csv, VolumeTracings.csv, annotation/Masks/), CAMUS in
# data/camus/CAMUS_public (or CAMUS_DIR). TRAIN trains, VAL selects checkpoints and calibrates the interval, TEST and CAMUS are
# only evaluated.
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEED=0
STAGES=${STAGES:-"data frame beat eval"}     # e.g. STAGES="eval" ./reproduce.sh to evaluate the released checkpoints only
has() { [[ " $STAGES " == *" $1 "* ]]; }

# Data: one record per annotated frame, and the 13 frames (offsets -24..24, step 4) around every annotated frame.
if has data; then
python scripts/build_dataset.py
python scripts/build_context_bank.py --out data_cache/context_frames.npz
fi

# FrameTracer (40 epochs), its chords on every frame, BeatFinder (16 epochs).
if has frame; then
python scripts/train_tracer.py --seed $SEED --out checkpoints/frametracer.pt
python scripts/cache_chords.py --ckpt checkpoints/frametracer.pt --out data_cache/chords.npz
python scripts/train_beatfinder.py --chords data_cache/chords.npz --seed $SEED --out checkpoints/beatfinder.pt
fi

# BeatTracer: pair pretraining (40 epochs), then the EF-unit formula-loss stage (12 epochs, lr 1e-4, no dropout).
if has beat; then
B="--pair --width 32 --scanner-prob 0 --seed $SEED"
python scripts/train_tracer.py $B --out checkpoints/beattracer_pre.pt
python scripts/train_tracer.py $B --w-ef 40 --epochs 12 --lr 1e-4 --dropout 0 --init checkpoints/beattracer_pre.pt --out checkpoints/beattracer.pt
fi

# Evaluation with src/pipeline.py (the code run.py uses): VAL, the conformal calibration on VAL, TEST once, CAMUS zero-shot.
if has eval; then
python scripts/eval_echonet.py --split VAL
python scripts/eval_echonet.py --split TEST
python scripts/conformal.py --val results/echonet_VAL.csv --test results/echonet_TEST.csv
python scripts/eval_camus.py
fi
