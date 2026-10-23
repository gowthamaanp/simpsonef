#!/usr/bin/env bash
# Rebuild every network of the final pipeline from the raw data, then evaluate. All runs use seed 0 (Python, NumPy, PyTorch CPU
# and GPU are seeded; GPU kernels are not bit-exact, so a rerun matches the reported numbers up to normal run-to-run variation);
# every checkpoint stores its full configuration under "config".
#
#   Network A  single-frame scale-space tracer (13-frame window)   -> per-frame chords
#   BiGRU      ED / ES frame detector on Network A's chord sequence -> the beats
#   Network B  pair tracer (ED window + ES window, 13 frames each) -> ED and ES tracings -> Simpson formula -> EF
#
# EchoNet-Dynamic is expected in data/ (Videos/, VolumeTracings.csv, master.csv), CAMUS in data/camus/CAMUS_public (or CAMUS_DIR).
# TRAIN is used for training, VAL only for checkpoint selection, TEST and CAMUS only for the final evaluation.
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEED=0
STAGES=${STAGES:-"data a b eval"}        # e.g. STAGES="a" ./reproduce.sh to rebuild only Network A and the BiGRU
has() { [[ " $STAGES " == *" $1 "* ]]; }

# Data: tracing records, and the frame windows around every annotated frame (13 frames, step 4).
if has data; then
python experiments/build_dataset.py
python experiments/build_context_bank.py --out data_cache/context_frames.npz
fi

# Network A and the BiGRU frame detector.
if has a; then
python experiments/train_shape_estimator.py --arch scalespace --width 24 --epochs 40 --ema 0.999 --dropout 0.1 \
    --scale-range 0.5,2.5 --batch-size 32 --seed $SEED --out checkpoints/net_a.pt
python experiments/cache_features.py --ckpt checkpoints/net_a.pt --out data_cache/features_net_a.npz
python experiments/train_heads.py --features data_cache/features_net_a.npz --seed $SEED --out checkpoints/bigru.pt
fi

# Network B (BeatTracer): pair pretraining at width 32 without scanner-style degradations, then one EF-unit formula-loss stage
# (EF loss in EF units, weighted by inverse sqrt EF-bin frequency; no dropout). This variant was selected on VAL among all archived
# variants (full pipeline, highest VAL R2); see baselines/review/ablation_val.py.
if has b; then
B="--arch scalespace --width 32 --pair --ema 0.999 --batch-size 32 --scanner-prob 0 --seed $SEED"
python experiments/train_shape_estimator.py $B --epochs 40 --dropout 0.1 --scale-range 0.5,2.5 --out checkpoints/net_b_pre.pt
python experiments/train_shape_estimator.py $B --w-ef 40 --epochs 12 --lr 1.0e-4 --dropout 0.0 --init checkpoints/net_b_pre.pt --out checkpoints/net_b.pt
fi

# Evaluation: EchoNet TEST (all videos; ED/ES from Network A + BiGRU; tracings from Network B; 15-copy test-time averaging;
# formula EF per beat, median over beats) and CAMUS test zero-shot (Network A + BiGRU on the released half-sequences).
if has eval; then
python experiments/eval_pair.py --ckpt checkpoints/net_b.pt --features data_cache/features_net_a.npz --det checkpoints/bigru.pt \
    --only echonet --tta
python experiments/camus_blind_ef.py --detector --ckpt checkpoints/net_a.pt --head checkpoints/bigru.pt
fi
