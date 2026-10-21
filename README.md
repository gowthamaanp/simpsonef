# SimpsonEF

**Explainable left-ventricular ejection fraction from predicted sonographer tracings and Simpson's rule.**

SimpsonEF computes LVEF the way a sonographer does. It draws the volume tracing (long axis and 20 chords) at end-diastole (ED)
and end-systole (ES) of every beat and applies Simpson's method of disks; no learned component maps an image to EF.

**[Project page and in-browser demo](https://gowthamaanp.github.io/simpsonef/)** · [Paper (PDF)](docs/assets/simpsonef.pdf)

![SimpsonEF](docs/assets/fig1_overview.png)

| EchoNet-Dynamic TEST, all 1276 videos, no annotated frame | LVEF R² | MAE | Params |
|---|---|---|---|
| SimpsonEF (formula EF) | 0.808 | 4.00 | 3.38M |
| Zero-shot CAMUS (biplane, CAMUS convention) | 0.71 | 5.08 | |

## Pipeline

1. **FrameTracer** (`src/shape_estimator.py`, `checkpoints/shape_estimator.pt`): scale-space encoder–decoder that traces the 42
   tracing points on every frame from a 13-frame window.
2. **BeatFinder** (`src/temporal_detector.py`, `checkpoints/ef_head.pt`): bidirectional GRU on the chord widths over time; marks
   ED and ES of every beat.
3. **BeatTracer** (`checkpoints/d2s3.pt`): reads the 25-frame windows around a beat's ED and ES together and draws both tracings.
4. **Simpson's rule**: V = π/4 · ℓ/20 · Σ dᵢ², EF = 1 − V_ES/V_ED per beat, median over beats, with a conformal 90% interval
   (`checkpoints/conformal_formula.json`).

## Use

```bash
pip install -r requirements.txt
# EchoNet-Dynamic in data/ (Videos/, VolumeTracings.csv, master.csv), CAMUS in data/camus/CAMUS_public
python experiments/eval_pair.py --ckpt checkpoints/d2s3.pt --only echonet --tta   # EchoNet TEST, formula EF
python experiments/camus_blind_ef.py --detector                                   # CAMUS zero-shot
./reproduce.sh                                                                    # rebuild all three networks and evaluate
```

`reproduce.sh` rebuilds every network from the data with fixed seeds (a rebuild gives R² 0.807 on TEST and 0.713 on CAMUS).
The browser demo (`docs/`) runs the ONNX exports (`tools/export_onnx.py`) with ONNX Runtime Web; `docs/js/pipeline.js` matches
the PyTorch pipeline to within 0.1 EF point (checked with `tools/reference_ef.py`). It omits the 15-copy input averaging used in
the paper (TEST R² 0.801 instead of 0.808).

## Data

EchoNet-Dynamic and CAMUS are not included; obtain them from their providers under their terms.

## Licence

Code: MIT (`LICENSE`). Model weights: CC BY-NC 4.0, research use only, subject to the EchoNet-Dynamic research use agreement
(`MODEL_LICENSE.md`). Not a medical device.

## Citation

```bibtex
@misc{palani2026simpsonef,
  title        = {SimpsonEF: Explainable Ejection Fraction from Predicted Sonographer Tracings and Simpson's Rule},
  author       = {Palani, Gowthamaan and Krishnamurthi, Ganapathy},
  year         = {2026},
  howpublished = {\url{https://github.com/gowthamaanp/simpsonef}}
}
```
