[Uploading README .md…]()
# MSPC-Net: Geometry-aware Semantic–Structural Consistency for Semi-supervised Medical Image Segmentation

MSPC-Net is a 3D student/EMA-teacher segmentation framework with TAP,
DEAC and SDM-BR regularizers. The supplied training source is preserved
under `new/`; `code/dataloaders` and `code/utils/test_patch.py` are a new,
independent replacement implementation because those directories were **not**
in the supplied source archive.

> **Reproducibility status**: The trainable modules and six user-supplied model
> checkpoints are included. The replacement data reader has not been validated
> against the original preprocessing pipeline that produced the checkpoints.
> The archive lacks a BTCV trainer and full LA/IAN training lists; BTCV 24/6 split files are now included for transparency. Do not claim
> exact reproduction until these gaps are resolved.

## Quick start (RTX 3090)

Tested checkpoint format: PyTorch `state_dict`, one-channel input,
2-class output. Original training hardware supplied by the authors:
**NVIDIA GeForce RTX 3090**. Exact CUDA/PyTorch versions of the original
runs have not been supplied; install mutually compatible versions:

```bash
conda create -n mspcnet python=3.10 -y
conda activate mspcnet
# Select a matching supported CUDA build from https://pytorch.org/get-started/locally/
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

Add your datasets under `code/data/` and preserve the actual `.list` splits
(see [`docs/DATA_SPLITS.md`](docs/DATA_SPLITS.md)). The repository does not ship
medical images.

Training commands (require real paired volumes and list files):

```bash
python new/train_MSPCNet.py --dataset_name LA --labelnum 8 --gpu 0
python new/train_MSPCNet.py --dataset_name Pancreas --labelnum 6 --gpu 0
python new/train_MSPCNet.py --dataset_name ToothFairy --labelnum 13 --gpu 0
```

**Important**: these are entry-point demonstrations of the uploaded training
script. They are not proof that this file reproduces the uploaded checkpoints.
Actual results depend on transforms and source-image layout.

Checkpoint evaluation (user must supply dataset with test.list and ground truth):

```bash
python scripts/evaluate_checkpoint.py --checkpoint checkpoints/LA/10pct.pth --dataset LA --data-dir code/data/LA --patch-size 112 112 80
python scripts/evaluate_checkpoint.py --checkpoint checkpoints/Pancreas/10pct.pth --dataset Pancreas --data-dir code/data/Pancreas --patch-size 96 96 96
python scripts/evaluate_checkpoint.py --checkpoint checkpoints/IAN/10pct.pth --dataset ToothFairy --data-dir code/data/ToothFairy_Dataset/alpha --patch-size 128 128 128
```

These metric implementations use image axis order `(D,H,W)` and any spacing
provided by the reader (HDF5/NPY defaults to **1 voxel**, not 1 mm).
A genuine HD95/ASD **in mm** requires the original physical spacing and
alignment conventions. They are **not** estimated here.

## Directory layout

```text
new/train_MSPCNet.py            Uploaded trainer
new/networks/                Uploaded model + inference compatibility loader
new/losses/                  TAP / DEAC / SDM-BR
code/dataloaders/            New reader / transforms / batch sampler
code/utils/test_patch.py      New 3D sliding-window evaluator
code/data/                  Allowed split lists (no images)
checkpoints/LA/             10%,20% checkpoints
checkpoints/Pancreas/       10%,20% checkpoints
checkpoints/IAN/            10%,20% checkpoints
scripts/                    Checkpoint audit, split check, evaluation
```

## Checkpoint provenance

See [`checkpoints/manifest.json`](checkpoints/manifest.json) for the six SHA256 values
and differences in the auxiliary edge head architecture. `IAN/20pct.pth`
contains a different edge head from the public training source and is loaded
for **segmentation-only inference** with the auxiliary edge branch ignored.
The uploaded weights are not modified or re-trained by this release.

## Publication / permissions

Code availability wording for the paper **after a real public repository is
live**: "The implementation and model checkpoints are publicly available at
[the verified GitHub repository URL]." Until it is live, do not use this claim.

The uploaded source describes itself as deriving from DuCiSC. Before
publishing, verify the original code license and comply with attribution
and redistribution obligations. No license has been guessed for this project.

## Training diagnostics

`new/train_MSPCNet.py` logs its normalized optimization objective using
`loss.backward()`; it separately records `loss_all`, an unnormalized display
sum. Those are not identical. Recheck the manuscript's description against
the final script before stating that the displayed `loss_all` was optimized.

## GitHub release

Checkpoints are approximately 36 MB each, six files in total. The repository
provides `.gitattributes` for Git LFS. Install it before the first push:

```bash
git lfs install
git init
git add .
git commit -m "Release MSPC-Net source and checkpoints"
git branch -M main
# Create a GitHub repo via your authenticated account, then:
git remote add origin https://github.com/YOUR_USERNAME/MSPC-Net.git
git push -u origin main
```

DO NOT publish a URL until that actual repository exists and contains the code.

## Provided split-list inventory (2026-11-03)

- **LA:** The experiment states 80 train / 20 test, but full ID lists were not supplied. Add the original `code/data/LA/train.list` and `test.list`.
- **NIH-Pancreas:** Full `code/data/Pancreas/train.list` (0001–0062) and `test.list` (0063–0082), plus `labeled_6.list` / `labeled_12.list` are present.
- **BTCV:** `code/data/BTCV/lists/train.list` and `test.list` represent the supplied 24/6 DuCiSC `sample_XXXX` split, while `sample_to_case.csv` records the exact corresponding case IDs. Neither a 9/12-label MSPC-Net protocol nor an 18/12 MagicNet alternative is verified by these files.
- **ToothFairy/IAN:** The supplied `lists/val.list` (8 IDs) and `lists/test.list` (15 IDs) are present; ordered 138-case `lists/train.list` is still required.

See `docs/DATA_SPLITS.md` and `docs/SPLIT_STATUS_CN.md` for the verified details.
