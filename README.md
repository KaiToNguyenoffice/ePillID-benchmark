# ePillID Benchmark (Modernized)

A benchmark for developing and evaluating computer vision models for **pill identification** using multi-head metric learning. Originally published at CVPR 2020 Workshop by Usuyama et al. (Microsoft Research), this fork has been modernized to run on current Python/PyTorch versions with additional features.

## Overview

ePillID is a **low-shot fine-grained recognition** benchmark:

- **13,000 images** representing **9,804 appearance classes** (two sides for 4,902 pill types)
- **Reference images** (professional, controlled) vs **consumer images** (real-world settings)
- Most classes have only **1 reference image** per side, making it a challenging low-shot setting

The best performing approach is a **multi-head metric learning** model combining a CNN encoder with bilinear pooling and multiple loss functions.

## What's New (Modernized Fork)

| Feature | Original | Modernized |
|---------|----------|------------|
| Python | 3.6 | 3.10+ |
| PyTorch | 0.4.1 | 2.x |
| Azure ML | Required | Optional (runs locally) |
| Backbones | ResNet, DenseNet | + EfficientNet, ConvNeXt, ViT, Swin Transformer |
| Losses | CE + ArcFace + Contrastive + Triplet + Focal | + Circle Loss |
| Augmentation | imgaug only | + Albumentations backend |
| Evaluation | Torch, Annoy | + FAISS support |
| Label Encoder | Fixed sklearn pickle | Incremental (expandable) |
| Deployment | None | ONNX export |
| Data Paths | Hardcoded Linux paths | Auto-detection with double-nested zip fix |

## Project Structure

```
ePillID-benchmark-master/
├── src/
│   ├── train_nocv.py              # Single-fold training entry point
│   ├── train_cv.py                # Cross-validation training entry point
│   ├── multihead_trainer.py       # Training loop
│   ├── pillid_datasets.py         # PyTorch datasets and balanced sampler
│   ├── image_augmentators.py      # Augmentation (imgaug / albumentations)
│   ├── metrics.py                 # Evaluation metrics (AP, MAP, GAP, PR)
│   ├── metric_test_eval.py        # Evaluators (metric embedding / logit)
│   ├── metric_utils.py            # Hard negative mining for pairs/triplets
│   ├── sanitytest_eval.py         # Base evaluator and dataloader factory
│   ├── classif_utils.py           # Dataset path utilities
│   ├── arguments.py               # CLI argument definitions
│   ├── aml_wrapper.py             # Azure ML abstraction (optional)
│   ├── data_path_utils.py         # Data path validation and auto-fix
│   ├── label_utils.py             # Incremental label encoder
│   ├── export_onnx.py             # ONNX model export
│   ├── run_train.py               # Training wrapper with logging
│   ├── configs/
│   │   └── params.json            # Default training config
│   └── models/
│       ├── multihead_model.py     # MultiheadModel (embedding + classification heads)
│       ├── embedding_model.py     # Backbone + embedding MLP
│       ├── losses.py              # All loss functions (incl. Circle Loss)
│       ├── margin_linear.py       # ArcFace margin layer
│       ├── focal_loss.py          # Focal loss
│       ├── fast_MPN_COV_wrapper.py
│       └── fast-MPN-COV/          # Bilinear pooling library (forked)
├── docker/
│   └── conda/epillidpy36_env.yml
├── ePillID_tutorial_colab.ipynb
└── LICENSE
```

## Model Architecture

```
Input Image (224x224)
       │
   ┌───▼───────────────────┐
   │  Backbone (CNN/ViT)   │  ResNet, EfficientNet, Swin, ConvNeXt, ...
   │  + Pooling Layer      │  GAvP, MPNCOV, CBP, BCNN
   └───┬───────────────────┘
       │ features
   ┌───▼───────────────────┐
   │  Embedding MLP        │  Linear(2048→1000) → BN → ReLU → Linear(1000→2048) → Tanh
   │  + Dropout            │
   └───┬───────────────────┘
       │ embedding (2048-d)
       ├──────────────────────────┬─────────────────────┐
       ▼                         ▼                      ▼
  BinaryHead               MarginHead            Online Mining
  (CE logits)             (ArcFace logits)    (pairs / triplets)
       │                         │                      │
  Cross-Entropy             ArcFace Loss      Contrastive + Triplet
  + Focal Loss                                  + Circle Loss
       │                         │                      │
       └─────────┬───────────────┘──────────────────────┘
                 ▼
          Weighted Sum → Total Loss → Backprop
```

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone and enter directory
cd "D:\few shot\ePillID-benchmark-master"

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Install core dependencies
pip install torch torchvision                          # CPU version
# OR for GPU (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install scikit-learn pandas numpy tqdm matplotlib pillow

# Install augmentation backend (pick one or both)
pip install albumentations                             # Recommended
pip install imgaug                                     # Legacy (requires numpy<2.0)

# Optional
pip install faiss-cpu                                  # Fast nearest-neighbor search
pip install onnx                                       # ONNX model export/validation
```

### Data

Download from the [GitHub Releases page](https://github.com/usuyama/ePillID-benchmark/releases) and extract. Expected structure:

```
<data_root_dir>/
├── classification_data/          # Pill images
├── folds/
│   └── pilltypeid_nih_sidelbls0.01_metric_5folds/
│       └── base/
│           ├── *_all.csv         # All images
│           ├── *_0.csv ... *_4.csv  # 5 folds
│           └── label_encoder.pickle
└── all_labels.csv
```

If the zip creates a double-nested directory (e.g. `ePillID_data/ePillID_data/`), the training scripts will auto-detect and correct this.

## Usage

All commands run from the `src/` directory with venv activated.

```bash
cd src
```

### Quick Test (1 epoch, small model)

```bash
python train_nocv.py \
    --data_root_dir "<path_to_data>" \
    --appearance_network resnet18 \
    --max_epochs 1 \
    --batch_size 16 \
    --aug_backend albumentations
```

### Single-Fold Training

```bash
python train_nocv.py \
    --data_root_dir "<path_to_data>" \
    --appearance_network resnet50 \
    --max_epochs 300 \
    --aug_backend albumentations
```

### Cross-Validation Training

```bash
python train_cv.py \
    --data_root_dir "<path_to_data>" \
    --appearance_network resnet50 \
    --max_epochs 300 \
    --aug_backend albumentations
python train_cv.py --appearance_network resnet18 --max_epochs 10 --data_root_dir "D:\few shot\ePillID-benchmark-master\ePillID_data\ePillID_data" --aug_backend albumentations
```

### Using Modern Backbones

```bash
# EfficientNet V2
python train_nocv.py --data_root_dir "<path>" --appearance_network efficientnet_v2_s --aug_backend albumentations

# Swin Transformer
python train_nocv.py --data_root_dir "<path>" --appearance_network swin_t --aug_backend albumentations

# ConvNeXt
python train_nocv.py --data_root_dir "<path>" --appearance_network convnext_tiny --aug_backend albumentations
```

### Using Circle Loss

```bash
python train_nocv.py --data_root_dir "<path>" --circle_w 1.0 --aug_backend albumentations
```

### Using Config File

```bash
python train_nocv.py --load_config configs/params.json --data_root_dir "<path>"
```

### ONNX Export

```bash
python export_onnx.py \
    --model_path "<path_to_model>.pth" \
    --appearance_network resnet50 \
    --n_classes 4902 \
    --output embedding_model.onnx
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root_dir` | `/mydata` | Root directory containing `folds/` and `classification_data/` |
| `--appearance_network` | `resnet50` | Backbone model (see supported list below) |
| `--max_epochs` | `300` | Maximum training epochs |
| `--batch_size` | `48` | Mini-batch size |
| `--optimizer` | `adam` | Optimizer: `adam`, `adamw`, `momentum`, `adamdelta` |
| `--weight_decay` | `0.0` | Weight decay for optimizer |
| `--init_lr` | `1e-4` | Initial learning rate |
| `--aug_backend` | `imgaug` | Augmentation: `imgaug` or `albumentations` |
| `--metric_evaluator_type` | `cosine` | Distance metric: `cosine`, `euclidean`, `ann`, `faiss` |
| `--pooling` | `GAvP` | Pooling layer (classic CNNs only): `GAvP`, `MPNCOV`, `CBP`, `BCNN` |

### Loss Weights

| Argument | Default | Description |
|----------|---------|-------------|
| `--ce_w` | `1.0` | Cross-entropy loss weight |
| `--arcface_w` | `0.1` | ArcFace loss weight |
| `--contrastive_w` | `1.0` | Contrastive loss weight |
| `--triplet_w` | `1.0` | Triplet loss weight |
| `--focal_w` | `0.0` | Focal loss weight |
| `--circle_w` | `0.0` | Circle loss weight |

### Supported Backbones

**Classic (with bilinear pooling support):**
`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`,
`densenet121`, `densenet161`, `densenet201`

**Modern (torchvision pretrained):**
`efficientnet_b0`-`b7`, `efficientnet_v2_s/m/l`,
`convnext_tiny/small/base/large`,
`vit_b_16/32`, `vit_l_16/32`,
`swin_t/s/b`, `swin_v2_t/s/b`,
`maxvit_t`

## Output

Training produces:

| Output | Location |
|--------|----------|
| Model weights | `<data_root>/classification_results/<run_id>/<fold>.pth` |
| Predictions CSV | `src/outputs/eval_predictions_<fold>.csv` |
| Metrics | Logged via Azure ML or printed to console |

## Citation

```bibtex
@inproceedings{usuyama2020epillid,
  title={ePillID Dataset: A Low-Shot Fine-Grained Benchmark for Pill Identification},
  author={Usuyama, Naoto and Delgado, Natalia Larios and Hall, Amanda K and Lundin, Jessica},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2020}
}
```

## Acknowledgments

- Original ePillID benchmark by [Naoto Usuyama et al.](https://github.com/usuyama/ePillID-benchmark)
- NIH NLM [Pill Image Recognition Challenge](https://pir.nlm.nih.gov/challenge/) and [Pillbox](https://pillbox.nlm.nih.gov/statistics.html) datasets
- [fast-MPN-COV](https://github.com/jiangtaoxie/fast-MPN-COV) for bilinear pooling

## License

This dataset and software are released for research purposes only. See [LICENSE](LICENSE).
