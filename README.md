# SuperHugeAAD

> Extending Deep Learning-based Auditory Attention Decoding to multiple datasets.

SuperHugeAAD is a deep learning framework for **Auditory Attention Decoding (AAD)** -- the task of determining which speaker a listener is attending to from EEG brain signals. Built on PyTorch Lightning, it supports multi-dataset training across 10+ EEG datasets with flexible cross-validation strategies and a variety of model architectures (CNN, Transformer, SSM, and linear baselines).

---

## Table of Contents

- [Overview](#overview)
- [Supported Models](#supported-models)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Cross-Validation Strategies](#cross-validation-strategies)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

---

## Overview

The framework addresses the core AAD tasks:

| Task | Description | Model Types |
|------|-------------|-------------|
| **Envelope Reconstruction** | Reconstruct the attended speaker's speech envelope from EEG | DNN (VLAAI, Deformer, SSMamba, etc.) + Linear (Wiener Filter, CCA) |
| **Spatial Attention Classification** | Classify which spatial direction the listener attends to | CSP, RGC, DNN classifiers |

### Key Features

- **Multi-dataset training** -- aggregate data from AHU, DTU, Estart, KUL, KUL-AV-GC, NJU, NUS, PKU, PKU-NBD, sparKULee
- **Flexible cross-validation** -- leave-one-trial-out (LOTO), leave-one-subject-out (LOSO), leave-one-dataset-out (LODO), within-trial
- **Rich transform pipeline** -- bandpass filter, z-score, resample, channel selection, PCA, Riemannian alignment
- **Experiment management** -- YAML-driven multi-run CLI with automatic fold iteration, checkpointing, and metric accumulation
- **MATLAB interoperability** -- create data interfaces directly from MATLAB via `matlab_utils.create_data_interface()`

### Relationship to SuperPrepare

This repository is the **downstream consumer** of the [SuperPrepare](https://github.com/zymzhang/SuperPrepare) preprocessing pipeline. SuperPrepare standardizes raw EEG data from multiple datasets into `.npy` files and a `metadata.pkl` file. SuperHugeAAD loads this preprocessed data for model training and evaluation.

---

## Supported Models

### Deep Neural Networks

| Model | Config File | Description |
|-------|-------------|-------------|
| **VLAAI** | `vlaai.yaml` | Multi-block CNN extractor with output context (4 blocks, 256-128 kernels) |
| **SimpleCNN** | `simple_cnn.yaml` | Minimal baseline: Conv2D + ReLU |
| **LSM-CNN** | `lsm_cnn.yaml` | Learnable Spatial Mapping CNN (2D grid projection) |
| **Deformer** | `deformer.yaml` | EEG-Deformer: CNN pre-conv + transformer encoder |
| **SSMamba** | `ssmamba.yaml` | State-space model: S4 + Bimamba + UNet + self-attention |
| **VLAAI_ws** | `rebok_vlaai.yaml` | Weight-sharing variant of VLAAI |
| **Deformer_ws** | `rebok_deformer.yaml` | Weight-sharing variant of Deformer |

### Linear Models

| Model | Config File | Description |
|-------|-------------|-------------|
| **Wiener Filter** | `wf.yaml` | Linear decoder with lagged input, Ledoit-Wolf shrinkage |
| **CCA** | -- | Canonical Correlation Analysis |
| **Filterbank CCA** | -- | Multi-band CCA |
| **Riemannian WF** | -- | Wiener Filter with Riemannian geometry |

---

## Repository Structure

```
SuperHugeAAD/
├── superhuge/                    # Main Python package
│   ├── data/                     # Data pipeline
│   │   ├── datasets/             # EegDataset, regression/classification variants
│   │   ├── interface/            # DInterface (LightningDataModule), MATLAB utils
│   │   ├── metadata_filters/     # Dataset/subject/trial selection filters
│   │   ├── metadata_processing/  # Metadata types, CV splitting functions
│   │   └── transforms/           # ZScore, Filter, Resample, PCA, ChannelSelection...
│   ├── model/                    # Model definitions
│   │   ├── pure_cnn/             # VLAAI, SimpleCNN, LSM-CNN
│   │   ├── former/               # Conformer, Deformer, EEG-Deformer
│   │   ├── ssmamba/              # S4, Bimamba, SSMamba
│   │   ├── linear/               # Wiener Filter, CCA, CSP
│   │   ├── loss/                 # Pearson, Contrastive Pearson, F1, AUC-ROC
│   │   ├── module/               # Building blocks (conv, attention, residuals)
│   │   └── interface/            # MInterface, RegressionInterface, ClassifyInterface
│   ├── utils/                    # CLI, config parser, channel enum, visualization
│   └── test/                     # Package-level tests
├── train/                        # Training entry points
│   ├── main.py                   # DNN training entry
│   ├── kulavgc.py                # Linear model training entry
│   └── configs/                  # YAML configs (data, task, trainer, models)
├── tests/                        # Test suite (pytest)
├── visualize/                    # Visualization scripts
└── pyproject.toml                # Package metadata and dependencies
```

---

## Installation

### Prerequisites

- **Python >= 3.11**
- **PyTorch** with CUDA support (recommended for DNN training)
- Preprocessed data from [SuperPrepare](https://github.com/zymzhang/SuperPrepare) pipeline

### Install from Source

```bash
git clone https://github.com/zymzhang/SuperHugeAAD.git
cd SuperHugeAAD
pip install -e .
```

### Dependencies

Core dependencies (auto-installed):

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision`, `torchaudio` | Deep learning framework |
| `lightning[extra]` | Training loop, logging, callbacks |
| `numpy`, `scipy` | Numerical computing |
| `einops` | Tensor operations |
| `pydantic` | Data validation |
| `pyyaml` | Config file parsing |
| `librosa` | Audio processing |
| `rich` | Terminal formatting |
| `torchinfo` | Model summary |

---

## Quick Start

### 1. Prepare the Data

Run the SuperPrepare pipeline to generate preprocessed data. The expected directory structure:

```
E:/derivatives/SuperHuge/
└── preprocessed/
    ├── meta/
    │   └── metadata.pkl       # Trial metadata dictionary
    ├── eeg/
    │   ├── dataset-001-subject-001-trial-001.npy
    │   └── ...
    └── stimuli/
        └── env/
            ├── dataset-001-subject-001-trial-001_env.npy
            └── ...
```

### 2. Configure the Data Path

Edit `train/configs/data_config.yaml` (Windows) or `train/configs/wsl_data_config.yaml` (WSL/Linux) to set `root_path` to your data directory:

```yaml
init_args:
  root_path: E:/derivatives/SuperHuge   # Windows
  # root_path: /mnt/e/derivatives/SuperHuge   # WSL
```

Select which datasets to include by uncommenting their IDs under `meta_filter_func`.

### 3. Train a DNN Model

```bash
cd train
python main.py
```

This runs envelope reconstruction with the **SSMamba** model using leave-one-trial-out cross-validation. The entry point:

- Reads `data_config.yaml`, `ssmamba.yaml`, `trainer_config.yaml`, `optimizer_config.yaml`, and `task_config.yaml`
- Iterates through enabled tasks + cross-validation strategies + fold combinations
- Trains, validates, and tests each experiment

### 4. Train a Linear Model

```bash
cd train
python kulavgc.py
```

This runs the **Wiener Filter** with leave-one-subject-out cross-validation (CPU-only, single epoch).

### 5. Monitor Training

TensorBoard logs are saved to `logs/tb_logs/`. Launch with:

```bash
tensorboard --logdir logs/tb_logs
```

CSV logs are also available at `logs/csv_logs/`.

---

## Configuration

All configuration is YAML-driven. The key config files:

| File | Purpose |
|------|---------|
| `data_config.yaml` | Data paths, dataset selection, window length, sampling rate, transforms |
| `task_config.yaml` | Task definitions (regression/classification), loss functions, enabled CV strategies |
| `trainer_config.yaml` | PyTorch Lightning Trainer settings (epochs, callbacks, logging) |
| `linear_trainer.yaml` | Trainer overrides for linear models (CPU, 1 epoch, no checkpointing) |
| `optimizer_config.yaml` | Optimizer and LR scheduler settings |
| `models/*.yaml` | Per-model architecture hyperparameters |

### Task Configuration Example

From `task_config.yaml`, the `dnn_envelope_reconstruction` task:

```yaml
dnn_envelope_reconstruction:
  type: regression
  general:
    data:
      init_args:
        dataset_class: superhuge.data.datasets.EegRegressionBaseDataset
        n_folds: 4
        metadata_fields: ["env"]
    model:
      class_path: superhuge.model.interface.RegressionInterface
  tasks:
    pcc_diff:
      enable: 1
      model:
        init_args:
          loss:
            class_path: superhuge.model.loss.ContrastivePearsonLoss
  cross_validation:
    leave_one_trial_out:
      enable: 1
      data:
        init_args:
          meta_group_func: superhuge.data.metadata_processing.loto
```

Tasks and CV strategies are enabled/disabled by toggling `enable: 1` / `enable: 0`. The `TaskConfigParser` generates all combinations and `MultiRunCLI` executes them sequentially.

### Switching Models

Models are selected via the `--model` flag in `main.py`:

```python
# In main.py, change the model_config path:
model_config = os.path.join(project_path, "configs", "models", "vlaai.yaml")
```

Available model configs: `vlaai.yaml`, `simple_cnn.yaml`, `lsm_cnn.yaml`, `deformer.yaml`, `ssmamba.yaml`, `wf.yaml`, `rebok_deformer.yaml`, `rebok_vlaai.yaml`.

---

## Cross-Validation Strategies

| Strategy | Function | Description |
|----------|----------|-------------|
| **LOTO** | `loto()` | Leave-one-trial-out -- splits each subject's trials across folds |
| **LOSO** | `loso()` | Leave-one-subject-out -- splits subjects across folds |
| **LODO** | `lodo()` | Leave-one-dataset-out -- splits datasets across folds |
| **Within-Trial** | `within_trial()` | Time-split within each trial (temporal cross-validation) |

The task config determines which strategies are active. `MultiRunCLI` automatically iterates through all `(val_fold_idx, test_fold_idx)` combinations for the specified `n_folds`.

---

## Documentation

Full documentation is available in the [`docs/`](docs/index.md) directory:

- [Getting Started](docs/getting-started.md)
- [Configuration Reference](docs/configuration.md)
- [Architecture Overview](docs/architecture.md)
- [Models](docs/models.md)
- [Training Guide](docs/training.md)
- [Data Pipeline](docs/data-pipeline.md)
- [Cross-Validation](docs/cross-validation.md)
- [API Reference](docs/api-reference.md)

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

---

## References

If you use this code in your research, please cite:

```bibtex
@software{zhang2026superhugeaad,
  author = {Yuanming Zhang and Yayun Liang},
  title = {SuperHugeAAD: Multi-Dataset Auditory Attention Decoding},
  year = {2026},
  url = {https://github.com/zymzhang/SuperHugeAAD},
}
```

This framework builds on the [PyTorch Lightning Template](https://github.com/miracleyoo/lightning-template) by Zhongyang Zhang.

---

## License

MIT License.
