# SuperHugeAAD Documentation

Welcome to the SuperHugeAAD documentation. This guide covers everything from installation to model architecture details.

## Contents

| Section | Description |
|---------|-------------|
| [Getting Started](getting-started.md) | Installation, prerequisites, first training run |
| [Configuration Reference](configuration.md) | All YAML config files explained |
| [Architecture Overview](architecture.md) | System architecture, data flow, module interactions |
| [Models](models.md) | All DNN and linear model architectures |
| [Training Guide](training.md) | How to train, monitor, and resume experiments |
| [Data Pipeline](data-pipeline.md) | Data loading, transforms, metadata, MATLAB interop |
| [Cross-Validation](cross-validation.md) | CV strategies (LOTO, LOSO, LODO, within-trial) |
| [API Reference](api-reference.md) | Key classes and functions |

## Quick Links

- **Entry point**: [`train/main.py`](../train/main.py) (DNN) / [`train/kulavgc.py`](../train/kulavgc.py) (Linear)
- **Data config**: [`train/configs/data_config.yaml`](../train/configs/data_config.yaml)
- **Task config**: [`train/configs/task_config.yaml`](../train/configs/task_config.yaml)

## Project Overview

SuperHugeAAD is a deep learning framework for **Auditory Attention Decoding (AAD)**. It reconstructs the attended speaker's speech envelope from EEG signals using both deep neural networks (CNN, Transformer, SSM) and linear models (Wiener Filter, CCA).

The framework is designed for **multi-dataset training** -- it aggregates preprocessed EEG data from 10+ different AAD datasets and provides flexible cross-validation strategies to evaluate model generalization across subjects, trials, and datasets.

### Data Flow

```
SuperPrepare Pipeline          SuperHugeAAD
─────────────────────          ────────────
Raw EEG (.bdf/.edf)            metadata.pkl
        │                            │
        ▼                            ▼
  Preprocessing          DInterface.load_metadata()
        │                      │
        ▼                      ▼
  .npy files          Metadata filtering + CV split
        │                      │
        ▼                      ▼
  metadata.pkl        EegDataset.__getitem__()
                             │
                             ▼
                      Model training (MInterface)
                             │
                             ▼
                      Evaluation metrics (PCC, F1)
```
