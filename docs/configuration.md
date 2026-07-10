# Configuration Reference

SuperHugeAAD is fully driven by YAML configuration files under `train/configs/`. This page documents every config file and its parameters.

---

## Overview

| File | Purpose | Required |
|------|---------|----------|
| `data_config.yaml` / `wsl_data_config.yaml` | Data paths, dataset selection, windowing, transforms | Yes |
| `task_config.yaml` | Task definitions, loss functions, CV strategies | Yes |
| `trainer_config.yaml` | PyTorch Lightning Trainer settings | Yes |
| `linear_trainer.yaml` | Trainer overrides for linear models | For linear only |
| `optimizer_config.yaml` | Optimizer and LR scheduler settings | Yes |
| `models/*.yaml` | Per-model architecture hyperparameters | Yes |

Configs are merged hierarchically by `TaskConfigParser`: **general → task → cross_validation**, with deeper keys overriding shallower ones.

---

## Data Configuration (`data_config.yaml`)

```yaml
class_path: superhuge.data.interface.DInterface
init_args:
  # Data paths
  root_path: E:/derivatives/SuperHuge     # Root of preprocessed data
  preproc_stage: preprocessed             # Subdirectory under root_path

  # Data module settings
  dataset_class: superhuge.data.datasets.EegDataset
  fs: 128                                 # Sampling rate (Hz)
  window_length: 10                       # Window length (seconds)
  overlap: 1                              # Overlap ratio (1 = no overlap)

  # Dataloader
  dataloader_args:
    batch_size: 32
    drop_last: false
    num_workers: 2
    persistent_workers: true
    pin_memory: true
    prefetch_factor: 8

  # Dataset selection filter
  meta_filter_func:
    - class_path: superhuge.data.metadata_filters.MetadataValueSelector
      init_args:
        attribute_name: dataset_id
        attribute_value:
          - 2    # DTU
          - 4    # KUL
          - 5    # KUL-AV-GC
          - 6    # NJU
          - 10   # sparKULee

  # Transforms (optional pipeline)
  transform:
    null
    # Uncomment to enable:
    # - class_path: superhuge.data.transforms.ZScore
    #   init_args:
    #     apply_prob: 1.0
    #     when: before_slicing
    #     whom: [eeg, env]
    # - class_path: superhuge.data.transforms.Filter
    #   init_args:
    #     Wn: [1., 32.]
    #     btype: bandpass
    #     order: 5
    #     fs: ${init_args.fs}
    #     when: before_returning
    #     whom: eeg

  summary_verbose: true
```

### Transforms

Available transforms (in `superhuge.data.transforms`):

| Transform | Description |
|-----------|-------------|
| `ZScore` | Per-channel z-score normalization |
| `Filter` | Butterworth bandpass/bandstop filter |
| `Resample` | Signal resampling to a new sampling rate |
| `Scale` | Scaling with pre-computed factors from `scaling_factor.pkl` |
| `ChannelSelection` | Select subset of EEG channels |
| `ChannelMask` | Random channel masking |
| `PCA` | Principal Component Analysis |
| `RiemannianAlign` | Riemannian alignment |
| `ZScoreAlign` | Z-score alignment |
| `Clipper` | Value clipping |
| `PadSpeech` | Speech feature padding |
| `TimeShift` | Temporal shifting |

Each transform has:
- `apply_prob`: Probability of applying (0.0 to 1.0)
- `when`: `"before_slicing"` (on full signal) or `"before_returning"` (on windowed segment)
- `whom`: `"eeg"`, `"env"`, `"mel"`, or a list

### WSL/Linux Data Config (`wsl_data_config.yaml`)

Same structure but with Unix paths and typically higher `num_workers`:

```yaml
init_args:
  root_path: /mnt/e/derivatives/SuperHuge
  dataloader_args:
    batch_size: 64
    num_workers: 4
```

The correct config is auto-selected in `main.py` via `os.name`:
- `"nt"` → `data_config.yaml`
- Everything else → `wsl_data_config.yaml`

---

## Task Configuration (`task_config.yaml`)

```yaml
task_name:
  type: regression                      # or "classification"
  general:                              # Base config shared across tasks
    data:
      init_args:
        dataset_class: ...              # EegRegressionBaseDataset or EegClassifyBaseDataset
        n_folds: 4                      # Number of CV folds
        metadata_fields: ["env"]        # Speech feature fields to load
    model:
      class_path: ...                   # RegressionInterface, ClassifyInterface, etc.

  tasks:                                # Individual task definitions
    pcc_diff:
      enable: 1                         # 1 = enabled, 0 = disabled
      model:
        init_args:
          loss:
            class_path: superhuge.model.loss.ContrastivePearsonLoss

  cross_validation:                     # CV strategy definitions
    leave_one_trial_out:
      enable: 1
      data:
        init_args:
          meta_group_func: superhuge.data.metadata_processing.loto
```

### Task Types

**`regression`** — Envelope reconstruction:
- Dataset class: `EegRegressionBaseDataset`
- Interface: `RegressionInterface` or `LinearRegressionInterface`
- Key model args: `num_audio_features` (e.g., `env: 1`)

**`classification`** — Spatial attention classification:
- Dataset class: `EegClassifyBaseDataset`
- Interface: `ClassifyInterface` or `LinearClassifyInterface`
- Key model args: `num_class`

### Loss Functions

| Loss | Use Case |
|------|----------|
| `PearsonLoss` | Maximize PCC with attended speaker |
| `ContrastivePearsonLoss` | Maximize PCC with attended, minimize with unattended |
| `AbsPearsonLoss` | Absolute-value variant |
| `ContrastiveAbsPearsonLoss` | Absolute-value contrastive variant |
| `MSELoss` | Mean squared error regression |
| `ContrastiveMSELoss` | Contrastive MSE |
| `CrossEntropyLoss` | Classification |
| `TwoStagePearsonLoss` | Switches from Pearson to contrastive after N epochs |
| `F1Score`, `AUCROC` | Classification metrics (not losses) |

---

## Trainer Configuration (`trainer_config.yaml`)

```yaml
trainer:
  accelerator: auto           # "auto", "cpu", "gpu"
  devices: auto               # "auto" or int
  strategy: auto              # "auto", "ddp", etc.
  max_epochs: 100

  callbacks:
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args:
        logging_interval: epoch
    - class_path: lightning.pytorch.callbacks.EarlyStopping
      init_args:
        monitor: val/loss
        patience: 10
        mode: min
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        monitor: val/loss
        save_top_k: 2
        mode: min
    - class_path: superhuge.utils.fancy_progress_bar.FancyProgressBar
      init_args:
        refresh_rate: 5

  logger:
    - class_path: TensorBoardLogger
      init_args:
        save_dir: ./logs/tb_logs
    - class_path: CSVLogger
      init_args:
        save_dir: ./logs/csv_logs
```

### Linear Trainer Overrides (`linear_trainer.yaml`)

Linear models (Wiener Filter, CCA) fit in one epoch on CPU:

```yaml
trainer:
  accelerator: cpu
  max_epochs: 1
  enable_progress_bar: false
  enable_checkpointing: false
  callbacks: null
  num_sanity_val_steps: 0
```

---

## Optimizer Configuration (`optimizer_config.yaml`)

```yaml
init_args:
  optimizer_class: torch.optim.AdamW
  optimizer_args:
    lr: 5.e-4
    weight_decay: 5.e-4
  lr_scheduler_class: torch.optim.lr_scheduler.ReduceLROnPlateau
  lr_scheduler_args:
    mode: min
    factor: 0.5
    patience: 5
    min_lr: 1.e-9
```

The `MInterface.configure_optimizers()` method also applies **separate weight decay** to bias/norm parameters (no decay) vs other parameters.

---

## Model Configurations

### VLAAI (`models/vlaai.yaml`)

```yaml
class_path: superhuge.model.interface.MInterface
init_args:
  model_class: superhuge.model.pure_cnn.VLAAI
  model_args:
    extractor_args:
      class_path: superhuge.model.pure_cnn.ExtractorParams
      init_args:
        kernel_sizes: [8]
        num_kernels: [256, 256, 256, 128, 128]
        num_layers: 5
    nb_blocks: 4
    output_context_kernel_size: 32
    dropout: 0.5
    use_skip: false
```

### Deformer (`models/deformer.yaml`)

```yaml
class_path: superhuge.model.interface.MInterface
init_args:
  model_class: superhuge.model.former.Deformer
  model_args:
    num_kernels: 8
    temporal_kernel_size: 13
    mha_depth: 2
    mha_embed_dim: 32
    mha_num_heads: 2
    ff_hidden_dim: 64
    dropout: 0.3
```

### SSMamba (`models/ssmamba.yaml`)

```yaml
class_path: superhuge.model.interface.MInterface
init_args:
  model_class: superhuge.model.ssmamba.SSMamba
  model_args:
    d_inner: 256
    n_head: 2
    n_layers: 1
    dropout: 0.5
    within_sub_num: 110
  summary_at_cuda: true
```

### Wiener Filter (`models/wf.yaml`)

```yaml
class_path: superhuge.model.interface.MInterface
init_args:
  model_class: superhuge.model.linear.WienerFilter
  model_args:
    use_lwcov: true        # Ledoit-Wolf shrinkage
    pre_lag: 0.0           # Pre-stimulus lag (seconds)
    post_lag: 0.4          # Post-stimulus lag (seconds)
    l2: 0.0                # L2 regularization
```

### Other Models

| Config | Model Class | Key Details |
|--------|-------------|-------------|
| `simple_cnn.yaml` | `SimpleCNN` | `temporal_kernel_size=17`, `num_kernels=5` |
| `lsm_cnn.yaml` | `LSM_CNN` | `lsm_chan_dim=8`, `cnn_num_layers=1` |
| `rebok_vlaai.yaml` | `VLAAI_ws` | Weight-sharing VLAAI variant |
| `rebok_deformer.yaml` | `Deformer_ws` | Weight-sharing Deformer variant |

---

## Config Merging Logic

The `TaskConfigParser._deep_merge_dicts()` method merges configurations hierarchically:

```
general.data ──► task.data ──► cv.data  ──► merged data config
general.model ─► task.model ─► cv.model ─► merged model config
```

Arrays are **concatenated** (not overridden). Scalars and nested dicts are **overridden** by deeper levels.
