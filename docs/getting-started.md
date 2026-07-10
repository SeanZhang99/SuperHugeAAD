# Getting Started

## Prerequisites

- **Python >= 3.11**
- **PyTorch** with CUDA support (recommended for DNN training; CPU-only works for linear models)
- **Preprocessed data** from the [SuperPrepare](https://github.com/zymzhang/SuperPrepare) pipeline
- Git (for cloning the repository)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/zymzhang/SuperHugeAAD.git
cd SuperHugeAAD
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / WSL
source .venv/bin/activate
```

### 3. Install the Package

```bash
pip install -e .
```

This installs the `superhuge` package in editable mode with all dependencies:
- `torch`, `torchvision`, `torchaudio`
- `lightning[extra]` (PyTorch Lightning)
- `numpy`, `scipy`, `einops`
- `pydantic`, `pyyaml`
- `librosa`, `rich`, `torchinfo`
- `keras` (for the `KERAS_BACKEND=torch` setting)

### 4. Verify Installation

```python
import superhuge
print(superhuge.__version__)  # Should print 0.1.0
```

## Prepare the Data

SuperHugeAAD expects preprocessed data produced by the SuperPrepare pipeline. The expected directory structure:

```
<root_path>/preprocessed/
├── meta/
│   └── metadata.pkl           # Pickled dict: entry → MetadataElement
├── eeg/
│   ├── dataset-001-subject-001-trial-001.npy   # (signal_length, num_channels)
│   ├── dataset-001-subject-001-trial-002.npy
│   └── ...
└── stimuli/
    └── env/
        ├── dataset-001-subject-001-trial-001_env.npy
        └── ...
```

**Key points:**
- EEG files are 2D NumPy arrays: `(signal_length, num_channels)` as `float32`
- The `metadata.pkl` file maps `DatasetSubjectTrialEntry` strings to metadata dictionaries
- Each metadata entry must contain: `dataset_id`, `subject_id`, `trial_id`, `fs`, `num_channel`, `signal_length`, `channel_infos`

## First Training Run

### Step 1: Configure Data Path

Edit `train/configs/data_config.yaml`:

```yaml
init_args:
  root_path: E:/derivatives/SuperHuge    # Set to your data directory
```

On WSL/Linux, use `train/configs/wsl_data_config.yaml` (automatically selected based on `os.name`).

### Step 2: Select Datasets

In the same file, uncomment the datasets you want to use:

```yaml
meta_filter_func:
  - class_path: superhuge.data.metadata_filters.MetadataValueSelector
    init_args:
      attribute_name: dataset_id
      attribute_value:
        - 4   # KUL
        - 6   # NJU
```

Available dataset IDs: 1 (AHU), 2 (DTU), 3 (Estart), 4 (KUL), 5 (KUL-AV-GC), 6 (NJU), 7 (NUS), 8 (PKU), 9 (PKU-NBD), 10 (sparKULee).

### Step 3: Run Training

```bash
cd train
python main.py
```

**What happens:**
1. `TaskConfigParser` reads `task_config.yaml` and generates all enabled experiment configs
2. `MultiRunCLI` creates a `DInterface` (data module) and `MInterface` (model wrapper)
3. For each `(val_fold_idx, test_fold_idx)` combination:
   - Data is split into train/val/test using the configured CV strategy
   - Model is trained (up to 100 epochs with early stopping, patience=10)
   - Best checkpoint is loaded for validation and testing
   - Metrics (PCC, F1, accuracy) are accumulated

### Step 4: View Results

```bash
tensorboard --logdir logs/tb_logs
```

CSV logs are also available:
```bash
cat logs/csv_logs/<experiment_path>/metrics.csv
```

## Quick Test: Linear Model

If you want a quick smoke test without a GPU:

```bash
python kulavgc.py
```

This runs the Wiener Filter on CPU with leave-one-subject-out CV (single epoch). It should complete in a few minutes depending on data size.

## Troubleshooting

### "File not found: metadata.pkl"

Ensure `root_path` in `data_config.yaml` points to the correct directory. The framework looks for:
```
{root_path}/{preproc_stage}/meta/metadata.pkl
```
Default `preproc_stage` is `"preprocessed"`.

### "CUDA out of memory"

Reduce `batch_size` in `data_config.yaml` (default: 32). Try 16 or 8.

### "No module named 'superhuge'"

Make sure you ran `pip install -e .` from the repository root. Verify with `pip list | grep superhuge`.

### "num_workers > 0" on Windows

Set `num_workers: 0` in the dataloader config if you encounter multiprocessing issues on Windows.

### SSMamba Import Error

SSMamba (`superhuge.model.ssmamba`) requires Linux/WSL. On native Windows, it won't import. Use other models (VLAAI, Deformer, SimpleCNN) instead.

## Next Steps

- Read the [Configuration Reference](configuration.md) to understand all config options
- Check the [Models](models.md) page for architecture details
- See [Training Guide](training.md) for advanced training workflows
