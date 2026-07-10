# Architecture Overview

This page explains the system architecture of SuperHugeAAD: how the components fit together, the data flow, and the design principles.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         train/main.py                            │
│                    (Entry Point + MultiRunCLI)                    │
└──────────────┬───────────────────────────────────┬───────────────┘
               │                                   │
               ▼                                   ▼
┌──────────────────────────────┐   ┌───────────────────────────────┐
│       TaskConfigParser       │   │        MultiRunCLI            │
│  • Reads task_config.yaml    │   │  • Generates exp configs      │
│  • Merges general/task/cv    │   │  • Iterates fold combinations │
│  • Yields CLI arg lists      │   │  • Runs train/val/test        │
└──────────────────────────────┘   └───────────┬───────────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────┐
                    │                          │                  │
                    ▼                          ▼                  ▼
        ┌──────────────────┐    ┌──────────────────┐    ┌────────────────┐
        │   DInterface     │    │    MInterface    │    │    Trainer     │
        │ (LightningData   │    │  (Lightning      │    │  (Lightning    │
        │  Module)          │    │   Module)        │    │   Trainer)     │
        └────────┬─────────┘    └────────┬─────────┘    └────────────────┘
                 │                       │
                 ▼                       ▼
        ┌──────────────────┐    ┌──────────────────┐
        │   EegDataset     │    │   Model (NN)     │
        │  (torch Dataset)  │    │  VLAAI/Deformer/ │
        │                   │    │  SSMamba/WF/...  │
        └────────┬─────────┘    └──────────────────┘
                 │
                 ▼
        ┌──────────────────┐
        │   .npy files     │
        │   metadata.pkl   │
        └──────────────────┘
```

---

## Core Components

### 1. Entry Points

Two main entry points under `train/`:

**`main.py`** — DNN model training:
- Loads SSMamba (default; switchable to any DNN config)
- Uses full `trainer_config.yaml`
- GPU training with early stopping and checkpointing

**`kulavgc.py`** — Linear model training:
- Loads Wiener Filter config
- Uses `linear_trainer.yaml` overrides (CPU, 1 epoch)
- LOSO cross-validation by default

Both use the same `MultiRunCLI` infrastructure.

### 2. MultiRunCLI

Located at `superhuge/utils/multi_run_cli.py`. Orchestrates the entire experiment workflow:

```python
class MultiRunCLI:
    def __init__(self, *args, task_config_path, model_checkpoint_path=None, cli_checkpoint_path=None)
    def run(self, verbose=True, save_config=True, extra_experiment_name="") -> dict
```

**Execution flow:**
1. `TaskConfigParser.generate_configs()` produces CLI arg lists for each (task × CV) combination
2. `ExperimentStates.__next__()` iterates through `(val_fold_idx, test_fold_idx)` combinations
3. For each experiment:
   - Creates a `NamedParamsCLI(LightningCLI)` with merged args
   - Links data properties (`fs`, `window_length`, `num_channels`) to model
   - Runs `cli.trainer.fit()`, `validate()`, `test()`
   - Accumulates results per key
   - Pickles `MultiRunCLI` state for resumability

**`NamedParamsCLI`** extends `LightningCLI` with:
- Cross-component argument linking (e.g., `data.fs` → `model.init_args.model_common_args.fs`)
- Automatic experiment path computation from `model_name + experiment_name + experiment_hash + window_length`
- Deterministic seeding from config hash

### 3. DInterface (Data Module)

Located at `superhuge/data/interface/data_interface.py`. A `LightningDataModule` that:

```python
class DInterface(pl2.LightningDataModule):
    def __init__(self, dataset_class, dataloader_args, root_path, window_length, fs, ...)
    def create_datasets(self)
    def load_metadata(self, dataset_class, metafile_path, metadata_fields) -> Metadata
    def filt_metadata(self, metadata, meta_filter_func) -> Metadata
```

**Data loading pipeline:**
```
metadata.pkl ──► load_metadata() ──► filt_metadata() ──► meta_group_func() ──► EegDataset()
```

- `load_metadata()`: Reads `{root_path}/{preproc_stage}/meta/metadata.pkl` and converts entries to `MetadataElement` objects
- `filt_metadata()`: Applies `MetadataFilterComposer` (dataset/subject/trial selection, task-specific filters)
- `meta_group_func()`: Splits into train/val/test using CV strategy
- `create_datasets()`: Instantiates `EegDataset` for each split

### 4. EegDataset

Located at `superhuge/data/datasets/eeg_dataset.py`. A `torch.utils.data.Dataset` that:

```
__init__()
  │
  ├── _validate_files()          # Ensure all files have metadata
  ├── _cache_and_prepare_segments()  # Pre-compute all valid window start indices
  └── _update_and_fit_stats_transform()  # Fit ZScore/PCA on training data

__getitem__(idx)
  │
  ├── load_data(idx)
  │   ├── np.load(file_path, mmap_mode="r")
  │   ├── transform(eeg, when="before_slicing")
  │   ├── Slice window: eeg[start : start + window_length * fs]
  │   └── transform(eeg, when="before_returning")
  └── Return {"meta": dict, "eeg": ndarray}
```

**Memory optimization:** Automatically decides whether to cache data in RAM based on available memory and total file size (caches if total < 80% available RAM).

**Segment caching:** Pre-computes all valid window start indices at init time, enabling O(1) `__getitem__()` lookups.

Subclasses:
- `EegRegressionBaseDataset` — additionally loads speech features (`env`, `mel`)
- `EegClassifyBaseDataset` — loads classification labels

### 5. MInterface (Model Wrapper)

Located at `superhuge/model/interface/model_interface.py`. A `LightningModule` that:

```python
class MInterface(pl2.LightningModule, ABC):
    def __init__(self, model_class, model_args, model_common_args, loss, ...)
    def forward(self, data: dict) -> tuple[Tensor, ...]
    def training_step(self, batch, batch_idx) -> Tensor
    def configure_optimizers(self)
    @abstractmethod
    def get_stats(self, outputs, targets, *, meta) -> dict
```

**Key design decisions:**

- **`configure_input()`** — Introspects `model.forward()` signature to determine required inputs (`eeg`, `env`, `label`, `meta`, etc.). No manual specification needed.
- **`configure_output()`** — Introspects return type annotations to determine output structure.
- **Multi-dataset batching** — `training_step()` iterates over per-dataset sub-batches in `batch`, applies loss separately, and logs per-dataset metrics.
- **Loss wrapping** — Supports single loss, multi-loss (with weights), and per-class weights for classification.

Subclasses:
- **`RegressionInterface`** — Envelope reconstruction. Computes PCC, PCC-diff, accuracy-by-PCC, F1.
- **`LinearRegressionInterface`** — `LinearInterface` + `RegressionInterface` for linear models.
- **`ClassifyInterface`** — Classification. Computes accuracy, AUC, macro F1.
- **`ChannelMapping1DRegressionInterface`** / **`ChannelMapping2DRegressionInterface`** — With spatial channel pre-mapping.

### 6. Model Architectures

All models are standard `torch.nn.Module` subclasses. They receive inputs through `MInterface.forward()` and return tensors. See [Models](models.md) for detailed architecture descriptions.

### 7. Cross-Validation

See [Cross-Validation](cross-validation.md) for details on LOTO, LOSO, LODO, and within-trial strategies.

---

## Data Flow: End-to-End

```
1. User runs: python main.py
2. MultiRunCLI.__init__():
   └── TaskConfigParser reads task_config.yaml
       └── Generates [(cli_args, experiment_name), ...]

3. MultiRunCLI.run():
   └── For each experiment:
       ├── NamedParamsCLI:
       │   ├── Instantiates DInterface
       │   │   ├── load_metadata("metadata.pkl")
       │   │   ├── filt_metadata(MetadataValueSelector(dataset_id=[2,4,5,6,10]))
       │   │   ├── meta_group_func = loto (n_folds=4)
       │   │   └── EegRegressionBaseDataset(files=splits["train"])
       │   │       └── Pre-computes valid window indices
       │   │       └── Fits ZScore stats (if enabled)
       │   └── Instantiates RegressionInterface(VLAAI, ContrastivePearsonLoss, AdamW)
       │       └── MInterface.__init__()
       │           ├── Instantiates VLAAI(**model_args, **model_common_args)
       │           ├── configure_input() → ["eeg", "env", "meta"]
       │           ├── configure_output() → ["eeg", "env"]
       │           └── Generates torchinfo summary
       ├── trainer.fit(model, datamodule)
       │   └── For each epoch:
       │       └── DInterface.train_dataloader() → DataLoader(EegRegressionBaseDataset, collate_fn=collect_multidataset)
       │           └── For each batch:
       │               ├── collect_multidataset groups items by dataset_id
       │               ├── MInterface.training_step():
       │               │   └── For each dataset's sub-batch:
       │               │       ├── forward(data) → (eeg_hat, env)
       │               │       ├── ContrastivePearsonLoss(eeg_hat, env)
       │               │       └── get_stats(eeg_hat, env) → {pcc, pcc_diff, acc, f1}
       │               └── Backward + optimizer step
       ├── trainer.validate(model, datamodule, ckpt_path="best")
       └── trainer.test(model, datamodule, ckpt_path="best")

4. Results returned: {"val/pcc": 0.XX, "test/pcc": 0.XX, ...}
```

---

## Design Principles

1. **Config-driven, not code-driven.** Every experiment is defined in YAML. No hardcoded hyperparameters.

2. **Separation of concerns.** Data loading (`DInterface` + `EegDataset`), model logic (`torch.nn.Module`), training orchestration (`MInterface` + `LightningCLI`), and experiment management (`MultiRunCLI`) are decoupled.

3. **Introspection over configuration.** `MInterface` auto-detects model inputs/outputs via `inspect.signature()`. Reduces boilerplate.

4. **Multi-dataset native.** `collect_multidataset` collation function groups batch items by `dataset_id`, enabling per-dataset loss computation and metric logging in a single batch.

5. **Reproducibility.** Every experiment gets a deterministic hash seed from its config. Logs include model version hash.

6. **Resumability.** `MultiRunCLI` state is pickled after each experiment. Resume with `cli_checkpoint_path`.
