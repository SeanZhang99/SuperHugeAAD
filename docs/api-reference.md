# API Reference

This page documents the key classes and functions in the `superhuge` package.

---

## Data Module

### DInterface

**File:** `superhuge.data.interface.DInterface`

```python
class DInterface(pl2.LightningDataModule):
    def __init__(
        self,
        /,
        *,
        dataset_class: type[EegDataset],
        dataloader_args: dict[str, Any],
        root_path: str,
        window_length: int | float,
        fs: int,
        meta_filter_func: MetadataFilter | Sequence[MetadataFilter] | None = None,
        add_meta_filter_func: MetadataFilter | Sequence[MetadataFilter] | None = None,
        meta_filter_func_args: Sequence | None = None,
        meta_group_func: Callable | None = None,
        test_fold_idx: int = 0,
        val_fold_idx: int = 1,
        n_folds: int = 5,
        overlap: int = 1,
        metadata_fields: list[MetadataField] | None = None,
        transform: Transform | Sequence[Transform] | None = None,
        preproc_stage: str | None = None,
        summary_verbose: bool = False,
        dataset_args: dict[str, Any] | None = None,
    )
```

The main LightningDataModule. Handles metadata loading, filtering, cross-validation splitting, and dataset creation.

**Properties:**
- `trainset`, `valset`, `testset` — `EegDataset` instances
- `batch_size` — get/set dataloader batch size
- `fs` — effective sampling rate (after resampling)
- `window_length` — window length in seconds
- `num_channels` — number of EEG channels (requires `ChannelSelection` transform)

**Methods:**
- `train_dataloader()`, `val_dataloader()`, `test_dataloader()` — standard PyTorch Lightning
- `summary(verbose=True)` — print dataset statistics

---

## Datasets

### EegDataset

**File:** `superhuge.data.datasets.EegDataset`

```python
class EegDataset(Dataset):
    def __init__(self, **kwargs)
    def __getitem__(self, idx: int) -> Mapping
    def __len__(self) -> int
```

**Required kwargs:**
| Key | Type | Description |
|-----|------|-------------|
| `eeg_path` | `str` | Path to directory containing `.npy` files |
| `files` | `Sequence[str]` | List of file basenames (without `.npy`) |
| `metadata` | `Metadata` | Dict mapping entry → metadata |
| `metadata_fields` | `list[MetadataField]` | Fields to include in output |
| `window_length` | `int \| float` | Window length in seconds |
| `fs` | `int` | Sampling rate |
| `overlap` | `int` | Overlap ratio (default 1 = no overlap) |
| `stage` | `Literal["train", "val", "test"]` | Dataset stage |

**Optional kwargs:**
| Key | Type | Description |
|-----|------|-------------|
| `transform` | `TransformComposer \| None` | Transform pipeline |
| `accept_range` | `Sequence[tuple[float, float]]` | Valid signal range (default: whole trial) |
| `reject_range` | `Sequence[tuple[float, float]]` | Rejected signal ranges |
| `save_on_memory` | `bool \| None` | Force/enable memory caching |

**Properties:**
- `files` — list of file basenames
- `valid_segments_cache` — `(file_names, offsets, valid_start_indices)` tuple
- `segment_length` — `fs * window_length`

### EegRegressionBaseDataset

**File:** `superhuge.data.datasets.EegRegressionBaseDataset`

```python
class EegRegressionBaseDataset(EegDataset):
    # Additionally loads speech features (env, mel)
    # __getitem__() returns {"meta": ..., "eeg": ..., "env": ..., "mel": ...}
```

### EegClassifyBaseDataset

**File:** `superhuge.data.datasets.EegClassifyBaseDataset`

```python
class EegClassifyBaseDataset(EegDataset):
    # Additionally loads classification labels
    # __getitem__() returns {"meta": ..., "eeg": ..., "label": ...}
```

---

## Model Interfaces

### MInterface

**File:** `superhuge.model.interface.MInterface`

```python
class MInterface(pl2.LightningModule, ABC):
    def __init__(
        self,
        /,
        *,
        model_class: type[nn.Module] | Callable[..., nn.Module],
        model_args: dict[str, Any],
        model_common_args: ModelInputArgs,
        loss: _Loss | Sequence[_Loss],
        multiclass_loss_weights: Sequence[float] | None = None,
        multiloss_weights: Sequence[float] | None = None,
        optimizer_class: type[Optimizer] = AdamW,
        optimizer_args: dict[str, Any] | None = None,
        lr_scheduler_class: type[LRScheduler] | None = None,
        lr_scheduler_args: dict[str, Any] | None = None,
        ckpt_path: str | None = None,
        log_grad: bool | None = None,
        log_norm: bool | None = None,
        summary_verbose: bool | None = None,
        summary_at_cuda: bool | None = False,
        get_stats_fn: Callable | None = None,
        diagnostic: bool | None = None,
    )
```

Base LightningModule for all models. Provides:
- Automatic input/output configuration via `inspect.signature()`
- Loss function wrapping (single or multi-loss)
- Optimizer with separate weight decay for bias/norm layers
- Multi-dataset batching in `training_step()`
- `torchinfo` model summary at init

**Abstract method:**
```python
@abstractmethod
def get_stats(self, outputs: Tensor, targets: Tensor, /, *, meta: dict) -> dict[str, Tensor]:
    """Compute validation/test metrics. Called during val/test steps."""
```

### RegressionInterface

**File:** `superhuge.model.interface.RegressionInterface`

```python
class RegressionInterface(MInterface):
    def __init__(self, /, *, num_audio_features: NumAudioFeaturesMixin, **kwargs)

    def get_stats(
        self, y_pred: Tensor, y_true: Tensor, /, *, meta: dict
    ) -> dict[str, Tensor]:
        # Computes:
        # - Per-speaker PCC (a_pcc, u1_pcc, u2_pcc, ...)
        # - PCC differences (u1_pcc_diff, u2_pcc_diff, ...)
        # - Accuracy by PCC (acc_by_pcc)
        # - Binary F1 (f1_by_pcc)
```

`num_audio_features` specifies which speech features to use:
```python
num_audio_features = {"env": 1}       # Envelope only (1 channel)
num_audio_features = {"mel": 128}     # Mel-spectrogram (128 channels)
num_audio_features = {"env": 1, "mel": 128}  # Both
```

### LinearRegressionInterface

**File:** `superhuge.model.interface.LinearRegressionInterface`

```python
class LinearRegressionInterface(LinearInterface, RegressionInterface):
    # Combines linear model support with regression metrics
```

### ClassifyInterface

**File:** `superhuge.model.interface.ClassifyInterface`

```python
class ClassifyInterface(MInterface):
    # Computes accuracy, AUC, macro F1 for classification tasks
```

### RegressionInterfaceWithEnvDump

**File:** `superhuge.model.interface.RegressionInterfaceWithEnvDump`

```python
class RegressionInterfaceWithEnvDump(RegressionInterface):
    # Dumps y_pred, y_true, trial_id, dataset_id, subject_id
    # to {log_dir}/test_env/ during testing for offline analysis
```

---

## Cross-Validation

### loto

**File:** `superhuge.data.metadata_processing.loto`

```python
def loto(
    metadata: Metadata,
    test_fold_idx: int,
    val_fold_idx: int,
    n_folds: int,
    seed: int = 42,
    **kwargs: Any,
) -> CrossValidationEntry:
```

Leave-one-trial-out. Splits trials within each subject across folds.

### loso

**File:** `superhuge.data.metadata_processing.loso`

```python
def loso(
    metadata: Metadata,
    test_fold_idx: int,
    val_fold_idx: int,
    n_folds: int,
    seed: int = 42,
    **kwargs: Any,
) -> CrossValidationEntry:
```

Leave-one-subject-out. Splits entire subjects across folds.

### lodo

**File:** `superhuge.data.metadata_processing.lodo`

```python
def lodo(
    metadata: Metadata,
    test_fold_idx: int,
    val_fold_idx: int,
    n_folds: int,
    seed: int = 42,
    **kwargs: Any,
) -> CrossValidationEntry:
```

Leave-one-dataset-out. Splits entire datasets across folds.

### within_trial

**File:** `superhuge.data.metadata_processing.within_trial`

```python
def within_trial(
    metadata: Metadata,
    test_fold_idx: int,
    val_fold_idx: int,
    n_folds: int,
    seed: int = 42,
    **kwargs: Any,
) -> CrossValidationEntry:
```

Temporal split within each trial. Returns `train_reject_range`, `val_accept_range`, `test_accept_range`.

---

## Loss Functions

### PearsonLoss

**File:** `superhuge.model.loss.PearsonLoss`

```python
class PearsonLoss(_Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor, *args, **kwargs) -> Tensor:
        # loss = -corrcoef(y_pred, y_true[..., 0], dim=1).mean(dim=1)
```

Negative Pearson correlation with the attended speaker.

### ContrastivePearsonLoss

**File:** `superhuge.model.loss.ContrastivePearsonLoss`

```python
class ContrastivePearsonLoss(_Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor, *args, **kwargs) -> Tensor:
        # pcc = corrcoef(y_pred, y_true, dim=1).mean(dim=1)
        # loss = -pcc[:, 0] + mean(pcc[:, 1:]) / (n_speakers - 1)
```

Maximizes PCC with attended speaker while penalizing PCC with unattended speakers.

### pearson_corrcoef

**File:** `superhuge.model.loss.regression.pearson_loss.pearson_corrcoef`

```python
def pearson_corrcoef(y_pred: Tensor, y_true: Tensor, dim: int) -> Tensor:
```

Computes Pearson correlation coefficient along a specified dimension. Handles tensors of arbitrary dimensionality (>= 2D). Uses `divide_no_nan` for numerical stability.

### Other Loss Functions

| Class | Description |
|-------|-------------|
| `MSELoss` | Mean squared error |
| `ContrastiveMSELoss` | Contrastive MSE |
| `AbsPearsonLoss` | Negative absolute PCC |
| `ContrastiveAbsPearsonLoss` | Contrastive absolute PCC |
| `SumPearsonLoss` | Summed PCC across features and speakers |
| `TwoStagePearsonLoss` | Switches from Pearson to 90% Pearson + 10% Contrastive |
| `CrossEntropyLoss` | Classification cross-entropy |

---

## Transforms

### Transform (ABC)

**File:** `superhuge.data.transforms.Transform`

```python
class Transform(ABC):
    apply_prob: float = 1.0
    when: str   # "before_slicing" or "before_returning"
    whom: str | Sequence[str]   # "eeg", "env", "mel", etc.

    @abstractmethod
    def __call__(self, data, /, *, meta, when, whom):
        ...
```

### ZScore

**File:** `superhuge.data.transforms.ZScore`

```python
class ZScore(StatisticalTransform):
    def __init__(self, apply_prob=1.0, when="before_slicing", whom=["eeg"]):
```

Per-channel z-score normalization. Fits mean/std on training data.

### Filter

**File:** `superhuge.data.transforms.Filter`

```python
class Filter(Transform):
    def __init__(self, Wn, btype, order, fs, apply_prob=1.0, when="before_slicing", whom=["eeg"]):
```

Butterworth filter. `Wn`: cutoff frequency(ies), `btype`: `"bandpass"` or `"bandstop"`.

### Resample

**File:** `superhuge.data.transforms.Resample`

```python
class Resample(Transform):
    def __init__(self, old_fs, new_fs, apply_prob=1.0, when="before_slicing", whom=["eeg"]):
```

Resample signal to `new_fs`.

### ChannelSelection

**File:** `superhuge.data.transforms.ChannelSelection`

```python
class ChannelSelection(Transform):
    def __init__(self, channels, apply_prob=1.0, when="before_slicing", whom=["eeg"]):
```

Select a subset of channels by index or name (uses `CHANNEL1D_ENUM`).

---

## Metadata

### MetadataElement

**File:** `superhuge.data.metadata_processing.MetadataElement`

```python
class MetadataElement(BaseModel, extra="allow"):
    dataset_id: int | None = 0
    subject_id: int | None = 0
    trial_id: int | None = 0
    num_channel: int | None = 0
    signal_length: int | None = 0
    fs: int | None = 0
    dataset_name: str | None = None
    channel_infos: Mapping[int, Mapping[str, Any]]
```

### MetadataValueSelector

**File:** `superhuge.data.metadata_filters.MetadataValueSelector`

```python
class MetadataValueSelector(MetadataFilter):
    def __init__(self, attribute_name: str, attribute_value: Any):
```

Filter metadata entries by a field value. Supports exact match or sequence membership.

---

## CLI and Config

### MultiRunCLI

**File:** `superhuge.utils.MultiRunCLI`

```python
class MultiRunCLI:
    def __init__(
        self,
        *args: str,
        task_config_path: str,
        model_checkpoint_path: str | None = None,
        cli_checkpoint_path: str | None = None,
    )
    def run(
        self,
        verbose: bool = True,
        save_config: bool = True,
        extra_experiment_name: str = "",
    ) -> dict[str, float]:
```

Orchestrates multi-experiment training. Arguments are passed through to `LightningCLI`.

### TaskConfigParser

**File:** `superhuge.utils.TaskConfigParser`

```python
class TaskConfigParser:
    def __init__(self, config_path: str)
    def generate_configs(self) -> Generator[tuple[list[str], str], Any, None]:
```

Reads `task_config.yaml`, generates CLI argument lists for all enabled task × CV combinations.

### NamedParamsCLI

**File:** `superhuge.utils.NamedParamsCLI`

```python
class NamedParamsCLI(LightningCLI):
    # Links data properties (fs, window_length, num_channels) to model
    # Adds --experiment_name, --model_name, --experiment_hash
    # Computes logger path from name + hash
```

---

## MATLAB Interop

### create_data_interface

**File:** `superhuge.data.interface.create_data_interface`

```python
def create_data_interface(
    root_path: str,
    window_length: int,
    fs,
    classify: bool = False,
    regression: bool = False,
    metadata_fields: list[str] | None = None,
    select_dataset: int | Sequence[int] | None = None,
    select_subject: int | Sequence[int] | None = None,
    select_trial: int | Sequence[int] | None = None,
    meta_filter_func_args: Sequence | None = None,
    leave_one_out: str | None = None,
    test_fold_idx: int = 0,
    val_fold_idx: int = 1,
    n_folds: int = 5,
    overlap: int = 1,
    preproc_stage: str | None = None,
    bandpass_wn: Sequence[float] | None = None,
    refs: int | None = None,
    zscore: bool = False,
    **kwargs,
) -> DInterface:
```

Designed to be called from MATLAB via Python-MATLAB bridge. Returns a configured `DInterface`.

---

## Channel Reference

**File:** `superhuge.utils.channel_enum`

```python
NUM_ELECTRODES = 74

class CHANNEL1D_ENUM(Enum):
    A1=0, A2=1, AF3=2, AF4=3, AF7=4, AF8=5, AFz=6,
    C1=7, C2=8, C3=9, C4=10, C5=11, C6=12,
    CP1=13, CP2=14, CP3=15, CP4=16, CP5=17, CP6=18, CPz=19, Cz=20,
    F1=21, F2=22, F3=23, F4=24, F5=25, F6=26, F7=27, F8=28, Fz=29,
    FC1=30, ...  # 74 electrodes total from 10-20 system

class CHANNEL2D_ENUM(Enum):
    # Maps channel names to 2D grid positions for spatial models
```
