# Data Pipeline

This page documents the data loading pipeline: metadata, datasets, transforms, and MATLAB interoperability.

---

## Expected Data Format

The data is expected to be preprocessed by the [SuperPrepare](https://github.com/zymzhang/SuperPrepare) pipeline and stored in this structure:

```
{root_path}/
└── {preproc_stage}/          # Default: "preprocessed"
    ├── meta/
    │   └── metadata.pkl      # Pickled Metadata dictionary
    ├── eeg/
    │   ├── dataset-001-subject-001-trial-001.npy
    │   ├── dataset-001-subject-001-trial-002.npy
    │   └── ...
    └── stimuli/
        ├── env/
        │   ├── dataset-001-subject-001-trial-001_env.npy
        │   └── ...
        ├── mel/
        │   └── ...
        └── wav2vec2/
            └── ...
```

### EEG .npy Files

- **Shape:** `(signal_length, num_channels)` — time × channels
- **Dtype:** `float32`
- **Naming:** `dataset-{id:03d}-subject-{id:03d}-trial-{id:03d}.npy`
- Loaded via `np.load(file_path, mmap_mode="r")` for memory efficiency

### Speech Feature .npy Files

- **Shape:** `(signal_length, num_features)` — time × features
- **Naming:** `{entry}_{feature_type}.npy` (e.g., `dataset-001-subject-001-trial-001_env.npy`)
- Feature types: `env` (envelope, 1 channel), `mel` (mel-spectrogram, 128 channels), `wav2vec2` (768 channels)

### metadata.pkl

A pickled Python dictionary:

```python
{
    "dataset-001-subject-001-trial-001": {
        "dataset_id": 1,
        "dataset_name": "AHU",
        "subject_id": 1,
        "trial_id": 1,
        "fs": 128,
        "num_channel": 64,
        "signal_length": 38400,
        "channel_infos": {
            0: {"name": "Fz", "x": 0.0, "y": 0.5, ...},
            ...
        },
        "env": "dataset-001-subject-001-trial-001_env.npy",
        "mel": "dataset-001-subject-001-trial-001_mel.npy",
    },
    ...
}
```

---

## Metadata Processing

### MetadataElement

**File:** `superhuge/data/metadata_processing/data.py`

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

Subclasses add task-specific fields:
- `RegressionMetadataElement` → adds `env`, `mel`, `wav2vec2` paths
- `ClassifyMetadataElement` → adds `label`

### Metadata Filters

**File:** `superhuge/data/metadata_filters/`

Filters select which trials to include:

| Filter | Purpose |
|--------|---------|
| `MetadataValueSelector` | Include entries where a field matches a value |
| `MetadataValueExcluder` | Exclude entries where a field matches a value |
| `MetadataFilterComposer` | Chain multiple filters |
| `RegressionMetadataFilter` (abstract) | Filter by available speech features |
| `EnvFilter` | Include only entries with envelope data |
| `MelFilter` | Include only entries with mel-spectrogram data |
| `AudioFilter` | Include entries with any audio feature |
| `ClassifyMetadataFilter` (abstract) | Filter by classification label |
| `BinaryLeftRight` | 2-class: left vs right |
| `BinaryFrontRear` | 2-class: front vs rear |
| `FourClass` | 4-class spatial directions |
| `EightClass` | 8-class spatial directions |

Example: selecting specific datasets and subjects:

```yaml
meta_filter_func:
  - class_path: superhuge.data.metadata_filters.MetadataValueSelector
    init_args:
      attribute_name: dataset_id
      attribute_value: [2, 4, 6]
  - class_path: superhuge.data.metadata_filters.MetadataValueSelector
    init_args:
      attribute_name: subject_id
      attribute_value: [1, 2, 3]
```

---

## Datasets

### EegDataset (Base)

**File:** `superhuge/data/datasets/eeg_dataset.py`

The base dataset class for loading EEG data.

**Key features:**
- **Memory-aware caching:** Auto-detects if total file size < 80% of available RAM and caches in memory
- **Segment pre-computation:** All valid window start indices are computed at init; O(1) `__getitem__()`
- **Accept/reject ranges:** Specify which portions of each trial to use/exclude
- **Two-phase transforms:** `before_slicing` (full signal) and `before_returning` (windowed segment)

**`__getitem__()` returns:**
```python
{
    "meta": {
        "dataset_id": int,
        "subject_id": int,
        "trial_id": int,
        "fs": int,
        "num_channel": int,
        "signal_length": int,
        ...
    },
    "eeg": np.ndarray  # (window_length * fs, num_channels) as float32
}
```

### EegRegressionBaseDataset

**File:** `superhuge/data/datasets/eeg_regression_base_dataset.py`

Extends `EegDataset` for envelope reconstruction. Additionally loads speech features:

```python
{
    "meta": {...},
    "eeg": ndarray,     # (T, C)
    "env": ndarray,     # (T, 1)   -- if metadata_fields includes "env"
    "mel": ndarray,     # (T, 128) -- if metadata_fields includes "mel"
}
```

### EegClassifyBaseDataset

**File:** `superhuge/data/datasets/eeg_classify_base_dataset.py`

Extends `EegDataset` for spatial attention classification:

```python
{
    "meta": {...},
    "eeg": ndarray,     # (T, C)
    "label": int,       # Class label
}
```

### Multi-Dataset Collation

**File:** `superhuge/data/datasets/collect_multidataset.py`

The `collect_multidataset()` function is used as the DataLoader's `collate_fn`. It:

1. Groups batch items by `dataset_id`
2. Applies `default_collate` within each group
3. Returns `{dataset_id: collated_batch}`

This enables per-dataset loss computation and metric logging in `MInterface.training_step()`.

---

## Transforms

**Directory:** `superhuge/data/transforms/`

All transforms extend `Transform` (ABC) with `__call__(self, data, *, meta, when, whom)`. They are composed via `TransformComposer`.

### Available Transforms

| Transform | Description | Key Parameters |
|-----------|-------------|----------------|
| `ZScore` | Per-channel mean-std normalization | `apply_prob`, `when`, `whom` |
| `Filter` | Butterworth filter | `Wn`, `btype`, `order`, `fs` |
| `Resample` | Signal resampling | `old_fs`, `new_fs` |
| `Scale` | Pre-computed scaling | `root_path`, `scaling_factor_path` |
| `ChannelSelection` | Select channel subset | `channels` (indices or names) |
| `ChannelMask` | Random channel masking | `mask_prob` |
| `PCA` | Dimensionality reduction | `n_components` |
| `RiemannianAlign` | Riemannian geometry alignment | -- |
| `ZScoreAlign` | Z-score alignment | -- |
| `Clipper` | Value clipping | `min_val`, `max_val` |
| `PadSpeech` | Speech feature padding | `pad_length` |
| `TimeShift` | Temporal shifting | `shift_range` |

### Statistical Transforms

**File:** `superhuge/data/transforms/stats_abc.py`

`ZScore` and `PCA` extend `StatisticalTransform`, which accumulates statistics during `update()` and computes final parameters during `fit()`.

**Training-only fitting:**
- Stats are accumulated on the **training set** during `EegDataset.__init__()`
- Then synced to val/test sets via `sync_transform_stats()`
- This prevents data leakage

### Transform Phases

Transforms run in two phases:

1. **`before_slicing`** — Applied to the full trial signal before windowing. Good for: filtering, resampling, z-scoring.

2. **`before_returning`** — Applied to the windowed segment. Good for: channel masking, clipping.

```

Full Signal (N × C)
    │
    ├── transform(when="before_slicing")
    │
    ├── Slice: eeg[start : start + W]
    │
    ├── transform(when="before_returning")
    │
    └── Return windowed segment (W × C)
```

---

## MATLAB Interoperability

**File:** `superhuge/data/interface/matlab_utils.py`

The `create_data_interface()` function enables MATLAB users to access the Python data pipeline:

```python
def create_data_interface(
    root_path: str,
    window_length: int,
    fs,
    classify: bool = False,
    regression: bool = False,
    select_dataset: int | Sequence[int] | None = None,
    select_subject: int | Sequence[int] | None = None,
    select_trial: int | Sequence[int] | None = None,
    leave_one_out: str | None = None,
    test_fold_idx: int = 0,
    val_fold_idx: int = 1,
    n_folds: int = 5,
    bandpass_wn: Sequence[float] | None = None,
    refs: int | None = None,
    zscore: bool = False,
    **kwargs,
) -> DInterface
```

**From MATLAB:**

```matlab
% Set up Python environment
pyenv('Version', 'C:\path\to\python.exe');

% Call create_data_interface
data_interface = py.superhuge.data.interface.matlab_utils.create_data_interface(...
    'E:/derivatives/SuperHuge', ...  % root_path
    10, ...                           % window_length
    128, ...                          % fs
    pyargs(...
        'regression', true, ...
        'select_dataset', 4, ...
        'leave_one_out', 'loto', ...
        'n_folds', 5 ...
    )...
);

% Access datasets
trainset = data_interface.trainset;
valset = data_interface.valset;
testset = data_interface.testset;
```

The convenience parameters (`bandpass_wn`, `refs`, `zscore`) automatically configure the corresponding transforms.
