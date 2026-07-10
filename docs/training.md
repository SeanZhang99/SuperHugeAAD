# Training Guide

This page covers training workflows, monitoring, checkpointing, and advanced usage.

---

## Basic Training

### DNN Training

```bash
cd train
python main.py
```

**Default behavior:**
- Model: SSMamba (`models/ssmamba.yaml`)
- Task: `dnn_envelope_reconstruction` with `ContrastivePearsonLoss`
- CV: Leave-one-trial-out, 4 folds
- Epochs: 100 (with early stopping, patience=10)
- Optimizer: AdamW (lr=5e-4, weight_decay=5e-4)
- LR schedule: ReduceLROnPlateau

**To switch models**, edit `main.py` line 41:

```python
# SSMamba (default)
model_config = os.path.join(project_path, "configs", "models", "ssmamba.yaml")

# VLAAI
model_config = os.path.join(project_path, "configs", "models", "vlaai.yaml")

# Deformer
model_config = os.path.join(project_path, "configs", "models", "deformer.yaml")

# SimpleCNN
model_config = os.path.join(project_path, "configs", "models", "simple_cnn.yaml")
```

### Linear Model Training

```bash
cd train
python kulavgc.py
```

**Default behavior:**
- Model: Wiener Filter (`models/wf.yaml`)
- Task: `linear_envelope_reconstruction`
- CV: Leave-one-subject-out, 12 folds
- CPU, 1 epoch, no checkpointing

---

## Enabling Tasks and CV Strategies

Edit `train/configs/task_config.yaml` to toggle tasks and cross-validation:

```yaml
dnn_envelope_reconstruction:
  tasks:
    pcc_diff:
      enable: 1        # Contrastive Pearson Loss (on)
    pcc:
      enable: 1        # Standard Pearson Loss (on)
  cross_validation:
    leave_one_trial_out:
      enable: 1        # LOTO (on)
    leave_one_subject_out:
      enable: 0        # LOSO (off)
    leave_one_dataset_out:
      enable: 0        # LODO (off)
```

`MultiRunCLI` will generate experiments for all enabled combinations:
```
dnn_envelope_reconstruction-pcc_diff-loto
dnn_envelope_reconstruction-pcc-loto
```

And for each, iterate through all fold combinations: `(val=0, test=1), (val=0, test=2), (val=0, test=3), (val=1, test=0), ...`

---

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/tb_logs --port 6006
```

Open `http://localhost:6006` in your browser.

### CSV Logs

```bash
ls logs/csv_logs/
# Look for directories named: <model_name>/<experiment_name>/<hash>/<window_length>/
```

### Live GPU Monitoring

The `nvitop` package (included as dependency) can be used to monitor GPU:

```bash
nvitop
```

### Console Output

The `FancyProgressBar` callback provides a rich progress bar with:
- Current epoch / total epochs
- Batch progress within epoch
- Loss value (real-time)

Dataset summary is printed at startup (if `summary_verbose: true`):
```
Dataset Summary
┌──────┬────────────┬──────────┬──────────┐
│ Set  │ # Subjects │ # Trials │ # Samples│
├──────┼────────────┼──────────┼──────────┤
│ Train│    42      │   320    │  12800   │
│ Val  │    12      │    80    │   3200   │
│ Test │    12      │    80    │   3200   │
└──────┴────────────┴──────────┴──────────┘
```

---

## Checkpointing

### Automatic

`ModelCheckpoint` callback saves:
- `best` checkpoint (lowest `val/loss`)
- Top-2 checkpoints
- `last` checkpoint

Checkpoints are saved under `logs/tb_logs/<experiment_path>/checkpoints/`.

### Resume Training

To resume from a saved `MultiRunCLI` state:

```python
cli = MultiRunCLI(
    ...,
    cli_checkpoint_path="path/to/cli_ckpt.pkl",
)
cli.run()
```

This restores the experiment iterator state, skipping already-completed experiments.

### Load Model from Checkpoint

To test a saved model checkpoint without retraining:

```python
cli = MultiRunCLI(
    ...,
    model_checkpoint_path="path/to/checkpoint.ckpt",
)
cli.run()
```

This skips `trainer.fit()` and directly runs validation + testing.

---

## Advanced: Custom Experiment

### Adding a New Task

1. Add to `task_config.yaml`:

```yaml
my_custom_task:
  type: regression
  general:
    data:
      init_args:
        dataset_class: superhuge.data.datasets.EegRegressionBaseDataset
        n_folds: 5
        metadata_fields: ["mel"]
    model:
      class_path: superhuge.model.interface.RegressionInterface
      init_args:
        num_audio_features:
          mel: 128
  tasks:
    mse:
      enable: 1
      model:
        init_args:
          loss:
            class_path: superhuge.model.loss.MSELoss
  cross_validation:
    leave_one_subject_out:
      enable: 1
      data:
        init_args:
          meta_group_func: superhuge.data.metadata_processing.loso
```

2. Run `python main.py` -- the task will be automatically picked up.

### Using Transforms

Uncomment transforms in `data_config.yaml`:

```yaml
transform:
  - class_path: superhuge.data.transforms.ZScore
    init_args:
      apply_prob: 1.0
      when: before_slicing
      whom: [eeg]
  - class_path: superhuge.data.transforms.Filter
    init_args:
      Wn: [1., 32.]
      btype: bandpass
      order: 5
      fs: ${init_args.fs}
      when: before_slicing
      whom: eeg
```

The `ZScore` statistics are fit on the **training set only** during `EegDataset.__init__()`, then synchronized to validation and test sets via `sync_transform_stats()`.

### Multi-GPU Training

Set in `trainer_config.yaml`:

```yaml
trainer:
  accelerator: gpu
  devices: 2           # or [0, 1] for specific GPUs
  strategy: ddp        # DistributedDataParallel
```

### Mixed Precision

```yaml
trainer:
  precision: 16-mixed   # or "bf16-mixed"
```

---

## Evaluation Metrics

### Regression (Envelope Reconstruction)

Logged metrics (per epoch, per stage):

| Metric | Description |
|--------|-------------|
| `loss` | Main loss value |
| `a_pcc` | Pearson correlation with attended speaker envelope |
| `u1_pcc` | PCC with first unattended speaker |
| `u1_pcc_diff` | PCC difference (attended - unattended) |
| `acc_by_pcc` | Accuracy: which speaker has highest PCC? |
| `f1_by_pcc` | Binary F1 for attended vs unattended |

### Classification

| Metric | Description |
|--------|-------------|
| `loss` | Cross-entropy loss |
| `accuracy` | Classification accuracy |
| `auc` | Area under ROC curve |
| `macro_f1` | Macro-averaged F1 score |

---

## Experiment Output Structure

```
logs/
├── tb_logs/
│   └── <model_name>/
│       └── <experiment_name>/
│           └── <hash>/
│               └── <window_length>/
│                   ├── events.out.tfevents.*
│                   └── checkpoints/
│                       ├── last.ckpt
│                       └── epoch=XX-step=XX.ckpt
├── csv_logs/
│   └── <same hierarchy>/
│       └── metrics.csv
└── (cli_ckpt.pkl -- pickled MultiRunCLI state)
```
