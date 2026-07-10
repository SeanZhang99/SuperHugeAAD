# Cross-Validation

SuperHugeAAD supports four cross-validation strategies, all defined in `superhuge/data/metadata_processing/group.py`.

---

## Strategy Comparison

| Strategy | Function | What's Held Out | Best For |
|----------|----------|-----------------|----------|
| **LOTO** | `loto()` | Individual trials | Within-subject generalization |
| **LOSO** | `loso()` | Entire subjects | Across-subject generalization |
| **LODO** | `lodo()` | Entire datasets | Across-dataset generalization |
| **Within-Trial** | `within_trial()` | Time segments | Temporal generalization |

---

## LOTO: Leave-One-Trial-Out

```python
def loto(metadata, test_fold_idx, val_fold_idx, n_folds, seed=42) -> CrossValidationEntry
```

**Algorithm:**
1. Group trials by `(dataset_id, subject_id)`
2. For each subject, shuffle trials and distribute evenly across folds
3. Hold out one fold for validation, one for testing, rest for training

**Use case:** Evaluating how well a model generalizes to unseen trials from the same subjects.

**Configuration:**
```yaml
meta_group_func: superhuge.data.metadata_processing.loto
n_folds: 4    # 4-fold trial-level CV
```

**Note:** Trials from the same subject appear in multiple folds, so LOTO does not test subject generalization.

---

## LOSO: Leave-One-Subject-Out

```python
def loso(metadata, test_fold_idx, val_fold_idx, n_folds, seed=42) -> CrossValidationEntry
```

**Algorithm:**
1. Group subjects by dataset
2. Shuffle subjects and distribute evenly across folds
3. All trials from held-out subjects go to val/test

**Use case:** Evaluating cross-subject generalization — the most common AAD evaluation setting.

**Configuration:**
```yaml
meta_group_func: superhuge.data.metadata_processing.loso
n_folds: 12   # 12-fold subject-level CV
```

> **Important:** `n_folds` should be ≤ the number of subjects. If `n_folds` > number of subjects, some folds will be empty.

---

## LODO: Leave-One-Dataset-Out

```python
def lodo(metadata, test_fold_idx, val_fold_idx, n_folds, seed=42) -> CrossValidationEntry
```

**Algorithm:**
1. Shuffle datasets
2. Distribute datasets evenly across folds
3. All subjects and trials from held-out datasets go to val/test

**Use case:** Testing whether a model trained on some datasets generalizes to completely unseen datasets — the most rigorous evaluation.

**Configuration:**
```yaml
meta_group_func: superhuge.data.metadata_processing.lodo
n_folds: 5    # Number of dataset groups
```

> **Important:** The number of datasets must be ≥ `n_folds` for meaningful LODO.

---

## Within-Trial

```python
def within_trial(metadata, test_fold_idx, val_fold_idx, n_folds, seed=42) -> CrossValidationEntry
```

**Algorithm:**
1. All trials go to train/val/test
2. Instead of splitting trials, splits the **time axis** of each trial:
   - Validation partition: `[val_fold_idx/n_folds, (val_fold_idx+1)/n_folds)` of each trial
   - Test partition: `[test_fold_idx/n_folds, (test_fold_idx+1)/n_folds)` of each trial
   - Training uses the complement (and optionally reject ranges)

**Use case:** Testing temporal generalization within the same trial. Useful when the number of trials is very limited.

**Configuration:**
```yaml
meta_group_func: superhuge.data.metadata_processing.within_trial
n_folds: 5
```

**Return format (special):**
```python
{
    "train": all_trials,
    "val": all_trials,
    "test": all_trials,
    "train_reject_range": (val_range, test_range),  # Exclude these from training
    "val_accept_range": (val_range,),                # Only use this portion
    "test_accept_range": (test_range,),              # Only use this portion
}
```

The `train_reject_range` prevents data leakage by excluding the val/test time regions from training, while `val_accept_range` and `test_accept_range` restrict val/test to their respective time segments.

---

## How MultiRunCLI Handles Folds

`MultiRunCLI.ExperimentStates.__prepare_fold_idx()` generates all `(val_fold_idx, test_fold_idx)` combinations:

```python
# If val_fold_idx not specified in CLI: use all [0..n_folds-1]
# If test_fold_idx not specified in CLI: use all [0..n_folds-1]
# Product of all (val, test) combinations where val != test
```

Example for `n_folds=4`:
```
(val=0, test=1), (val=0, test=2), (val=0, test=3),
(val=1, test=0), (val=1, test=2), (val=1, test=3),
(val=2, test=0), (val=2, test=1), (val=2, test=3),
(val=3, test=0), (val=3, test=1), (val=3, test=2)
```

That's 12 experiments for a single task. You can limit this via CLI:

```bash
python main.py --data.init_args.val_fold_idx 0 --data.init_args.test_fold_idx 1
```

---

## Under the Hood

### collect_dataset_subject_trials()

```python
def collect_dataset_subject_trials(metadata: Metadata):
```

Organizes metadata into a nested dict:
```python
{
    dataset_id: {
        subject_id: [(entry, metadata_element), ...],
        ...
    },
    ...
}
```

This structure is used by all CV functions as the starting point.

### divide_sets()

```python
def divide_sets(all_folds, n_folds, test_fold_idx, val_fold_idx):
```

Common function that takes a dict of `{fold_idx: [entries]}` and returns `(train_set, val_set, test_set)`:
- Test = `all_folds[test_fold_idx]`
- Val = `all_folds[val_fold_idx]`
- Train = all other folds, sorted

### Adding a Custom CV Strategy

1. Define a function with the same signature as `loto`/`loso`/`lodo`:

```python
def my_cv_strategy(
    metadata: Metadata,
    test_fold_idx: int,
    val_fold_idx: int,
    n_folds: int,
    seed: int = 42,
    **kwargs,
) -> CrossValidationEntry:
    ...
    return {"train": train_set, "val": val_set, "test": test_set}
```

2. Reference it in `task_config.yaml`:

```yaml
meta_group_func: my_module.my_cv_strategy
```

The function will be called with the current `test_fold_idx`, `val_fold_idx`, and `n_folds` from the CLI args.
