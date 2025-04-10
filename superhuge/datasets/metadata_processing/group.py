from typing import Any, cast
import random
from .data import (
    DatasetSubjectTrialEntry,
    GroupingFunction,
    MetaData,
    CrossValidationEntry,
)


def leave_one_out_input_decorator(func) -> GroupingFunction:
    def wrapper(
        metadata: MetaData,
        test_fold_idx: int,
        val_fold_idx: int,
        n_folds: int,
        seed: int = 42,
        **kwargs: Any,
    ) -> CrossValidationEntry:
        return func(metadata, test_fold_idx, val_fold_idx, n_folds, seed, **kwargs)

    return cast(GroupingFunction, wrapper)


def collect_dataset_subject_trials(
    metadata: MetaData,
):
    dataset_subject_trials: dict[
        int, dict[int, list[tuple[DatasetSubjectTrialEntry, MetaData]]]
    ] = {}
    for trial_id, trial_metadata in metadata.items():
        dataset_id = trial_metadata.dataset_id
        subject_id = trial_metadata.subject_id
        assert dataset_id is not None, "Dataset ID cannot be None"
        assert subject_id is not None, "Subject ID cannot be None"
        dataset_subject_trials.setdefault(dataset_id, {}).setdefault(
            subject_id, []
        ).append((trial_id, trial_metadata))
    return dataset_subject_trials


def divide_sets(
    all_folds: dict[int, list[DatasetSubjectTrialEntry]],
    n_folds: int,
    test_fold_idx: int,
    val_fold_idx: int,
):
    test_set = set(all_folds[test_fold_idx])
    val_set = set(all_folds[val_fold_idx])
    train_set = list(
        set(
            item
            for i in range(n_folds)
            if i != test_fold_idx and i != val_fold_idx
            for item in all_folds[i]
        )
    )
    train_set.sort()
    val_set = list(val_set)
    val_set.sort()
    test_set = list(test_set)
    test_set.sort()

    return train_set, val_set, test_set


@leave_one_out_input_decorator
def loto(
    metadata: MetaData,
    test_fold_idx: int,
    val_fold_idx: int,
    n_folds: int,
    seed: int = 42,
    **kwargs: Any,
) -> CrossValidationEntry:
    assert (
        0 <= test_fold_idx < n_folds
    ), f"test_fold_idx must be in the range [0, {n_folds})"
    assert (
        0 <= val_fold_idx < n_folds
    ), f"val_fold_idx must be in the range [0, {n_folds})"
    random.seed(seed)

    dataset_subject_trials = collect_dataset_subject_trials(metadata)

    # Distribute trials evenly across folds
    all_folds = {i: [] for i in range(n_folds)}

    for dataset_id, subjects in dataset_subject_trials.items():
        for subject_id, trials in subjects.items():
            trials: list[tuple[DatasetSubjectTrialEntry, MetaData]]
            random.shuffle(trials)
            trials_per_fold = len(trials) // n_folds

            for i in range(n_folds):
                start_idx = i * trials_per_fold
                end_idx = (i + 1) * trials_per_fold if i != n_folds - 1 else len(trials)
                all_folds[i].extend(item[0] for item in trials[start_idx:end_idx])

    train_set, val_set, test_set = divide_sets(
        all_folds, n_folds, test_fold_idx, val_fold_idx
    )

    return {"train": train_set, "val": val_set, "test": test_set}


@leave_one_out_input_decorator
def loso(
    metadata: MetaData,
    test_fold_idx: int,
    val_fold_idx: int,
    n_folds: int,
    seed: int = 42,
    **kwargs: Any,
) -> CrossValidationEntry:
    random.seed(seed)

    dataset_subject_trials = collect_dataset_subject_trials(metadata)

    # Get all subjects for cross-validation
    all_folds = {i: [] for i in range(n_folds)}

    for dataset_id, subjects in dataset_subject_trials.items():
        # Shuffle the dictionary by shuffling the keys
        keys = list(subjects.keys())
        random.shuffle(keys)
        subjects = {key: subjects[key] for key in keys}
        # divide each fold
        subjects_per_fold = len(subjects) // n_folds
        for i in range(n_folds):
            start_idx = i * subjects_per_fold
            end_idx = (i + 1) * subjects_per_fold if i != n_folds - 1 else len(subjects)
            for subject_id in list(subjects.keys())[start_idx:end_idx]:
                all_folds[i].extend(item[0] for item in subjects[subject_id])

    train_set, val_set, test_set = divide_sets(
        all_folds, n_folds, test_fold_idx, val_fold_idx
    )

    return {"train": train_set, "val": val_set, "test": test_set}


@leave_one_out_input_decorator
def lodo(
    metadata: MetaData,
    test_fold_idx: int,
    val_fold_idx: int,
    n_folds: int,
    seed: int = 42,
    **kwargs: Any,
) -> CrossValidationEntry:
    random.seed(seed)

    dataset_subject_trials = collect_dataset_subject_trials(metadata)

    # Get all subjects for cross-validation
    all_folds = {i: [] for i in range(n_folds)}

    datasets = list(dataset_subject_trials.keys())
    random.shuffle(datasets)
    dataset_per_fold = len(datasets) // n_folds
    for i in range(n_folds):
        start_idx = i * dataset_per_fold
        end_idx = (i + 1) * dataset_per_fold if i != n_folds - 1 else len(datasets)
        for dataset_id in datasets[start_idx:end_idx]:
            for trial_entry in dataset_subject_trials[dataset_id].values():
                all_folds[i].extend(trial_entry[0])

    train_set, val_set, test_set = divide_sets(
        all_folds, n_folds, test_fold_idx, val_fold_idx
    )
    return {"train": train_set, "val": val_set, "test": test_set}
