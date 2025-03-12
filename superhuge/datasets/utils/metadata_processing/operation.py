from typing import Any, cast
import random
from .data import GroupingFunction, MetaData, CrossValidationEntry


def leave_one_out_input_decorator(func) -> GroupingFunction:
    def wrapper(
        metadata: MetaData,
        fold_index: int,
        n_folds: int,
        seed: int = 42,
        **kwargs: Any,
    ) -> CrossValidationEntry:
        return func(metadata, fold_index, n_folds, seed, **kwargs)

    return cast(GroupingFunction, wrapper)


@leave_one_out_input_decorator
def loto(
    metadata: MetaData, fold_index: int, n_folds: int, seed: int = 42, **kwargs: Any
) -> CrossValidationEntry:
    random.seed(seed)

    dataset_subject_trials = {0: {0: []}}

    # Organize trials by dataset and subject
    for trial_entry, trial_metadata in metadata.items():
        dataset_id = trial_metadata.dataset_id
        subject_id = trial_metadata.subject_id
        assert dataset_id is not None, "Dataset ID cannot be None"
        assert subject_id is not None, "Subject ID cannot be None"
        dataset_subject_trials.setdefault(dataset_id, {}).setdefault(
            subject_id, []
        ).append(trial_entry)

    # Distribute trials evenly across folds
    all_folds = {i: [] for i in range(n_folds)}

    for dataset_id, subjects in dataset_subject_trials.items():
        for subject_id, trials in subjects.items():
            random.shuffle(trials)
            trials_per_fold = len(trials) // n_folds

            for i in range(n_folds):
                start_idx = i * trials_per_fold
                end_idx = (i + 1) * trials_per_fold if i != n_folds - 1 else len(trials)
                all_folds[i].extend(trials[start_idx:end_idx])

    test_set = set(all_folds[fold_index])
    val_trials = set(all_folds[(fold_index + 1) % n_folds])
    train_trials = (
        set(trial for fold in all_folds.values() for trial in fold)
        - test_set
        - val_trials
    )
    train_trials = list(train_trials)
    train_trials.sort()
    val_trials = list(val_trials)
    val_trials.sort()
    test_set = list(test_set)
    test_set.sort()

    return {"train": train_trials, "val": val_trials, "test": test_set}


@leave_one_out_input_decorator
def loso(
    metadata: MetaData, fold_index: int, n_folds: int, seed: int = 42, **kwargs: Any
) -> CrossValidationEntry:
    random.seed(seed)

    dataset_subject_trials = {0: {0: []}}

    # Organize trials by dataset and subject
    for trial_id, trial_metadata in metadata.items():
        dataset_id = trial_metadata.dataset_id
        subject_id = trial_metadata.subject_id
        assert dataset_id is not None, "Dataset ID cannot be None"
        assert subject_id is not None, "Subject ID cannot be None"
        dataset_subject_trials.setdefault(dataset_id, {}).setdefault(
            subject_id, []
        ).append(trial_id)

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
                all_folds[i].extend(subjects[subject_id])

    test_set = set(all_folds[fold_index])
    val_set = set(all_folds[(fold_index + 1) % n_folds])
    train_set = set(
        trial_id
        for subjects in dataset_subject_trials.values()
        for trials in subjects.values()
        for trial_id in trials
    )
    train_set -= test_set | val_set

    return {"train": list(train_set), "val": list(val_set), "test": list(test_set)}


@leave_one_out_input_decorator
def lodo(
    metadata: MetaData, fold_index: int, n_folds: int, seed: int = 42, **kwargs: Any
) -> CrossValidationEntry:
    random.seed(seed)

    dataset_subject_trials = {0: {0: []}}

    # Organize trials by dataset and subject
    for trial_id, trial_metadata in metadata.items():
        dataset_id = trial_metadata.dataset_id
        subject_id = trial_metadata.subject_id
        assert dataset_id is not None, "Dataset ID cannot be None"
        assert subject_id is not None, "Subject ID cannot be None"
        dataset_subject_trials.setdefault(dataset_id, {}).setdefault(
            subject_id, []
        ).append(trial_id)

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
                all_folds[i].extend(trial_entry)

    test_set = set(all_folds[fold_index])
    val_set = set(all_folds[(fold_index + 1) % n_folds])
    train_set = set(
        trial_id
        for subjects in dataset_subject_trials.values()
        for trials in subjects.values()
        for trial_id in trials
    )
    train_set -= test_set | val_set

    return {"train": list(train_set), "val": list(val_set), "test": list(test_set)}
