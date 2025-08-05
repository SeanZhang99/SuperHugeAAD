import random
from typing import Any, TypeAlias

from .data import (
    CrossValidationEntry,
    DatasetSubjectTrialEntry,
    Metadata,
    MetadataElement,
)

DatasetID: TypeAlias = int
SubjectID: TypeAlias = int


def collect_dataset_subject_trials(
    metadata: Metadata,
):
    dataset_subject_trials: dict[
        DatasetID,
        dict[SubjectID, list[tuple[DatasetSubjectTrialEntry, MetadataElement]]],
    ] = {}
    for entry, trial_metadata in metadata.items():
        dataset_id = trial_metadata.dataset_id
        subject_id = trial_metadata.subject_id
        assert dataset_id is not None, "Dataset ID cannot be None"
        assert subject_id is not None, "Subject ID cannot be None"
        dataset_subject_trials.setdefault(dataset_id, {}).setdefault(
            subject_id, []
        ).append((entry, trial_metadata))
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


def loto(
    metadata: Metadata,
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
            trials: list[tuple[DatasetSubjectTrialEntry, MetadataElement]]
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


def loso(
    metadata: Metadata,
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


def lodo(
    metadata: Metadata,
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


def cgrid_attention_switch_loto(
    metadata: Metadata,
    test_fold_idx: int,
    val_fold_idx: int,
    n_folds: int,
    seed: int = 42,
    **kwargs: Any,
) -> CrossValidationEntry:
    """
    This function implements a leave-one-out cross-validation strategy for the CGrid Attention Switch dataset.
    Divide training/validation/test set based on the 'true_trial_id' field in the metadata.

    For other implementations, see: `loto`
    """

    assert (
        0 <= test_fold_idx < n_folds
    ), f"test_fold_idx must be in the range [0, {n_folds})"
    assert (
        0 <= val_fold_idx < n_folds
    ), f"val_fold_idx must be in the range [0, {n_folds})"
    random.seed(seed)

    # check all metadata entries have true_trial_id
    for entry, trial_metadata in metadata.items():
        if not hasattr(trial_metadata, "true_trial_id") or trial_metadata.true_trial_id is None:  # type: ignore
            raise ValueError(
                f"Metadata entry {entry} does not have a true_trial_id. "
                "This function requires all entries to have a true_trial_id."
            )

    dataset_subject_trials = collect_dataset_subject_trials(metadata)

    all_folds = {i: [] for i in range(n_folds)}

    for dataset_id, subjects in dataset_subject_trials.items():
        for subject_id, trials in subjects.items():
            trials: list[tuple[DatasetSubjectTrialEntry, MetadataElement]]
            # get unique true_trial_ids
            true_trial_ids = list(
                set(
                    trial[1].true_trial_id for trial in trials if trial[1].true_trial_id  # type: ignore
                )
            )
            # shuffle the true_trial_ids, and divide sets based on true_trial_ids.
            random.shuffle(true_trial_ids)
            true_trial_ids_per_fold = len(true_trial_ids) // n_folds
            for i in range(n_folds):
                start_idx = i * true_trial_ids_per_fold
                end_idx = (
                    (i + 1) * true_trial_ids_per_fold
                    if i != n_folds - 1
                    else len(true_trial_ids)
                )
                for trial in trials:
                    if trial[1].true_trial_id in true_trial_ids[start_idx:end_idx]:  # type: ignore
                        all_folds[i].append(trial[0])

    train_set, val_set, test_set = divide_sets(
        all_folds, n_folds, test_fold_idx, val_fold_idx
    )

    return {"train": train_set, "val": val_set, "test": test_set}


def cgrid_leave_one_speaker_out(
    metadata: Metadata,
    test_fold_idx: int,
    val_fold_idx: int,
    n_folds: int = 3,
    seed: int = 42,
    **kwargs: Any,
) -> CrossValidationEntry:
    """
    This function implements a leave-one-speaker-out cross-validation strategy for the CGrid dataset.
    Divide training/validation/test set based on the 'speaker_id' field in the metadata.

    For other implementations, see: `loto`
    """

    assert n_folds == 3, "n_folds must be 3 for leave-one-speaker-out cross-validation"
    assert (
        0 <= test_fold_idx < n_folds
    ), f"test_fold_idx must be in the range [0, {n_folds})"
    assert (
        0 <= val_fold_idx < n_folds
    ), f"val_fold_idx must be in the range [0, {n_folds})"
    random.seed(seed)

    dataset_subject_trials = collect_dataset_subject_trials(metadata)

    train_trials = []
    val_trials = []
    test_trials = []

    # test_fold = 0 -> speaker_id = 4 -> validation, speaker_id = 5,6 -> test
    # test_fold = 1 -> speaker_id = 5 -> validation, speaker_id = 4,6 -> test
    # test_fold = 2 -> speaker_id = 6 -> validation, speaker_id = 4,5 -> test
    # training: speaker_id = 1,2,3
    # total speaker_id : [1,2,3,4,5,6]

    for dataset_id, subjects in dataset_subject_trials.items():
        for subject_id, trials in subjects.items():
            trials: list[tuple[DatasetSubjectTrialEntry, MetadataElement]]

            #  get test_trials based on test_fold_idx
            subject_test_trials = [
                trial[0] for trial in trials if trial[1].speaker_id == test_fold_idx + 4  # type: ignore
            ]
            #  get val_trials based on val_fold_idx
            subject_val_trials = [
                trial[0]
                for trial in trials
                if trial[1].speaker_id == (val_fold_idx + 4) % 3 + 4  # type: ignore
            ]
            #  get train_trials based on the rest of the speakers
            subject_train_trials = [
                trial[0] for trial in trials if trial[1].speaker_id in [1, 2, 3]  # type: ignore
            ]

            train_trials.extend(subject_train_trials)
            val_trials.extend(subject_val_trials)
            test_trials.extend(subject_test_trials)

    return {
        "train": train_trials,
        "val": val_trials,
        "test": test_trials,
    }


def cgrid_within_trial(
    metadata: Metadata,
    test_fold_idx: int,
    val_fold_idx: int,
    n_folds: int,
    seed: int = 42,
    **kwargs: Any,
) -> CrossValidationEntry:
    """
    This function implements a within-trial cross-validation strategy for the CGrid dataset.
    """

    assert (
        0 <= test_fold_idx < n_folds
    ), f"test_fold_idx must be in the range [0, {n_folds})"
    assert (
        0 <= val_fold_idx < n_folds
    ), f"val_fold_idx must be in the range [0, {n_folds})"
    random.seed(seed)

    dataset_subject_trials = collect_dataset_subject_trials(metadata)

    all_trials = []

    for dataset_id, subjects in dataset_subject_trials.items():
        for subject_id, trials in subjects.items():
            trials: list[tuple[DatasetSubjectTrialEntry, MetadataElement]]

            all_trials.extend(trial[0] for trial in trials)

    # val_partition: divide (0.0, 1.0) into n_folds parts, take the val_fold_idx-th part as validation set, and the test_fold_old_idx-th part as test set.
    all_partitions = [(i / n_folds, (i + 1) / n_folds) for i in range(n_folds)]

    val_partition = all_partitions[val_fold_idx]
    test_partition = all_partitions[test_fold_idx]

    return {
        "train": all_trials,
        "val": all_trials,
        "test": all_trials,
        "train_reject_range": (val_partition, test_partition),
        "val_accept_range": (val_partition,),
        "test_accept_range": (test_partition,),
    }
