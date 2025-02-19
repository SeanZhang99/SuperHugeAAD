import inspect
import random
from collections.abc import Callable
from typing import Any, Protocol, TypeAlias, cast

from pydantic import BaseModel
from pydantic_core import core_schema


class MetaDataElement(BaseModel, extra="allow"):
    dataset_id: int | None = 0
    subject_id: int | None = 0
    trial_id: int | None = 0
    num_channel: int | None = 0
    signal_length: int | None = 0
    fs: int | None = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.entry = f"dataset-{self.dataset_id:03d}-subject-{self.subject_id:03d}-trial-{self.trial_id:03d}"


class RegressionMetaDataElement(MetaDataElement):
    env: str | None
    mel: str | None


class ClassifyMetaDataElement(MetaDataElement):
    label: str | int | None


DatasetSubjectTrialEntry: TypeAlias = str
MetaDataField: TypeAlias = str
FoldIndicator: TypeAlias = str
MetaDataValue: TypeAlias = Any
MetaData: TypeAlias = dict[DatasetSubjectTrialEntry, MetaDataElement]
CrossValidationEntry: TypeAlias = dict[FoldIndicator, list[DatasetSubjectTrialEntry]]


class GroupingFunction(Protocol):
    def __call__(
        self,
        **kwargs,
    ) -> CrossValidationEntry: ...

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.no_info_after_validator_function(
            cls._validate, core_schema.callable_schema()
        )

    @staticmethod
    def _validate(value: Callable):
        # 获取函数签名
        func_signature = inspect.signature(value)

        # 定义期望的参数和类型
        expected_params = {
            "metadata": MetaData,
            "fold_index": int,
            "n_folds": int,
            "seed": int,
        }

        # 检查参数名称和类型
        for expected_param_name, expected_param_type in expected_params.items():
            if expected_param_name not in func_signature.parameters:
                raise ValueError(
                    f"METADATA_PROCESSING:GROUPING_FUNCTION:VALIDATE:FUNCTION_SIGNATURE_VALIDATION:SIGNATURE_ERROR: Missing required parameter: {
                        expected_param_name}"
                )

            func_param_type = func_signature.parameters[expected_param_name].annotation
            if func_param_type is inspect._empty:
                raise TypeError(
                    f"METADATA_PROCESSING:GROUPING_FUNCTION:VALIDATE:FUNCTION_SIGNATURE_VALIDATION:ANNOTATION_ERROR: Parameter {
                        expected_param_name} must have a type annotation."
                )

            if (
                expected_param_type != func_param_type
                and func_param_type not in expected_param_type
            ):
                raise TypeError(
                    f"METADATA_PROCESSING:GROUPING_FUNCTION:VALIDATE:FUNCTION_SIGNATURE_VALIDATION:ANNOTATION_ERROR: Expected parameter {expected_param_name} to be {
                        expected_param_type}, but got {func_param_type}"
                )
            # 验证类型是否匹配

        test_metadata = generate_test_metadata()
        test_fold_index = 0
        test_n_folds = 5
        test_random_seed = None
        result = value(
            metadata=test_metadata,
            fold_index=test_fold_index,
            n_folds=test_n_folds,
            random_seed=test_random_seed,
        )
        validate_is_dict = isinstance(result, dict)
        if validate_is_dict:
            validate_key_is_str = all([isinstance(key, str) for key in result.keys()])
        else:
            validate_key_is_str = False
        if validate_key_is_str:
            validate_value_is_list = all(
                [isinstance(value, list) for value in result.values()]
            )
        else:
            validate_value_is_list = False
        if validate_value_is_list:
            validate_value_element_is_str = all(
                [isinstance(e, str) for l in result.values() for e in l]
            )
        else:
            validate_value_element_is_str = False
        if validate_is_dict and validate_key_is_str and validate_value_element_is_str:
            pass
        else:
            raise TypeError(
                f"METADATA_PROCESSING:GROUPING_FUNCTION:VALIDATE:RETURN_VALIDATION:TYPE_ERROR: Return value must be a dictionary, but got {type(result)}"
            )

        return value


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

    return {
        "train": list(train_trials),
        "val": list(val_trials),
        "test": list(test_set),
    }


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


def generate_test_metadata(
    num_datasets=10,
    num_subjects_per_dataset=10,
    min_trials_per_subject=1,
    max_trials_per_subject=20,
) -> MetaData:
    metadata = {}
    for i in range(num_datasets):
        for j in range(num_subjects_per_dataset):
            num_trials = random.randint(min_trials_per_subject, max_trials_per_subject)
            for k in range(num_trials):
                metadata[f"dataset-{i+1:03d}-subject-{j+1:03d}-trial-{k+1:03d}"] = (
                    MetaDataElement(
                        dataset_id=i + 1,
                        subject_id=j + 1,
                        trial_id=k + 1,
                        num_channel=32,
                        signal_length=10000,
                        fs=128,
                    )
                )

    return metadata


if __name__ == "__main__":

    # 生成元数据的函数
    metadata = generate_test_metadata()

    # Testing LOTO, LOSO, and LODO with multiple folds and random seeds
    for fold_index in range(3):
        print(f"Testing LOTO for fold {fold_index}:")
        result_loto = loto(metadata=metadata, fold_index=fold_index, n_folds=3, seed=42)
        print(f"Train set: {result_loto['train']}")
        print(f"Validation set: {result_loto['val']}")
        print(f"Test set: {result_loto['test']}\n")

    for fold_index in range(5):
        print(f"Testing LOSO for fold {fold_index}:")
        result_loso = loso(metadata=metadata, fold_index=fold_index, n_folds=5, seed=42)
        print(f"Train set: {result_loso['train']}")
        print(f"Validation set: {result_loso['val']}")
        print(f"Test set: {result_loso['test']}\n")

    for fold_index in range(2):
        print(f"Testing LODO for fold {fold_index}:")
        result_lodo = lodo(metadata=metadata, fold_index=fold_index, n_folds=2, seed=42)
        print(f"Train set: {result_lodo['train']}")
        print(f"Validation set: {result_lodo['val']}")
        print(f"Test set: {result_lodo['test']}\n")
