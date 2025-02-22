import inspect
import random
from collections.abc import Callable
from typing import Any, Protocol, TypeAlias

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


def generate_test_metadata(
    num_datasets=10,
    num_subjects_per_dataset=10,
    min_trials_per_subject=1,
    max_trials_per_subject=20,
) -> MetaData:
    metadata = {}
    random.seed(42)
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
