import importlib
from typing import Sequence
from warnings import warn
from .data_interface import DInterface
from ..datasets import EegClassifyBaseDataset, EegRegressionBaseDataset
from ..metadata_filters import MetadataValueSelector


def create_data_interface(
    root_path: str,
    window_length: int,
    fs,
    classify: bool = False,
    regression: bool = False,
    metadata_fields: list[str] | None = None,
    select_subject: int | Sequence[int] | None = None,
    select_trial: int | Sequence[int] | None = None,
    meta_filter_func_args: Sequence | None = None,
    leave_one_out: str = "loto",
    test_fold_idx: int = 0,
    val_fold_idx: int = 1,
    n_folds: int = 5,
    overlap: int = 1,
    preproc_stage: str | None = None,
    bandpass_wn: Sequence[float] | None = None,
    refs: int | None = None,
):
    """
    Create a data interface for MATLAB.
    This function is intended to be called from MATLAB to create an instance of the DInterface class.

    Returns:
        DInterface: An instance of the DInterface class.
    """
    assert classify or regression, "Either classify or regression must be True."
    dataset_class = EegClassifyBaseDataset if classify else EegRegressionBaseDataset

    metadata_filter = []
    if select_subject:
        assert isinstance(
            select_subject, (int, Sequence)
        ), "select_subject must be an int or a sequence of ints."
        metadata_filter.append(MetadataValueSelector("subject_id", select_subject))
    if select_trial:
        assert isinstance(
            select_trial, (int, Sequence)
        ), "select_trial must be an int or a sequence of ints."
        metadata_filter.append(MetadataValueSelector("trial_id", select_trial))

    metadata_filter = metadata_filter if metadata_filter else None

    try:
        group_pkg = importlib.import_module(
            ".data.metadata_processing.group", "superhuge"
        )
        meta_group_func = getattr(group_pkg, leave_one_out)
    except ImportError as e:
        raise ImportError(
            f"Could not import metadata_processing.group from package `superhuge`"
        ) from e
    except AttributeError as e:
        warn(
            f"Function {leave_one_out} not found in metadata_processing.group. Using default grouping function.",
            UserWarning,
        )
        from ..metadata_processing.group import loto

        meta_group_func = loto

    transforms = []
    if bandpass_wn:
        from ..transforms import Filter

        transforms.append(Filter(Wn=bandpass_wn, fs=fs, btype="bandpass", order=5))
    if refs:
        from ..transforms import Resample

        transforms.append(Resample(old_fs=fs, new_fs=refs))

    transforms = transforms if transforms else None

    return DInterface(
        dataset_class=dataset_class,
        dataloader_args={"batch_size": 1, "shuffle": False, "num_workers": 0},
        root_path=root_path,
        window_length=window_length,
        fs=fs,
        meta_filter_func=metadata_filter,
        meta_filter_func_args=meta_filter_func_args,
        meta_group_func=meta_group_func,
        metadata_fields=metadata_fields,
        test_fold_idx=test_fold_idx,
        val_fold_idx=val_fold_idx,
        n_folds=n_folds,
        transform=transforms,
        overlap=overlap,
        preproc_stage=preproc_stage,
    )
