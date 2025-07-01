from typing import Sequence
from .data_interface import DInterface
from ..datasets import EegClassifyBaseDataset, EegRegressionBaseDataset
from ..metadata_filters import MetadataValueSelector


def create_data_interface(
    root_path: str,
    window_length: int,
    fs,
    classify: bool = False,
    regression: bool = False,
    select_subject: int | Sequence[int] | None = None,
    select_trial: int | Sequence[int] | None = None,
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
        metadata_filter.append()

    return DInterface(
        dataset_class=dataset_class,
        dataloader_args={"batch_size": 1, "shuffle": False, "num_workers": 0},
        root_path=root_path,
        window_length=window_length,
        fs=fs,
    )
