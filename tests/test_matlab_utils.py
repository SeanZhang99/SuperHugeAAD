import pytest
from superhuge.data.interface.matlab_utils import create_data_interface
from superhuge.data.interface.data_interface import DInterface


def test_create_data_interface_classify():
    interface = create_data_interface(
        root_path="E:\\derivatives\\SuperHuge",
        window_length=10,
        fs=128,
        classify=True,
        regression=False,
        select_subject=1,
        select_trial=None,
        leave_one_out="loto",
        test_fold_idx=0,
        val_fold_idx=1,
        n_folds=5,
        overlap=1,
        preproc_stage="preprocessed",
        bandpass_wn=[0.5, 30],
        refs=None,
        metadata_fields=["label"],
        meta_filter_func_args=[
            "binary_leftright",
        ],
    )
    assert isinstance(interface, DInterface)
    assert interface.dataset_cfg.dataset_class.__name__ == "EegClassifyBaseDataset"


def test_create_data_interface_regression():
    interface = create_data_interface(
        root_path="E:\\derivatives\\SuperHuge",
        window_length=10,
        fs=128,
        classify=False,
        regression=True,
        select_subject=None,
        select_trial=2,
        leave_one_out="loto",
        test_fold_idx=0,
        val_fold_idx=1,
        n_folds=5,
        overlap=1,
        preproc_stage="preprocessed",
        bandpass_wn=[1, 10],
        refs=40,
        metadata_fields=["env"],
        meta_filter_func_args=[
            "env",
        ],
    )
    assert isinstance(interface, DInterface)
    assert interface.dataset_cfg.dataset_class.__name__ == "EegRegressionBaseDataset"


def test_create_data_interface_invalid_args():
    with pytest.raises(AssertionError):
        create_data_interface(
            root_path="E:\\derivatives\\SuperHuge",
            window_length=100,
            fs=128,
            classify=False,
            regression=False,  # Both classify and regression are False
        )
