from matplotlib.pylab import f
import pytest
import numpy as np
from superhuge.data.datasets.eeg_dataset import EegDataset
from superhuge.data.metadata_processing.data import Metadata, MetadataElement


@pytest.fixture
def mock_metadata():
    metadata = {
        "dataset-001-subject-001": MetadataElement(
            signal_length=1000,
            dataset_id=1,
            subject_id=1,
            channel_infos=["C1", "C2"],
        ),
        "dataset-001-subject-002": MetadataElement(
            signal_length=800,
            dataset_id=1,
            subject_id=2,
            channel_infos=["C1", "C2"],
        ),
    }
    return metadata


@pytest.fixture
def mock_files():
    return ["dataset-001-subject-001", "dataset-001-subject-002"]


@pytest.fixture
def mock_eeg_path(tmp_path):
    eeg_path = tmp_path / "eeg_data"
    eeg_path.mkdir()
    np.save(
        eeg_path / "dataset-001-subject-001.npy",
        np.random.rand(
            1000,
            2,
        ).astype(np.float32),
    )
    np.save(
        eeg_path / "dataset-001-subject-002.npy",
        np.random.rand(800, 2).astype(np.float32),
    )
    return eeg_path


def test_accept_range(mock_metadata, mock_files, mock_eeg_path):
    dataset = EegDataset(
        eeg_path=str(mock_eeg_path),
        files=mock_files,
        metadata=mock_metadata,
        metadata_fields=[],
        accept_range=(0.2, 0.8),
        window_length=1,
        fs=100,
        overlap=2,
    )
    stride = dataset.segment_length // dataset.overlap
    assert len(dataset) > 0, "Dataset should have valid segments within accept_range."
    for idx in range(len(dataset)):
        sample = dataset[idx]
        file_name = f"dataset-{sample['meta']['dataset_id']:03d}-subject-{sample['meta']['subject_id']:03d}"
        eeg_original = np.load(mock_eeg_path / f"{file_name}.npy", mmap_mode="r")
        eeg_segment = sample["eeg"]

        # Calculate start_idx by manually slicing through the original EEG file
        for start_idx in range(0, eeg_original.shape[0] - dataset.segment_length + 1):
            if np.array_equal(
                eeg_original[start_idx : start_idx + dataset.segment_length],
                eeg_segment,  # type: ignore
            ):
                break
        else:
            raise AssertionError(
                f"Segment not found in the original EEG file. {file_name} {idx}"
            )

        end_idx = start_idx + dataset.segment_length
        signal_length = mock_metadata[file_name].signal_length

        assert (
            0.2 * signal_length <= start_idx < 0.8 * signal_length
        ), f"Start index is not within the accept range. Expected: [0.2 * signal_length, 0.8 * signal_length], Got: {start_idx}"
        assert (
            0.2 * signal_length < end_idx <= 0.8 * signal_length
        ), f"End index is not within the accept range. Expected: (0.2 * signal_length, 0.8 * signal_length], Got: {end_idx}"


def test_reject_range(mock_metadata, mock_files, mock_eeg_path):
    dataset = EegDataset(
        eeg_path=str(mock_eeg_path),
        files=mock_files,
        metadata=mock_metadata,
        metadata_fields=[],
        reject_range=(0.4, 0.6),
        window_length=1,
        fs=100,
        overlap=2,
    )
    stride = dataset.segment_length // dataset.overlap
    assert len(dataset) > 0, "Dataset should have valid segments outside reject_range."
    for idx in range(len(dataset)):
        sample = dataset[idx]
        file_name = f"dataset-{sample['meta']['dataset_id']:03d}-subject-{sample['meta']['subject_id']:03d}"
        eeg_original = np.load(mock_eeg_path / f"{file_name}.npy", mmap_mode="r")
        eeg_segment = sample["eeg"]

        # Calculate start_idx by manually slicing through the original EEG file
        for start_idx in range(0, eeg_original.shape[0] - dataset.segment_length + 1):
            if np.array_equal(
                eeg_original[start_idx : start_idx + dataset.segment_length],
                eeg_segment,
            ):
                break
        else:
            raise AssertionError("Segment not found in the original EEG file.")

        end_idx = start_idx + dataset.segment_length
        signal_length = mock_metadata[file_name].signal_length

        assert not (0.4 * signal_length <= start_idx < 0.6 * signal_length)
        assert not (0.4 * signal_length < end_idx <= 0.6 * signal_length)


def test_combined_ranges(mock_metadata, mock_files, mock_eeg_path):
    dataset = EegDataset(
        eeg_path=str(mock_eeg_path),
        files=mock_files,
        metadata=mock_metadata,
        metadata_fields=[],
        accept_range=(0.2, 0.8),
        reject_range=(0.4, 0.6),
        window_length=1,
        fs=100,
        overlap=2,
    )
    stride = dataset.segment_length // dataset.overlap
    assert (
        len(dataset) > 0
    ), "Dataset should have valid segments within accept_range and outside reject_range."
    for idx in range(len(dataset)):
        sample = dataset[idx]
        file_name = f"dataset-{sample['meta']['dataset_id']:03d}-subject-{sample['meta']['subject_id']:03d}"
        eeg_original = np.load(mock_eeg_path / f"{file_name}.npy", mmap_mode="r")
        eeg_segment = sample["eeg"]

        # Calculate start_idx by manually slicing through the original EEG file
        for start_idx in range(0, eeg_original.shape[0] - dataset.segment_length + 1):
            if np.array_equal(
                eeg_original[start_idx : start_idx + dataset.segment_length],
                eeg_segment,
            ):
                break
        else:
            raise AssertionError(
                f"Segment not found in the original EEG file. {file_name} {idx}"
            )

        end_idx = start_idx + dataset.segment_length
        signal_length = mock_metadata[file_name].signal_length

        assert 0.2 * signal_length <= start_idx < 0.8 * signal_length
        assert 0.2 * signal_length < end_idx <= 0.8 * signal_length
        assert not (0.4 * signal_length <= start_idx < 0.6 * signal_length)
        assert not (0.4 * signal_length < end_idx <= 0.6 * signal_length)
