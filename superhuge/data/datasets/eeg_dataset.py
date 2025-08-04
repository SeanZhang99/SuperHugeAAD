from math import ceil, floor
import os
from typing import Sequence

from matplotlib.transforms import Transform
import numpy as np
import tqdm
from torch.utils.data import Dataset

from ..transforms.stats_abc import StatisticalTransform

from ...utils.validate import validate_kwargs
from ..metadata_processing.data import Metadata, MetadataElement, MetadataField
from ..transforms.composer import TransformComposer


class EegDataset(Dataset):
    metadata_cls = MetadataElement
    """
    EEG Dataset for loading and processing EEG signal data.

    Attributes:
        eeg_path (str): Path to the dataset folder.
        files (Sequence[str]): List of file names.
        metadata (Metadata): Metadata containing information for each trial.
        segment_length (int): Length of each signal segment.
        overlap (int): Overlap ratio, determines the stride for segment slicing.
        transform (TransformComposer | None): Transformations applied to samples.
        metadata_fields (list[MetadataField]): Metadata fields to record.
        accept_range (tuple[float, float] | None): Range of valid signal segments (start, end). If None, all segments are accepted.
        reject_range (tuple[float, float] | None): Range of rejected signal segments (start, end). If None, no segments are rejected.
    """

    def __init__(self, **kwargs):
        """
        Initialize the EEG Dataset.

        Args:
            eeg_path (str): Path to the dataset folder.
            files (list): List of file names.
            metadata (dict): Metadata containing information for each trial.
            window_length (int): Length of each signal segment.
            overlap (int): Overlap ratio, determines the stride for segment slicing.
            transform (TransformComposer | None): Transformations applied to samples.
            metadata_fields (list): Metadata fields to record.
            accept_range (tuple[float, float] | None): Range of valid signal segments (start, end).
            reject_range (tuple[float, float] | None): Range of rejected signal segments (start, end).
        """
        required_keys = ["eeg_path", "files", "metadata", "metadata_fields"]
        self._validate_kwargs(kwargs.keys(), required_keys)

        self.eeg_path: str = kwargs["eeg_path"]
        self._input_files_list: Sequence[str] = kwargs[
            "files"
        ]  # Renamed from _tmp_files_list
        self.metadata: Metadata = kwargs["metadata"]
        self.segment_length: int | float = kwargs.get("fs", 128.0) * kwargs.get(
            "window_length", 10.0
        )  # Default segment length is 1280
        self.overlap: int = kwargs.get("overlap", 1)  # Default no overlap
        self.transform: TransformComposer | None = kwargs.get("transform", None)
        self.metadata_fields: list[MetadataField] = kwargs["metadata_fields"]
        self.accept_range: tuple[float, float] | Sequence[tuple[float, float]] = (
            kwargs.get("accept_range", None)
        )
        self.reject_range: (
            tuple[float, float] | Sequence[tuple[float, float]] | None
        ) = kwargs.get("reject_range", None)

        if self.accept_range is None:
            self.accept_range = (0.0, 1.0)

        self._validate_ranges()

        # Ensure files are a subset of metadata's keys
        self._validate_files()

        # Cache valid segments using numpy arrays
        self._cache_and_prepare_segments()

        # Delete temporary files list to avoid memory leakage
        del self._input_files_list

        self._stage = kwargs.get("stage", None)
        self._update_and_fit_stats_transform()

    def _update_and_fit_stats_transform(self):
        """
        Update the transform to include metadata fields and fit it if necessary.
        """
        if (
            isinstance(self.transform, TransformComposer)
            and any(
                [isinstance(t, StatisticalTransform) for t in self.transform.transforms]
            )
            and self._stage == "train"
        ):
            updatable_transforms = [
                t
                for t in self.transform.transforms
                if isinstance(t, StatisticalTransform)
            ]
            for file_name in tqdm.tqdm(self.files):
                eeg = np.load(os.path.join(self.eeg_path, file_name + ".npy"))
                for t in updatable_transforms:
                    t.update(eeg)
            for t in updatable_transforms:
                t.fit()

    def sync_transform_stats(
        self, source_transform: StatisticalTransform | TransformComposer
    ):
        """
        Synchronize the statistics of the transform with another StatisticalTransform or TransformComposer.

        Args:
            source_transform (StatisticalTransform | TransformComposer): The source transform to synchronize with.
        """
        if isinstance(source_transform, StatisticalTransform):
            for t in self.transform.transforms:
                if (
                    isinstance(t, type(source_transform))
                    and t.whom == source_transform.whom
                    and t.when == source_transform.when
                ):
                    t.stat = source_transform.stat
        elif isinstance(source_transform, TransformComposer):
            for source_t in source_transform.transforms:
                if isinstance(source_t, StatisticalTransform):
                    for t in self.transform.transforms:
                        if (
                            isinstance(t, type(source_t))
                            and t.whom == source_t.whom
                            and t.when == source_t.when
                        ):
                            t.stat = source_t.stat

    @property
    def files(self):
        """
        Access the file names.

        Returns:
            list[str]: A list of file names (as strings).
        """
        return self._file_names.tolist()

    def _validate_ranges(self):
        """
        Validate the accept_range and reject_range attributes.
        """

        def validate_range(range_seq):
            for r in range_seq:
                assert (
                    0.0 <= r[0] < r[1] <= 1.0
                ), "Each range must be a tuple of two floats in the range [0.0, 1.0] with start < end."

        if isinstance(self.accept_range, tuple) and isinstance(
            self.accept_range[0], float
        ):
            self.accept_range = (self.accept_range,)
        validate_range(self.accept_range)

        if self.reject_range:
            if isinstance(self.reject_range, tuple) and isinstance(
                self.reject_range[0], float
            ):
                self.reject_range = (self.reject_range,)
            validate_range(self.reject_range)

    def _validate_files(self):
        """
        Ensure that the files attribute is a subset of the metadata keys.
        """
        assert set(self._input_files_list).issubset(  # Updated attribute name
            self.metadata.keys()
        ), f"Some files specified in `files` do not have corresponding metadata: {set(self._input_files_list) - set(self.metadata.keys())}"

    @property
    def valid_segments_cache(self):
        """
        Access the cached valid segments.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: A numpy array of file names (as strings).
                - np.ndarray: A numpy array of offsets mapping file indices to valid segments.
                - np.ndarray: A single array of valid segment start indices.
        """
        return (
            self._file_names,
            self._file_to_valid_start_indices_offsets,
            self._valid_start_indices,
        )

    def _cache_and_prepare_segments(self):
        """
        Cache valid segments for all files using numpy arrays.
        """
        file_names = []
        file_to_valid_start_indices_offsets = np.zeros(
            len(self._input_files_list) + 1, dtype=np.int32
        )  # Preallocate offsets (+1 for boundary)
        valid_start_indices = []
        per_file_sample_count = []

        current_index = 0

        for file_idx, file in enumerate(self._input_files_list):
            trial_length = self.metadata[file].signal_length
            assert trial_length, f"Metadata for file {file} is missing signal_length."
            start_indices = self._find_valid_start_idx(trial_length)

            # Store file name and update offsets
            file_names.append(file)
            file_to_valid_start_indices_offsets[file_idx] = current_index
            valid_start_indices.extend(start_indices)
            current_index += len(start_indices)
            per_file_sample_count.append(len(start_indices))

        # Add the final boundary for offsets
        file_to_valid_start_indices_offsets[len(self._input_files_list)] = current_index

        self._file_names = np.array(file_names, dtype=np.dtypes.StrDType)
        self._file_to_valid_start_indices_offsets = file_to_valid_start_indices_offsets
        self._valid_start_indices = np.array(valid_start_indices, dtype=np.int32)
        self._per_file_sample_count = np.array(per_file_sample_count, dtype=np.int32)

    def _find_valid_start_idx(self, trial_length: int):
        """
        Compute valid segments for a given trial length.

        Args:
            trial_length (int): The length of the trial.

        Returns:
            list[int]: A list of valid segment start indices.
        """
        stride = ceil(self.segment_length // self.overlap)
        start_indices = []

        for accept_range in self.accept_range:
            accept_start = int(accept_range[0] * trial_length)
            accept_end = int(accept_range[1] * trial_length)

            for start_idx in range(accept_start, accept_end, stride):
                end_idx = start_idx + self.segment_length
                if end_idx > accept_end:
                    break
                if self.reject_range:
                    reject_flag = False
                    for reject_range in self.reject_range:
                        reject_start = int(reject_range[0] * trial_length)
                        reject_end = int(reject_range[1] * trial_length)
                        if (
                            reject_start <= start_idx < reject_end
                            or reject_start < end_idx <= reject_end
                        ):
                            reject_flag = True
                            break
                    if reject_flag:
                        continue
                start_indices.append(start_idx)
        return start_indices

    def __len__(self):
        return self._file_to_valid_start_indices_offsets[-1]

    @property
    def len(self):
        return len(self)

    def load_data(self, idx):
        """
        Retrieve a sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing metadata and the EEG signal segment.

        Raises:
            IndexError: If the index is out of range.
        """
        file_idx, start_idx = self._map_idx_to_file_and_segment(idx)
        file_name = self.files[file_idx]  # Redirect to self._file_names via property
        file_path = os.path.join(self.eeg_path, file_name + ".npy")

        meta = self.metadata[file_name].model_dump()

        eeg: np.ndarray | np.memmap = np.load(
            file_path, mmap_mode="r", allow_pickle=False
        )

        self._validate_eeg_shape(eeg, file_name)

        if self.transform:
            eeg = self.transform(eeg, meta=meta, when="before_slicing", whom="eeg")[0]

        eeg_seg = eeg[start_idx : start_idx + self.segment_length].copy()

        if self.transform:
            eeg_seg = self.transform(
                eeg_seg, meta=meta, when="before_returning", whom="eeg"
            )[0]

        return {"meta": meta, "eeg": eeg_seg.astype(np.float32)}

    def __getitem__(self, idx):
        return self.load_data(idx)

    def _validate_eeg_shape(self, eeg: np.ndarray, file_name: str):
        """
        Validate the shape of the loaded EEG data.

        Args:
            eeg (np.ndarray): The loaded EEG data.
            file_name (str): The name of the file.

        Raises:
            AssertionError: If the shape of the EEG data is invalid.
        """
        assert (
            eeg.shape[0] == self.metadata[file_name].signal_length
        ), f"Loaded data shape {eeg.shape[0]} does not match expected signal_length {self.metadata[file_name].signal_length} for file {file_name}."
        assert (
            eeg.ndim == 2
        ), f"Loaded data is not 2D, but {eeg.ndim}D for file {file_name}."
        assert eeg.shape[1] == len(
            self.metadata[file_name].channel_infos
        ), f"Number of channels {eeg.shape[1]} does not match expected {len(self.metadata[file_name].channel_infos)} for file {file_name}."

    def _map_idx_to_file_and_segment(self, idx: int):
        """
        Map a global index to a specific file and segment index.

        Args:
            idx (int): Global index.

        Returns:
            tuple[int,int]: File index and start_idx

        Raises:
            IndexError: If the index is out of range.
        """
        file_names, file_to_valid_start_indices_offsets, valid_segments = (
            self.valid_segments_cache
        )

        # Find the file corresponding to the global index
        file_idx: int = (
            np.searchsorted(file_to_valid_start_indices_offsets, idx, side="right") - 1
        )
        if file_idx < 0 or file_idx >= len(file_names):
            raise IndexError(
                f"Index {idx} is out of range for the dataset. Total valid segments: {file_to_valid_start_indices_offsets[-1]}."
            )

        # Compute the segment index within the file
        local_idx = idx - file_to_valid_start_indices_offsets[file_idx]
        if local_idx < 0 or local_idx >= (
            file_to_valid_start_indices_offsets[file_idx + 1]
            - file_to_valid_start_indices_offsets[file_idx]
        ):
            raise IndexError(
                f"Local index {local_idx} is out of range for file {file_names[file_idx]}."
            )

        return (
            file_idx,
            valid_segments[file_to_valid_start_indices_offsets[file_idx] + local_idx],
        )

    def _validate_kwargs(self, kwargs: set[str], required_keys: list[str]):
        """
        Validate the provided keyword arguments.

        Args:
            kwargs (set[str]): The provided keyword arguments.
            required_keys (list[str]): The required keys.

        Raises:
            AssertionError: If any required key is missing.
        """
        validate_kwargs(kwargs, required_keys)


if __name__ == "__main__":
    root_path = r"E:/derivatives/SuperHuge"
    datasets = EegDataset().create_datasets(root_path=root_path)
    (x for x in tqdm.tqdm(datasets))
    pass
