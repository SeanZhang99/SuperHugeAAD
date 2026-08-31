from typing import Mapping, override
import numpy as np
from superhuge.data.metadata_processing.data import RegressionMetadataElement

from superhuge.data.datasets.eeg_regression_base_dataset import (
    EegRegressionBaseDataset,
)


class EEGMismatchDataset(EegRegressionBaseDataset):
    """Dataset for the EEG-AAD matching-mismatch (N-way speaker classification) task.

    Each sample is one EEG window plus one or more candidate speaker streams
    (env envelopes). The attended (matching) speaker is placed at index 0 of the
    speaker axis; the label is the index of the attended speaker after optional
    shuffling, i.e. the position of the attended speaker among all candidates.

    Adapted from the single-speaker reference implementation
    (25-12-SpeechMatching/EEG_Speech_Matching) for multi-speaker / multi-dataset
    training:
      - `num_additional_speakers` samples extra speakers (legacy behavior);
      - `target_num_class` (new) derives the number of additional speakers per
        sample from the native speaker count of the loaded envelope, so that a
        fixed class count can be kept across datasets with different native
        speaker counts (e.g. DTU/KUL: 2, sparKULee: 1 -> 6 classes everywhere).
    """

    metadata_cls = RegressionMetadataElement

    def __init__(
        self,
        /,
        *,
        num_additional_speakers: int = 0,
        additional_speaker_from_same_trial_prob: float = 0.0,
        shuffle_order: bool = False,
        # if True, shuffle the order of speakers. Otherwise, keep the attended speaker at its label.
        target_num_class: int | None = None,
        **kwargs,
    ):
        """
        Initializes the MismatchDataset with the specified parameters.
        Args:
            num_additional_speakers (int): Number of additional speakers (with respect to the attended and interfering speakers, already in the dataset) to include.
            additional_speaker_from_same_trial_prob (float): Probability of selecting an additional speaker from the same trial. If not selected from the same trial, a random trial is chosen. Setting to 1.0 ensures all additional speakers are from the same trial. Setting to 0.0 ensures all additional speakers are from random trials.
            shuffle_order (bool): If True, shuffle the order of speakers. Otherwise, keep the attended speaker as its label indicated.
            target_num_class (int | None): If given, the number of additional speakers is derived per sample as
                `target_num_class - native_speaker_count` so that every sample has exactly `target_num_class`
                candidate speakers (must be >= native count). Takes precedence over `num_additional_speakers`
                when set. This enables cross-dataset training with a fixed number of classes.
        """
        super().__init__(**kwargs)
        self.stage = kwargs.get("stage", "train")
        if self.stage in ["val", "test"]:
            num_additional_speakers = 0
            additional_speaker_from_same_trial_prob = 0.0
            shuffle_order = False
            if target_num_class:
                target_num_class -= num_additional_speakers
        self.num_additional_speakers = num_additional_speakers
        self.additional_speaker_from_same_trial = (
            additional_speaker_from_same_trial_prob
        )
        self.shuffle_order = shuffle_order
        self.target_num_class = target_num_class

    def get_random_index(self, idx):
        return np.random.randint(0, len(self))

    def get_same_trial_random_index(self, idx):
        _file_idx = self._map_idx_to_file_and_segment(idx)[0]
        _file_start_idx = self._file_to_valid_start_indices_offsets[_file_idx]
        _file_len = (
            self._file_to_valid_start_indices_offsets[_file_idx + 1] - _file_start_idx
        )
        random_offset = np.random.randint(0, _file_len)
        return _file_start_idx + random_offset

    def get_extra_idx(self, idx):
        if self.additional_speaker_from_same_trial > np.random.rand():
            return self.get_same_trial_random_index(idx)
        else:
            return self.get_random_index(idx)

    @override
    def load_data(self, idx: int) -> Mapping[  # type: ignore
        str,
        np.ndarray | RegressionMetadataElement,
    ]:
        # load data normally, then process the metadata to include classification label
        data = super().load_data(idx)
        meta: RegressionMetadataElement = data["meta"]  # type: ignore
        eeg: np.ndarray = data["eeg"]  # type: ignore
        audio: dict[str, np.ndarray] = {k: v for k, v in data.items() if k in self.speech_feature_types}  # type: ignore
        # attended_label = int(getattr(meta, "label"))
        # audio size: (samples, features, speakers), the attended speaker is represented by metadata.label

        for speech_type in self.speech_feature_types:
            if self.target_num_class is not None:
                native_speakers = audio[speech_type].shape[-1]
                num_additional = self.target_num_class - native_speakers
                assert num_additional >= 0, (
                    f"MISMATCH_DATASET:LOAD_DATA:VALUE_ERROR: native speaker count "
                    f"({native_speakers}) exceeds target_num_class "
                    f"({self.target_num_class}) for speech type '{speech_type}'. "
                    f"Increase target_num_class or exclude this dataset."
                )
            else:
                num_additional = self.num_additional_speakers

            for _ in range(num_additional):
                # load extra random data and add to audio
                rnd_idx = idx
                while rnd_idx == idx:
                    rnd_idx = self.get_extra_idx(idx)
                extra_data: np.ndarray = super().load_data(rnd_idx)[speech_type][  # type: ignore
                    ..., 0
                ][
                    ..., np.newaxis
                ]
                audio[speech_type] = np.concatenate(
                    (audio[speech_type], extra_data), axis=2
                )
            if self.shuffle_order:
                perm_idx = np.random.permutation(audio[speech_type].shape[-1])
                audio[speech_type] = audio[speech_type][..., perm_idx]

                label = np.where(perm_idx == 0)[0][0]
                meta.label = type(getattr(meta, "label"))(label) if hasattr(meta, "label") else int(label)  # type: ignore
            elif self.stage == "train":
                assert (
                    getattr(meta, "label") is not None
                ), "Metadata must have 'label' attribute"
                label = int(getattr(meta, "label"))
                audio[speech_type][..., 0], audio[speech_type][..., label] = (
                    audio[speech_type][..., label],
                    audio[speech_type][..., 0],
                )
            else:
                label = 0

        return {"meta": meta, "eeg": eeg, "label": int(label), **audio}  # type: ignore

    @override
    def __getitem__(self, index: int) -> Mapping[str, np.ndarray | dict]:
        item = self.load_data(index)
        meta: dict = item["meta"].model_dump()  # type: ignore

        for field in ["env", "mel", "wav"]:
            if field not in self.speech_feature_types:
                if field in meta:
                    del meta[field]
                if f"{field}_fs" in meta:
                    del meta[f"{field}_fs"]
        return {"meta": meta, **{k: v for k, v in item.items() if k != "meta"}}  # type: ignore
