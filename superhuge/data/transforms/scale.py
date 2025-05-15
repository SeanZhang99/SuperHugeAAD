import os
import pickle
from warnings import warn

import numpy as np

from .abc import Transform


class Scale(Transform):
    """Scale the input data by a given factor."""

    def __init__(
        self,
        /,
        *,
        root_path: str,
        preproc_stage: str = "preprocessed",
        scaling_factor_path: str = "scaling_factor.pkl",
        scaling_factor_key: str = "eeg",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        with open(
            os.path.join(root_path, preproc_stage, "meta", scaling_factor_path), "rb"
        ) as f:
            self._scale: dict[str, dict[str, np.array]] = pickle.load(f)

        self._scaling_factor_key = scaling_factor_key

    def __call__(self, x: np.ndarray, meta, /, *args, **kwargs) -> np.ndarray:
        super().__call__(x)
        entry = f"dataset-{meta['dataset_id']:03d}-subject-{meta['subject_id']:03d}"
        if entry not in self._scale:
            warn(f"Scaling factor for {entry} not found. Skip applying scaling.")
            return x, *args
        else:
            return (
                x
                / self._scale[entry][
                    meta["speech_feature_type"] if self.whom == "audio" else "eeg"
                ],
                *args,
            )
