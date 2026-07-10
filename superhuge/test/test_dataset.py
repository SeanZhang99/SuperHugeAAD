import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..data.datasets.eeg_dataset import EegDataset


def test_dataset(dataset: "EegDataset"):
    test_data = (
        np.load(
            r"C:\Users\sean\Downloads\locus-of-auditory-attention-cnn-master\src\test_set.npy"
        )
        .astype(np.float32)
        .transpose(2, 0, 1)
    )
    test_labels = np.load(
        r"C:\Users\sean\Downloads\locus-of-auditory-attention-cnn-master\src\test_labels.npy"
    ).astype(np.long)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, data_otf in enumerate(dataloader):
        data_seg = {}
        data_seg["eeg"] = test_data[i][np.newaxis, ...]
        data_seg["label"] = test_labels[i] - 1
        seg_subject_id = i // 75 // 2 + 1
        seg_trial_id = i // 75 % 2 + 1
        otf_subject_id = data_otf["meta"]["subject_id"]
        otf_trial_id = data_otf["meta"]["trial_id"]
        data_otf["eeg"] = data_otf["eeg"].numpy()
        # subject_id=2, trial_id=1 第一个segment不对。todo：检查哪个数据集和原始数据是符合的
        if not np.allclose(data_otf["eeg"], data_seg["eeg"], atol=1e-5):
            print(f"Mismatch found at sample {i}, plotting comparison...")
            # use matplotlib to show data differences
            plt.subplot(3, 1, 1)
            plt.plot(data_otf["eeg"])
            plt.text(0, 0, str(data_otf["meta"]["entry"]))
            plt.title("OTF EEG Signal")
            plt.subplot(3, 1, 2)
            plt.plot(data_seg["eeg"])
            idx = i // 75
            # subject_id: from 1 to 16. trial_id: from 1 to 2.
            # idx = (subject_id - 1) * 2 + trial_id - 1
            plt.text(
                0,
                0,
                str(f"dataset-004-subject-{idx//2+1:03d}-trial-{idx%2+1:03d}"),
            )
            plt.title("SEG EEG Signal")
            plt.subplot(3, 1, 3)
            plt.plot(data_otf["eeg"] - data_seg["eeg"])
            plt.title("Difference")
            plt.show()

        if data_seg["label"] != data_otf["label"]:
            print(f"OTF label: {data_otf['label']}, SEG label: {data_seg['label']}")
