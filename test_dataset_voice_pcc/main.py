from pathlib import Path

import numpy as np


def test_single_file(path: Path | str):
    path = Path(path)
    assert path.exists()
    assert path.is_file()
    assert path.suffix == ".npy" or path.suffix == ".npz"

    x = np.load(path)
    assert isinstance(x, np.ndarray)
    # x: time x features x speakers

    speaker_labels = ["a", *[f"u_{i}" for i in range(1, x.shape[-1])]]
    assert len(speaker_labels) == x.shape[-1]
    assert x.ndim == 3

    pcc = np.zeros((len(speaker_labels) - 1,))
    if len(speaker_labels) > 1:
        # compute pcc between the first speaker and the rest

        mean_x = x.mean(axis=0, keepdims=True)
        std_x = x.std(axis=0, keepdims=True)

        for i in range(1, x.shape[-1]):

            pcc_value: np.ndarray = (
                (x[..., 0] - mean_x[..., 0]) * (x[..., i] - mean_x[..., i])
            ).mean() / (std_x[..., 0] * std_x[..., i] + 1e-8)

            pcc_value = pcc_value.flatten()

            pcc[i - 1] = pcc_value[0]

    return pcc, speaker_labels


if __name__ == "__main__":
    root_path = Path(r"E:\derivatives\SuperHuge\preprocessed\stimuli\env")
    for dataset in range(1, 10):
        files = list(root_path.glob(f"dataset-{dataset:03d}*.npy")) + list(
            root_path.glob(f"dataset-{dataset:03d}*.npz")
        )

        all_pcc = []
        for file in files:
            pcc, speaker_labels = test_single_file(file)
            all_pcc.append(pcc)

        if all_pcc:
            assert "speaker_labels" in locals()

            all_pcc = np.stack(all_pcc, axis=0)
            mean_pcc = all_pcc.mean(axis=0)
            print(
                f"dataset-{dataset:03d}: ",
                {
                    f"{label}_pcc": f"{value:.4f}"
                    for label, value in zip(locals()["speaker_labels"][1:], mean_pcc)
                },
            )
