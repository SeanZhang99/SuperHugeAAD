# This script visualizes PCA components based on given EEG samples, and plot the PCA components in 2D plot, with heatmap colorbar indicating time progression.
# It helps to understand how the PCA components evolve over time.


from itertools import product
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_eeg_data(root_path: Path | str, dataset_id: int, subject_id: int):
    root_path = Path(root_path)
    files = list(
        root_path.rglob(f"dataset-{dataset_id:03d}-subject-{subject_id:03d}*.npy")
    )

    if not files:
        raise ValueError(f"No .npy files found in {root_path}")

    data = []
    for file in files:
        eeg = np.load(file)
        data.append(eeg)

    return data


def plot_pca_vs_time(eeg_samples: list[np.ndarray], n_components: int = 2):
    # compute autocorrelatino for each sample in the list
    pca = PCA(n_components=n_components)
    pca_results = [pca.fit_transform(sample) for sample in eeg_samples]
    # plot the PCA results in 2D plot, with heatmap colorbar indicating time progression
    plt.figure(figsize=(10, 8))
    corr_x = []
    corr_y = []
    for i, pca_result in enumerate(pca_results):
        corr = np.cov(pca_result, rowvar=False)
        corr_x.append(corr[0, 0])
        corr_y.append(corr[0, 1])
    plt.scatter(
        np.stack(corr_x),
        np.stack(corr_y),
        c=np.linspace(0, 1, len(corr_x)),
        cmap="viridis",
        alpha=0.7,
    )
    plt.colorbar()
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA Components vs Time")
    plt.show()


if __name__ == "__main__":
    root_path = Path(
        rf"E:\derivatives\SuperHuge\preprocessed\eeg"
    )  # replace with your EEG data directory
    for dataset_id, subject_id in product(
        [
            4,
        ],
        range(1, 17),
    ):
        try:
            data = load_eeg_data(root_path, dataset_id, subject_id)
            plot_pca_vs_time(data)
        except ValueError as e:
            print(e)
