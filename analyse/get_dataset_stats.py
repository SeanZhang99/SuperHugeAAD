from matplotlib.axes import Axes
from tqdm import tqdm, trange
import superhuge
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest
import torch

from superhuge.model.module.pre_model import Channel1D


def filter_meta_by_dataset_name(dataset_name):
    return lambda meta: meta if meta.dataset_name == dataset_name else None


def remove_outliers(data, percentile=1):
    lower_bound = np.percentile(data, percentile // 2)
    upper_bound = np.percentile(data, 100 - percentile // 2)
    data = np.clip(data, lower_bound, upper_bound)
    return data


def clip_above(data, threshold=100):
    data = np.clip(data, -threshold, threshold)
    return data


def plot_channel_histograms(data: np.ndarray, dataset_index):
    num_channels = data.shape[1]
    channels_per_figure = 16
    num_figures = (num_channels + channels_per_figure - 1) // channels_per_figure

    for fig_idx in range(num_figures):
        start_ch = fig_idx * channels_per_figure
        end_ch = min(start_ch + channels_per_figure, num_channels)
        fig, axs = plt.subplots(4, 4, figsize=(15, 10))
        axs = axs.flatten()

        for ch in range(start_ch, end_ch):
            # ch_data = remove_outliers(data[:, ch], percentile=5)
            if data.ndim == 3:
                data = data[:, :, 0]
            ch_data = clip_above(data[:, ch], threshold=100)
            # ch_data = data[:, ch]
            data_mean = np.mean(ch_data)
            data_std = np.std(ch_data)
            _, p = normaltest(ch_data)
            is_normal = p > 0.05
            ax: Axes = axs[ch - start_ch]
            ax.hist(ch_data, bins=200)
            ax.set_title(f"ds {dataset_index}, ch {ch}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.text(
                0.95,
                0.95,
                f"Mean: {data_mean:.5f}\nStd: {data_std:.5f}\nNormal: {is_normal}\np-value: {p:.3e}",
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
            )

        # Hide any unused subplots
        for ax in axs[end_ch - start_ch :]:
            ax.axis("off")

        plt.tight_layout()


if __name__ == "__main__":
    datamodule = superhuge.data.interface.DInterface(
        dataset_class=superhuge.data.datasets.EegRegressionBaseDataset,
        dataloader_args={"batch_size": 1, "num_workers": 0},
        root_path=r"E:/derivatives/SuperHuge",
        window_length=10,
        fs=128,
        meta_filter_func=[
            superhuge.data.metadata_filters.MetadataValueSelector(
                attribute_name="dataset_id", attribute_value=9
            )
        ],
        meta_filter_func_args=["env"],
        metadata_fields=[
            "env",
        ],
        # transform=superhuge.data.transforms.Scale(
        #     root_path=r"E:/derivatives/SuperHuge"
        # ),
    )

    eeg_list = []
    env_list = []
    for dataset in datamodule.datasets:
        for i in trange(len(dataset)):
            data = dataset.__getitem__(i)
            eeg_list.append(data["eeg"].copy())
            env_list.append(data["audio"].copy())
        # data = np.concatenate(eeg_list, axis=0)  # Concatenate along the time dimension
        # plot_channel_histograms(data, 0)
        data = np.concatenate(env_list, axis=0)  # Concatenate along the time dimension
        plot_channel_histograms(data[:, :, 0], 0)
    plt.show()
