from .eeg_classify_base_dataset import EegClassifyBaseDataset


class EegClassifyDatasetWithSpectrum(EegClassifyBaseDataset):
    def __init__(self, **kwargs):
        """
        Args:
            spectrum_key (str): 提取谱图的元数据键，支持 'Pssl' 或 'Pmvdr'。
        """
        super().__init__(**kwargs)
        required_keys = ["spectrum_key"]
        self._validate_kwargs(kwargs, required_keys)
        self.spectrum_key = kwargs["spectrum_key"]
        assert (
            self.spectrum_key == "Pssl" or self.spectrum_key == "Pmvdr"
        ), f"{self.spectrum_key} is not supported currently."

    def __getitem__(self, idx):
        """
        加载样本数据，并返回元数据、信号段、标签和谱图。
        """
        meta, segment, label = super().__getitem__(idx).values()

        # 提取谱图
        spectrum = getattr(meta, self.spectrum_key, None)
        if spectrum is None:
            raise AttributeError(
                f"EEG_CLASSIFY_DATASET_WITH_SPECTRUM:__GETITEM__:GETATTR:ATTRIBUTE_ERROIR: Spectrum key '{self.spectrum_key}' not found in metadata for file {self.files[idx]}."
            )

        return {"meta": meta, "exg": segment, "label": label, "spectrum": spectrum}
