import os
from collections.abc import Callable

from scipy.io import loadmat

from .eeg_dataset import CreateDatasetsInputConfig, EegDataset
from .metadata_processing import RegressionMetaDataElement
from .regression_filter import ALLOWED_SPEECH_FEATURES, get_regression_filter


ENV_ALIASE = ["env", "envelope"]
MEL_ALIASE = ["mel", "mel spectrum", "mfcc"]


class EEGDatasetWithSpeechFeatureCreationConfig(CreateDatasetsInputConfig):
    speech_feature_path: str
    speech_feature_key: str


class EegRegressionBaseDataset(EegDataset):
    metadata_cls = RegressionMetaDataElement

    def __init__(self, **kwargs):
        """
        Args:
            speech_feature_path (str): 语音特征文件夹路径。
            speech_feature_key (str): 元数据中存储语音特征文件名的键。
        """
        super().__init__(**kwargs)
        config = EEGDatasetWithSpeechFeatureCreationConfig(**kwargs)

        # 处理支持的语音特征别名
        feature_type = config.speech_feature_key.lower()
        if feature_type in ENV_ALIASE:
            self.speech_feature_type = "env"
        elif feature_type in MEL_ALIASE:
            self.speech_feature_type = "mel"
        else:
            raise ValueError(
                f"EEG_REGRESSION_BASE_DATASET:__INIT__:VALUE_ERROR: Unsupported speech feature: {feature_type}. Supported values are {ENV_ALIASE+MEL_ALIASE}."
            )

        self.speech_feature_path = os.path.join(
            self.exg_path.replace("exg", "stimuli"), self.speech_feature_type
        )

    def __getitem__(self, idx):
        """
        加载样本数据，并返回元数据、EEG段、语音特征段和标签。
        """
        meta, exg = super().__getitem__(idx).values()

        # 获取语音特征文件名
        speech_feature_file = meta[self.speech_feature_type]

        # 加载语音特征
        speech_feature_path = os.path.join(
            self.speech_feature_path, speech_feature_file
        )
        mat_data = loadmat(speech_feature_path)

        # 根据 key 提取特征
        speech_feature = mat_data[self.speech_feature_type]

        # 根据 segment_length 和 overlap 截取语音特征段
        _, segment_idx = self._map_idx_to_file_and_segment(idx)
        stride = self.segment_length // self.overlap
        start_idx = segment_idx * stride
        speech_segment = speech_feature[start_idx : start_idx + self.segment_length, :]

        return {"meta": meta, "exg": exg, "audio": speech_segment}

    @classmethod
    def meta_filter_func_parser(
        cls,
        meta_filter_func: (
            Callable[
                [
                    RegressionMetaDataElement,
                ],
                RegressionMetaDataElement | None,
            ]
            | None
        ),
        *args,
        **kwargs,
    ):
        assert (
            len(args) > 0 or "target" in kwargs
        ), "EEG_REGRESSION_BASE_DATASET:META_FILTER_FUNC_PARSER:ASSERTION:INPUT_ARGUMENT_ERROR: target must be specified at the first positional argument or as a keyword argument"
        target = args[0] if len(args) > 0 else kwargs["target"]

        # Validate target is a valid argument
        assert (
            target in ALLOWED_SPEECH_FEATURES
        ), f"EEG_REGRESSION_BASE_DATASET:META_FILTER_FUNC_PARSER:ASSERTION:TARGET:VALUE_ERROR: target must be a valid value from ALLOWED_SPEECH_FEATURES: {ALLOWED_SPEECH_FEATURES}"

        if meta_filter_func is None:
            meta_filter_func = get_regression_filter(target)
        return meta_filter_func
