import stat
import torch
import torchinfo
from torchmetrics import ConfusionMatrix

from ..linear.classify.abc import ClassifierABC
from ..loss.classify.auc_roc import multiclass_auc_score
from ..loss.classify.f1_score import macro_f1_score
from ..module.post_model import classify_post_model
from .channel_mapping_interface import (
    ChannelMapping1DInterface,
    ChannelMixing1DInterface,
)
from .linear_interface import LinearInterface
from .model_interface import MInterface


class ClassifyInterface(MInterface):

    def __init__(self, /, *, num_class: int, hidden_dim: int, **kwargs):
        self.required_output_keys = ["eeg", "label"]
        super().__init__(**kwargs)
        self.num_class = num_class
        self.confusion_matrix = ConfusionMatrix(
            task="multiclass",
            num_classes=num_class,
        )
        self.post_model = classify_post_model(self.output_size, num_class, hidden_dim)
        torchinfo.summary(
            self.post_model, input_size=self.output_size, verbose=self.summary_verbose
        )

    def get_stats(self, pred: torch.Tensor, label: torch.Tensor, /, *, meta: dict):
        stats_dict = {}
        if pred.dtype is torch.float:
            prob, pred = pred, pred.argmax(dim=1)
            stats_dict[f"{self.stage}/auc"] = multiclass_auc_score(prob, label)

        stats_dict[f"{self.stage}/acc"] = (pred == label).float().mean()
        stats_dict[f"{self.stage}/f1"] = macro_f1_score(
            pred, label, num_classes=self.num_class
        )

        self.log_dict(
            stats_dict,
            prog_bar=True,
            batch_size=pred.shape[0],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            enable_graph=False,
        )

        return stats_dict


class ChannelMapping1DClassifyInterface(ChannelMapping1DInterface, ClassifyInterface):
    pass


class ChannelMixing1DClassifyInterface(ChannelMixing1DInterface, ClassifyInterface):
    pass


# class ChannelMapping2DClassifyInterface(ChannelMapping2DInterface,ClassifyInterface):
# pass


class LinearClassifyInterface(LinearInterface, ClassifyInterface):  # type: ignore
    model: ClassifierABC  # type: ignore

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.model.flush()

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.model.flush()

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.model.flush()
