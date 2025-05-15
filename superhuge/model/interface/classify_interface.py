import torch
import torchinfo
from torchmetrics import ConfusionMatrix

from ..loss.classify.auc_roc import multiclass_auc_score

from ..loss.classify.f1_score import macro_f1_score

from ..module.post_model import classify_post_model
from .model_interface import MInterface
from .channel_mapping_interface import (
    ChannelMapping1DInterface,
    ChannelMixing1DInterface,
)


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
        torchinfo.summary(self.post_model, input_size=self.output_size)

    def get_stats(self, pred: torch.Tensor, label: torch.Tensor, meta: dict):
        prob = pred
        pred = prob.argmax(dim=1)
        for sample_idx in range(pred.shape[0]):
            self.log_dict(
                {
                    # accuracy accumulated and reduced on each trial
                    f'detail/{self.stage}/{meta["entry"][sample_idx]}_acc': torch.Tensor(
                        pred[sample_idx] == label[sample_idx]
                    ).float(),
                    #         # accuracy accumulated and reduced on each subject
                    #         f'detail/{self.stage}/dataset-{meta["dataset_id"][sample_idx]:03d}-subject-{meta["subject_id"][sample_idx]:03d}_acc': torch.Tensor(
                    #             pred[sample_idx] == label[sample_idx]
                    #         ).float(),
                    # accuracy accumulated and reduced on each dataset
                    f'detail/{self.stage}/dataset-{meta["dataset_id"][sample_idx]:03d}_acc': torch.Tensor(
                        pred[sample_idx] == label[sample_idx]
                    ).float(),
                    # confusion matrix
                    # accuracy accumulated and reduced on each class
                    f"{self.stage}/{int(label[sample_idx])}_acc": torch.Tensor(
                        pred[sample_idx] == label[sample_idx]
                    ).float(),
                },
                batch_size=1,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                enable_graph=False,
            )
        self.log_dict(
            {
                f"{self.stage}/acc": (pred == label).float().mean(),
                f"{self.stage}/f1": macro_f1_score(
                    pred, label, num_classes=self.num_class
                ),
                f"{self.stage}/auc": multiclass_auc_score(prob, label),
            },
            prog_bar=True,
            batch_size=pred.shape[0],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            enable_graph=False,
        )


class ChannelMapping1DClassifyInterface(ChannelMapping1DInterface, ClassifyInterface):
    pass


class ChannelMixing1DClassifyInterface(ChannelMixing1DInterface, ClassifyInterface):
    pass


# class ChannelMapping2DClassifyInterface(ChannelMapping2DInterface,ClassifyInterface):
# pass
