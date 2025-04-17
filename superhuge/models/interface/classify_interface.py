import torch
from torchmetrics import ConfusionMatrix

from .channel_mapping_interface import (
    ChannelMapping1DInterface,
    ChannelMapping2DInterface,
    ChannelMixing1DInterface,
)
from .m_interface import MInterface
from ..tools.post_model import classify_post_model


class ClassifyInterface(MInterface):

    def __init__(self, /, *, num_class: int, hidden_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.confusion_matrix = ConfusionMatrix(
            task="multiclass",
            num_classes=num_class,
        )
        self.post_model = classify_post_model(self.input_size, num_class, hidden_dim)

    def training_closure(self, data):
        outputs: torch.Tensor = self.forward(data)
        targets = data["label"].to(torch.long)

        return outputs, targets

    def get_stats(self, pred: torch.Tensor, label: torch.Tensor, meta: dict):
        pred = pred.argmax(dim=1)
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
            },
            prog_bar=True,
            batch_size=pred.shape[0],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            enable_graph=False,
        )


class Channel1DMappingClassifyInterface(ChannelMapping1DInterface, ClassifyInterface):
    pass


class Channel1DMixingClassifyInterface(ChannelMixing1DInterface, ClassifyInterface):
    pass


class Channel2DClassifyInterface(ClassifyInterface, ChannelMapping2DInterface):
    pass
