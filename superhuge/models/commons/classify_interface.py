import inspect
from turtle import onscreenclick

import torch
from torchmetrics import ConfusionMatrix

from .channel_mapping_interface import (
    ChannelMapping1DInterface,
    ChannelMapping2DInterface,
)
from .m_interface import MInterface
from .post_model import classify_post_model


class ClassifyInterface(MInterface):

    def __init__(self, /, *, num_class=2, **kwargs):
        # Get the signature of the parent __init__ method
        parent_signature = inspect.signature(super().__init__)

        # Validate the arguments against the parent's signature
        bound_arguments = parent_signature.bind(**kwargs)
        bound_arguments.apply_defaults()  # Ensure default values are included

        # Forward the validated arguments to the parent
        super().__init__(**bound_arguments.kwargs)

        self.confusion_matrix = ConfusionMatrix(
            task="multiclass",
            num_classes=num_class,
        )

        self.post_model = classify_post_model(self.input_size, num_class)

        self.label_hash: dict[str, int] = {}

    sig = inspect.signature(MInterface.__init__)
    __init__.__signature__ = sig.replace(
        parameters=list(sig.parameters.values())
        + [
            inspect.Parameter(
                "num_class", inspect.Parameter.KEYWORD_ONLY, default=2, annotation=int
            )
        ]
    )
    del sig

    def training_closure(self, data):
        outputs: torch.Tensor = self.forward(data)
        targets = data["label"]

        return outputs, targets

    def get_stats(self, pred: torch.Tensor, label: torch.Tensor, meta: dict):
        pred = pred.argmax(dim=1)
        # for sample_idx in range(pred.shape[0]):
        # self.log_dict(
        #     {
        #         # # accuracy accumulated and reduced on each trial
        #         f'detail/{self.stage}/{meta["entry"][sample_idx]}_acc': torch.Tensor(
        #             pred[sample_idx] == label[sample_idx]
        #         ).float(),
        #         # accuracy accumulated and reduced on each subject
        #         f'detail/{self.stage}/dataset-{meta["dataset_id"][sample_idx]:03d}-subject-{meta["subject_id"][sample_idx]:03d}_acc': torch.Tensor(
        #             pred[sample_idx] == label[sample_idx]
        #         ).float(),
        #         # accuracy accumulated and reduced on each dataset
        #         f'detail/{self.stage}/dataset-{meta["dataset_id"][sample_idx]:03d}_acc': torch.Tensor(
        #             pred[sample_idx] == label[sample_idx]
        #         ).float(),
        #         # confusion matrix
        # accuracy accumulated and reduced on each class
        # f"{self.stage}/{int(label[sample_idx])}_acc": torch.Tensor(
        #     pred[sample_idx] == label[sample_idx]
        # ).float(),
        #     },
        #     batch_size=pred.shape[0],
        #     prog_bar=False,
        # )
        self.log_dict(
            {
                f"{self.stage}/acc": (pred == label).float().mean(),
            },
            prog_bar=True,
            batch_size=pred.shape[0],
            on_epoch=True,
            on_step=False,
        )


class Channel1DClassifyInterface(ClassifyInterface, ChannelMapping1DInterface):
    pass


class Channel2DClassifyInterface(ClassifyInterface, ChannelMapping2DInterface):
    pass
