from abc import ABC, abstractmethod
import torch
from collections.abc import Sequence


class lsBaseABCModel(ABC):
    def __init__(self, /, **kwargs):
        """
        Initializes the model.
        """
        super().__init__(**kwargs)

    @abstractmethod
    def predict(
        self,
        x,
        y,
        /,
        weights: (
            torch.Tensor | float | int | Sequence[torch.Tensor | float | int] | None
        ) = None,
        bias: (
            torch.Tensor | float | int | Sequence[torch.Tensor | float | int] | None
        ) = None,
    ) -> Sequence[torch.Tensor]:
        """
        predict: abstract method. Expected inputs and outputs:

        ''Inputs'':
        'x': torch.Tensor
        'y': torch.Tensor

        ''keyword-only Inputs'':
        'weights': torch.Tensor | Sequence[float] | Sequence[int] | float | int | None
        The weight(s) used for linear regression. If None, the model's weights are used. If using a forward or backward model (such as wiener filter), weights should be a single Tensor, if using a model such as CCA or eigen filter, the weights should be a sequence of Tensors.
        'bias': torch.Tensor | float | int | Sequence[torch.Tensor | float | int]  | None

        ''Outputs'':
        'x_pred': torch.Tensor, prediction from x (not dthe predicted value of x)
        'y_pred': torch.Tensor, prediction from y (not the predicted value of y)
        if self.is_bidirectional:
            y_pred = y @ weights[0] + bias[0] if bias[0] else y @ weights[0]
            x_pred = x @ weights[1] + bias[1] if bias[1] else x @ weights[1]
        else:
            y_pred = t
            x_pred = x @ weights + bias if bias else x @ weights
            !note! always predict y from x, no matter what direction the model is (from eeg to speech, or from speech to eeg).
        return x_pred, y_pred
        """
        pass

    @abstractmethod
    def fit(
        self,
        /,
        x: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        covar_mtx: torch.Tensor | None = None,
    ) -> Sequence[torch.Tensor | Sequence[torch.Tensor]]:
        """
        Computes the weights for the model.
        Inputs:
        'x': torch.Tensor
        'y': torch.Tensor
        Or use:
        'covar_mtx': torch.Tensor
        ...: other covariance matrix/vector if you need

        Outputs:
        If bidirectional model (such as CCA): [[weight_x, weight_y, ...], [bias_x, bias_y,...] ...]
        If unidirectional model (such as Wiener filter): [weight, bias, ...]
        """
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass
