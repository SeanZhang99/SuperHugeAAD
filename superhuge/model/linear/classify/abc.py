from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import final, TYPE_CHECKING
import torch
from einops import rearrange


if TYPE_CHECKING:
    from superhuge.model.types import EEG_TYPE, LABEL_TYPE
    from sklearn.base import ClassifierMixin
    from typing import Literal

    CODING_STRATEGY = Literal["onevsone", "onevsall", "onevsrest" "ovo", "ova", "ovr"]


class ClassifierABC(torch.nn.Module, ABC):
    _fitted: bool = False
    _n_samples: int = 0
    classifier: "ClassifierMixin"

    @abstractmethod
    def __init__(
        self,
        /,
        classifier: "ClassifierMixin",
        coding_strategy: "CODING_STRATEGY",
        **kwargs,
    ):
        super().__init__()
        self.classifier = classifier
        self.coding_strategy = coding_strategy

    @abstractmethod
    def estimate_feature(
        self,
        eeg: "EEG_TYPE",
        label: "LABEL_TYPE",
    ) -> torch.Tensor:
        """
        Estimate features from EEG data.
        Might have different logics for unfitted (self._fitted=False) and fitted (self._fitted=True) classifiers.
        E.g. unfitted: compute covariance matrix, store the matrix, convert data to features. fitted: convert to features

        Parameters:
        eeg: torch.Tensor, input EEG data. Shape: (batch_size, time_steps, channels).

        Returns:
        torch.Tensor, estimated features. Shape: (batch_size, feature_dim).
        """
        ...

    def update(self, eeg: "EEG_TYPE", labels: "LABEL_TYPE") -> None:
        """
        Update the classifier with new data.

        Parameters:
        eeg: torch.Tensor, input EEG data. Shape: (batch_size, time_steps, channels).
        labels: torch.Tensor, labels for the input data. Shape: (batch_size,).

        Returns:
        None
        """
        if not hasattr(self, "eeg"):
            self.eeg = eeg
            self.labels = labels
        else:
            self.eeg = torch.cat((self.eeg, eeg), dim=0)
            self.labels = torch.cat((self.labels, labels), dim=0)
        self._n_samples += eeg.shape[0]

    def fit(self):
        """
        Fit the classifier with the accumulated features and labels.

        Returns:
        self: The fitted classifier.
        """
        assert (
            self._n_samples > 0
        ), "No samples to fit the classifier. Please call update() first."
        assert (
            not self._fitted
        ), "Classifier has already been fitted. Please call update() to add more data."
        assert hasattr(self, "eeg") and hasattr(
            self, "labels"
        ), "EEG and labels must be set before fitting."
        assert hasattr(self.classifier, "fit"), "Classifier must have a fit method."
        features = self.estimate_feature(self.eeg, self.labels)
        self.classifier.fit(features, self.labels.cpu().numpy())
        self._fitted = True

    def predict(
        self, eeg: "EEG_TYPE", label: "LABEL_TYPE"
    ) -> tuple["EEG_TYPE", "LABEL_TYPE"]:
        """
        Predict labels for the input EEG data.

        Parameters:
        eeg: torch.Tensor, input EEG data. Shape: (batch_size, time_steps, channels).
        label: torch.Tensor, labels for the input data. Shape: (batch_size,).

        Returns:
        LABEL_TYPE, predicted labels. Shape: (batch_size,).
        """
        assert self._fitted, "Classifier must be fitted before predicting."
        assert hasattr(
            self.classifier, "predict"
        ), "Classifier must have a predict method."
        features = self.estimate_feature(eeg)
        predictions = self.classifier.predict(features.cpu().numpy())
        return features, torch.tensor(predictions, dtype=label.dtype).to(label.device)

    def forward(
        self, eeg: "EEG_TYPE", label: "LABEL_TYPE"
    ) -> tuple["EEG_TYPE", "LABEL_TYPE"]:
        """
        Forward pass of the model.

        Parameters:
        eeg: torch.Tensor, input EEG data. Shape: (batch_size, time_steps, channels).
        label: torch.Tensor, labels for the input data. Shape: (batch_size,).

        Returns:
        tuple of EEG and predicted labels.
        """
        if self._fitted:
            eeg, label = self.predict(eeg, label)
        else:
            self.update(eeg, label)
        return eeg, label

    @final
    def get_lag_mtx(self, x: torch.Tensor, *lag: int, **kwargs):
        """
        Construct a lagged matrix for the input with given lag.

        Parameters:
        x: torch.Tensor, input tensor. Shape: (batch_size, time_steps, ...). The lag (or advance) is operated along the time dimension, and lagged signals is inseted into the second dimension. Other dimensions remain unchanged.
        lag: two integers, the lag range. The first element specify where the lag begins, the second element specify where the lag ends. The first lag indicates advance, and the second indicates delay.

        Example:
        lag: 10, 10. return: (batch_size, 21, time_steps, ...). 10 means the input is advanced by 10 time steps, 10 means the input is lagged by 10 time steps.
        """
        x_lag = []
        assert (
            isinstance(lag, Sequence) and len(lag) == 2
        ), "lag must be a sequence of two integers"

        if lag[0] <= 0 and lag[1] >= 0:
            lag = (-lag[0], lag[1])

        for l in range(-lag[0], lag[1] + 1):
            if l >= 0:
                x_lag.append(
                    torch.cat(
                        [
                            torch.zeros((x.shape[0], l, *x.shape[2:])).type_as(x),
                            x[:, : x.shape[1] - l],
                        ],
                        dim=1,
                    )
                )
            elif l < 0:
                x_lag.append(
                    torch.cat(
                        [
                            x[:, -l:, :],
                            torch.zeros(x.shape[0], -l, *x.shape[2:]).type_as(x),
                        ],
                        dim=1,
                    )
                )
        x_lag = torch.stack(
            x_lag,
            dim=1,
        )
        return x_lag

    def lag_and_flatten(
        self, signal: torch.Tensor, pattern: str, *lag_samples: int
    ) -> torch.Tensor:
        """
        Create lagged matrix and flatten it.

        Args:
            signal: Input tensor [batch, time, features]
            lag_samples: Number of lag samples
            num_features: Number of features

        Returns:
            Flattened lagged matrix [batch*time, lag*features]
        """
        lagged_matrix = self.get_lag_mtx(signal, lag_samples[0], lag_samples[1])
        return rearrange(
            lagged_matrix,
            pattern,
        )
