from einops import einsum, rearrange
from pyriemann.utils.mean import mean_riemann, mean_logeuclid
import torch
from ...types import AUDIO_TYPE, EEG_TYPE
from .wf import WienerFilter


class RiemannianWienerFilter(WienerFilter):
    def __init__(self, /, **kwargs):
        super().__init__(**kwargs)
        self.Rxx_list = []

    def update(self, eeg: EEG_TYPE, audio: AUDIO_TYPE) -> None:

        super(WienerFilter, self).update(eeg, audio)

        x_lag = self.lag_and_flatten(
            eeg,
            "batch lag time channel -> (batch time) (lag channel)",
            self.cfg.pre_lag,  # type: ignore
            self.cfg.post_lag,  # type: ignore
        )
        y = rearrange(
            audio[..., 0],
            "batch time num_features -> (batch time) num_features",
            num_features=1,
        )

        self.rxy += x_lag.T @ y

        rxx = einsum(x_lag, x_lag, "b k, b l -> b k l") + torch.eye(
            self.cfg.nlag * self.cfg.num_channels, device=x_lag.device
        )

        self.Rxx_list.append(rxx)

    def fit(self) -> None:
        """
        Fit the model to the data.
        """
        assert not self._fitted, "Model is already fitted."
        assert self._n_samples > 0, "No data to fit the model."
        self.Rxx = torch.from_numpy(
            mean_riemann(torch.concat(self.Rxx_list, dim=0).cpu().numpy())
        ).to(device=self.rxy.device, dtype=self.rxy.dtype)
        self.Rxx += self.cfg.l2 * torch.eye(
            self.cfg.nlag * self.cfg.num_channels, device=self.Rxx.device
        )
        self.weights = torch.linalg.solve(self.Rxx, self.rxy * self._n_samples).detach()
        self._fitted = True
