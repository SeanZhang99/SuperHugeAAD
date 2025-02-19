from collections.abc import Callable
import torch


def zeros(eeg: torch.Tensor, arg: int = 0) -> torch.Tensor:
    return torch.zeros_like(eeg)


def randn(eeg: torch.Tensor, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn_like(eeg)


def eeg_parser(_file: str) -> Callable[[torch.Tensor, int], torch.Tensor] | None:
    if "EEG" not in _file:
        if "Noise" in _file:
            return randn
        else:
            return zeros
    else:
        return None
