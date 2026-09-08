"""Tests for the lag window ``lags: [start, end]`` support.

Requires a torch-capable environment (model-level tests import ``superhuge``).
Run with e.g. the ``keras+torch+pl`` conda env::

    python -m pytest tests/test_lag_window.py -v
"""
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from superhuge.model.linear.regression.abc import LinearABC  # noqa: E402
from superhuge.model.linear.regression.cca import CCA, CCAConfig  # noqa: E402
from superhuge.model.linear.regression.forward_wf import (  # noqa: E402
    WienerFilterConfig as ForwardWienerFilterConfig,
)
from superhuge.model.linear.regression.wf import (  # noqa: E402
    WienerFilterConfig,
)
from superhuge.model.linear.regression.wf_per_subject import (  # noqa: E402
    PerSubjectWienerFilter,
)

FS = 128


class _Dummy(LinearABC):
    """Minimal LinearABC subclass to exercise get_lag_mtx* on raw tensors."""

    def __init__(self):
        super().__init__()

    def update(self, eeg, env, **kwargs):
        super().update(eeg, env)

    def fit(self):
        pass

    def predict(self, eeg, env):
        return eeg, env


def _dummy():
    m = _Dummy()
    x = torch.randn(2, 500, 8)
    return m, x


# ------------------------------------------------------------ lag matrix


def test_get_lag_mtx_range_equals_legacy():
    m, x = _dummy()
    pre, post = 10, 25
    legacy = m.get_lag_mtx(x, pre, post)
    rng = m.get_lag_mtx_range(x, -pre, post)
    assert legacy.shape == rng.shape == (2, 500, pre + post + 1, 8)
    assert torch.equal(legacy, rng)


def test_get_lag_mtx_range_shift_semantics():
    m, x = _dummy()
    # shift +3: buf[t] = x[t+3] -> buf[0..T-4] holds x[3..T-1]; tail zeroed
    buf = m.get_lag_mtx_range(x, 3, 3)
    assert torch.equal(buf[:, :-3, 0, :], x[:, 3:, :])
    assert torch.count_nonzero(buf[:, -3:, 0, :]) == 0
    # shift -2: buf[t] = x[t-2]  -> buf[2..T-1] holds x[0..T-3]; head zeroed
    buf = m.get_lag_mtx_range(x, -2, -2)
    assert torch.equal(buf[:, 2:, 0, :], x[:, :-2, :])
    assert torch.count_nonzero(buf[:, :2, 0, :]) == 0
    # shift 0: buf = x
    buf = m.get_lag_mtx_range(x, 0, 0)
    assert torch.equal(buf[:, :, 0, :], x)


# ------------------------------------------------------------ config: WF


def test_lags_samples_conversion():
    cfg = WienerFilterConfig(lags=[0.1, 0.4], l2=0.0, fs=FS,
                             window_length=6400, num_channels=64)
    assert cfg.lag_start_samples == 12
    assert cfg.lag_end_samples == 51
    assert cfg.nlag == 40
    assert cfg.pre_lag is None and cfg.post_lag is None


def test_lags_negative_start():
    cfg = WienerFilterConfig(lags=[-0.1, 0.9], l2=0.0, fs=FS,
                             window_length=6400, num_channels=64)
    assert cfg.lag_start_samples == -12
    assert cfg.lag_end_samples == 115
    assert cfg.nlag == 128


def test_legacy_pre_post_unchanged():
    cfg = WienerFilterConfig(pre_lag=0.1, post_lag=0.4, l2=0.0, fs=FS,
                             window_length=6400, num_channels=64)
    assert cfg.pre_lag == 12 and cfg.post_lag == 51  # seconds -> samples as before
    assert cfg.lag_start_samples == -12 and cfg.lag_end_samples == 51
    assert cfg.nlag == 64  # = 12 + 51 + 1


def test_forward_wf_config_lags():
    cfg = ForwardWienerFilterConfig(lags=[0.1, 0.4], l2=0.0, fs=FS,
                                    num_features=1, num_channels=64)
    assert cfg.lag_start_samples == 12 and cfg.lag_end_samples == 51
    assert cfg.nlag == 40


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(pre_lag=0.1, post_lag=0.4, lags=[0.1, 0.4]),  # both modes
        dict(pre_lag=0.1),  # pre without post
        dict(post_lag=0.1),  # post without pre
        dict(lags=[0.1]),  # lags wrong length
        dict(lags=[0.5, 0.1]),  # end <= start
        dict(lags=["a", 0.4]),  # non-numeric
        dict(lags=[True, 0.4]),  # bool not allowed
        dict(),  # no lag mode at all
    ],
)
def test_lag_mode_validation_errors(kwargs):
    with pytest.raises(ValueError):
        WienerFilterConfig(l2=0.0, fs=FS, window_length=6400, num_channels=64,
                           **kwargs)


# ------------------------------------------------------- model end-to-end


def test_per_subject_lags_end_to_end():
    torch.manual_seed(0)
    m = PerSubjectWienerFilter(lags=[0.1, 0.4], l2=0.0, use_lwcov=False,
                               fs=FS, num_channels=64, window_length=6400)
    eeg = torch.randn(2, 2000, 64)
    env = torch.randn(2, 2000, 1, 2)
    meta = {"dataset_id": torch.tensor([1, 1]), "subject_id": torch.tensor([1, 2])}
    m.update(eeg, env, meta=meta)
    m.fit()
    assert m.weights.shape == (40 * 64, 1)
    y_pred, env_out = m.predict(eeg, env)
    assert y_pred.shape == (2, 2000, 1)


def test_legacy_pre_post_equals_equivalent_lags():
    torch.manual_seed(0)
    eeg = torch.randn(2, 2000, 64)
    env = torch.randn(2, 2000, 1, 2)
    meta = {"dataset_id": torch.tensor([1, 1]), "subject_id": torch.tensor([1, 2])}

    m_old = PerSubjectWienerFilter(pre_lag=0.1, post_lag=0.4, l2=0.0,
                                   use_lwcov=False, fs=FS, num_channels=64,
                                   window_length=6400)
    m_old.update(eeg, env, meta=meta)
    m_old.fit()

    m_eq = PerSubjectWienerFilter(lags=[-0.1, 0.4], l2=0.0, use_lwcov=False,
                                  fs=FS, num_channels=64, window_length=6400)
    m_eq.update(eeg, env, meta=meta)
    m_eq.fit()

    assert m_old.weights.shape == m_eq.weights.shape == (64 * 64, 1)
    assert torch.allclose(m_old.weights, m_eq.weights, atol=1e-9)


# ------------------------------------------------------------ CCA


def test_cca_config_lags_conversion():
    cfg = CCAConfig(x_lags=[0.1, 0.4], y_lags=[-0.05, 0.05], fs=FS, l2=0.0,
                    num_features_x=64, num_features_y=1, num_components=2)
    assert (cfg.x_lag_start_samples, cfg.x_lag_end_samples) == (12, 51)
    assert (cfg.y_lag_start_samples, cfg.y_lag_end_samples) == (-6, 6)
    assert (cfg.x_lag_end_samples - cfg.x_lag_start_samples + 1) == 40


def test_cca_legacy_sec_equals_symmetric_lags():
    cfg_sec = CCAConfig(x_lag_sec=0.1, y_lag_sec=0.05, fs=FS, l2=0.0,
                        num_features_x=64, num_features_y=1, num_components=2)
    cfg_lags = CCAConfig(x_lags=[-0.1, 0.1], y_lags=[-0.05, 0.05], fs=FS, l2=0.0,
                         num_features_x=64, num_features_y=1, num_components=2)
    assert cfg_sec.x_lag_start_samples == cfg_lags.x_lag_start_samples == -12
    assert cfg_sec.x_lag_end_samples == cfg_lags.x_lag_end_samples == 12
    assert cfg_sec.y_lag_start_samples == cfg_lags.y_lag_start_samples == -6
    assert cfg_sec.y_lag_end_samples == cfg_lags.y_lag_end_samples == 6


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(x_lag_sec=0.1, x_lags=[0.1, 0.4]),  # x both modes
        dict(y_lag_sec=0.1, y_lags=[0.1, 0.4]),  # y both modes
        dict(x_lags=[0.5, 0.1]),  # end < start
        dict(x_lags=[0.1]),  # wrong length
        dict(x_lag_sec=None, y_lag_sec=0.05),  # x missing both
    ],
)
def test_cca_lag_mode_validation_errors(kwargs):
    with pytest.raises(ValueError):
        CCAConfig(fs=FS, l2=0.0, num_features_x=64, num_features_y=1,
                  num_components=2, **kwargs)


def test_cca_lags_end_to_end():
    torch.manual_seed(0)
    m = CCA(x_lags=[0.1, 0.4], y_lags=[-0.05, 0.05], l2=0.0, num_components=2,
            fs=FS, num_channels=64, num_audio_features={"env": 1})
    eeg = torch.randn(2, 2000, 64)
    env = torch.randn(2, 2000, 1, 2)
    m.update(eeg, env)
    m.fit()
    xp, yp = m.predict(eeg, env)
    assert xp.shape == (2, 2000, 2)
    assert yp.ndim == 4
