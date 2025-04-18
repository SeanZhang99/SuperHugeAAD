from torch.nn import Module, LayerNorm
from torch import Tensor


class LazyLayerNorm(Module):
    layer: LayerNorm

    def __init__(
        self, start_dim: int | None = None, stop_dim: int | None = None, *args, **kwargs
    ):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        assert start_dim or stop_dim, "Either start_dim or stop_dim must be provided"
        self.dims = slice(start_dim, stop_dim)

    def forward(self: Module, x: Tensor):
        if not hasattr(self, "layer"):
            self.layer = LayerNorm(
                normalized_shape=x.shape[self.dims],
                *self.args,
                **self.kwargs,
                device=x.device,
                dtype=x.dtype,
            )
        return self.layer(x)
