from typing import Sequence
from torch import nn
from einops.layers.torch import Reduce, EinMix, Rearrange


def classify_post_model(
    input_size: Sequence[int | None], num_class: int, hidden_dim: int
):
    return nn.Sequential(
        Reduce("b t c -> b c", "mean"),
        # dense the last dimension
        nn.Linear(
            input_size[-1],
            hidden_dim,
        ),
        # nn.LayerNorm(hidden_dim),
        # nn.GELU(),
        nn.Sigmoid(),
        EinMix(
            "b hidden_dim -> b num_class",
            weight_shape="hidden_dim num_class",
            bias_shape="num_class",
            hidden_dim=hidden_dim,
            num_class=num_class,
        ),
    )


def regression_post_model(input_size: Sequence[int | None], num_features: int = 1):
    # If the last dimension of input_size is not equal to num_features, we need to add a linear layer to map it to num_features.
    # Otherwise, we can just use an identity layer.
    # This is useful for regression tasks where we want to predict a single value (num_features=1) from the input.
    return (
        nn.Sequential(
            EinMix(
                "b t c -> b t num_features",
                weight_shape="c num_features",
                c=input_size[-1],
                num_features=num_features,
            ),
        )
        if input_size[-1] != num_features
        else nn.Identity()
    )
