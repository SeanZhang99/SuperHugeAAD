from typing import Sequence
from torch import nn
from einops.layers.torch import Reduce, EinMix, Rearrange


def classify_post_model(
    input_size: Sequence[int | None], num_class: int, hidden_dim: int
):
    assert input_size[-1] is not None, "Input size must have a defined last dimension"
    return nn.Sequential(
        # Average pooling across the time dimension if exists
        Reduce("b t c -> b c", "mean") if len(input_size) == 3 else nn.Identity(),
        # dense the last dimension
        nn.Linear(input_size[-1], hidden_dim),
        nn.Sigmoid(),
        nn.Linear(hidden_dim, num_class),
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
