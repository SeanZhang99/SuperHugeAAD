from typing import Sequence
from torch import nn
import numpy as np
import einops
from einops.layers import torch as layers

from .lambda_layer import LambdaLayer


def classify_post_model(input_size: Sequence[int | None], num_class: int):
    # input = Input(shape=input_size)
    # x = layers.Flatten()(input)
    # x = layers.Dense(num_class)(x)
    # return Model(inputs=input, outputs=x)
    return nn.Sequential(
        layers.Reduce("b t c -> b c", reduction="mean"),
        layers.EinMix(
            "b c -> b num_class",
            weight_shape="c num_class",
            c=input_size[-1],
            num_class=num_class,
        ),
    )


def regression_post_model(input_size: Sequence[int | None]):
    # input = Input(shape=input_size)
    # x = layers.Dense(1)(input)
    # x = layers.Flatten()(x)
    # return Model(inputs=input, outputs=x)
    return nn.Sequential(
        layers.EinMix(
            "b t c -> b t num_features",
            weight_shape="c num_features",
            c=input_size[-1],
            num_features=1,
        ),
        LambdaLayer(lambda x: x.squeeze(-1)),
    )
