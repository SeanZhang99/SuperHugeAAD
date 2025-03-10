from typing import Sequence
from keras import layers, Model, Input


def classify_hood(input_size: Sequence[int | None], num_class: int):
    input = Input(shape=input_size)
    x = layers.Flatten()(input)
    x = layers.Dense(num_class, activation="softmax")(x)
    return Model(inputs=input, outputs=x)


def regression_hood(input_size: Sequence[int | None]):
    input = Input(shape=input_size)
    x = layers.Dense(1)(input)
    x = layers.Flatten()(x)
    return Model(inputs=input, outputs=x)
