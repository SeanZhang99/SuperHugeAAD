"""Code to construct the VLAAI network."""

import os
from typing import Callable, Sequence
from pydantic import BaseModel, Field
from typing import Annotated

os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import Model


class ExtractorParams(BaseModel):
    num_kernels: Annotated[Sequence[int], Field(min_length=1)]
    kernel_sizes: Annotated[Sequence[int], Field(min_length=1)]
    num_layers: Annotated[int | None, Field(gt=0)] = None
    input_channels: Annotated[int, Field(gt=0)] = 64
    normalization_fn: Callable | str | None = None
    activation_fn: Callable | str | None = None
    name: str = "extractor"


class OutputContextParams(BaseModel):
    num_kernel: Annotated[int, Field(gt=0)] = 64
    kernel_size: Annotated[int, Field(gt=0)] = 32
    input_channels: Annotated[int, Field(gt=0)] = 64
    normalization_fn: Callable | str | None = None
    activation_fn: Callable | str | None = None
    name: str = "output_context_model"


class VlaaiParams(BaseModel):
    nb_blocks: Annotated[int, Field(gt=0)] = 4
    extractor_model: Callable[..., Model] | None = None
    output_context_model: Callable[..., Model] | None = None
    use_skip: bool = True
    input_channels: Annotated[int, Field(gt=0)] = 64
    extractor_args: dict = {}
    output_context_args: dict = {}
    name: str = "vlaai"


def extractor(
    num_kernels: Sequence[int] = (256, 256, 256, 128, 128),
    kernel_sizes: Sequence[int] = (8,) * 5,
    num_layers: int | None = None,
    input_channels: int = 64,
    normalization_fn: Callable | str | None = None,
    activation_fn: Callable | str | None = None,
    name="extractor",
):
    params = ExtractorParams(
        num_kernels=num_kernels,
        kernel_sizes=kernel_sizes,
        num_layers=num_layers,
        input_channels=input_channels,
        normalization_fn=normalization_fn,
        activation_fn=activation_fn,
        name=name,
    )
    """Construct the extractor model.

    Parameters
    ----------
    num_kernels: Sequence[int]
        Number of num_kernels for each layer.
    kernel_sizes: Sequence[int]
        kernel_size size for each layer.
    input_channels: int
        Number of EEG channels in the input
    normalization_fn: Callable[[tf.Tensor], tf.Tensor]
        Function to normalize the contents of a tensor.
    activation_fn: Callable[[tf.Tensor], tf.Tensor]
        Function to apply an activation function to the contents of a tensor.
    name: str
        Name of the model.

    Returns
    -------
    tf.keras.models.Model
        The extractor model.
    """
    eeg = keras.layers.Input((None, input_channels))

    x = eeg
    assert (
        len(num_kernels) == len(kernel_sizes)
        or len(num_kernels) == 1
        or len(kernel_sizes) == 1
    ), f"num_kernels and kernel_sizes must: 1. have the same length, or 2. one of them contain only one element , but got {len(num_kernels)} and {len(kernel_sizes)}"
    num_kernels = list(num_kernels)
    kernel_sizes = list(kernel_sizes)

    if len(num_kernels) == 1:
        num_kernels = num_kernels * (num_layers if num_layers else len(kernel_sizes))
    if len(kernel_sizes) == 1:
        kernel_sizes = kernel_sizes * (num_layers if num_layers else len(num_kernels))

    if normalization_fn is None:
        normalization_fn = keras.layers.LayerNormalization
    elif isinstance(normalization_fn, str):
        normalization_fn = getattr(keras.layers, normalization_fn)

    if activation_fn is None:
        activation_fn = keras.layers.LeakyReLU
    elif isinstance(activation_fn, str):
        activation_fn = getattr(keras.layers, activation_fn)

    assert callable(
        normalization_fn
    ), f"normalization_fn must be a callable, but got {normalization_fn}"
    assert callable(
        activation_fn
    ), f"activation_fn must be a callable, but got {activation_fn}"

    # Add the convolutional layers
    for num_kernel, kernel_size in zip(num_kernels, kernel_sizes):
        x = keras.layers.Conv1D(num_kernel, kernel_size, padding="same")(x)
        x = normalization_fn()(x)
        x = activation_fn()(x)

    return keras.models.Model(inputs=eeg, outputs=x, name=name)


def output_context(
    num_kernel: int = 64,
    kernel_size: int = 32,
    input_channels: int = 64,
    normalization_fn: Callable | str | None = None,
    activation_fn: Callable | str | None = None,
    name="output_context_model",
):
    params = OutputContextParams(
        num_kernel=num_kernel,
        kernel_size=kernel_size,
        input_channels=input_channels,
        normalization_fn=normalization_fn,
        activation_fn=activation_fn,
        name=name,
    )
    """Construct the output context model.

    Parameters
    ----------
    num_kernel: int
        Number of num_kernels for the convolutional layer.
    kernel_size: int
        kernel_size size for the convolutional layer.
    input_channels: int
        Number of EEG channels in the input.
    normalization_fn: Callable[[tf.Tensor], tf.Tensor]
        Function to normalize the contents of a tensor.
    activation_fn: Callable[[tf.Tensor], tf.Tensor]
        Function to apply an activation function to the contents of a tensor.
    name: str
        Name of the model.

    Returns
    -------
    tf.keras.models.Model
        The output context model.
    """
    inp = keras.layers.Input((None, input_channels))

    if normalization_fn is None:
        normalization_fn = keras.layers.LayerNormalization
    elif isinstance(normalization_fn, str):
        normalization_fn = getattr(keras.layers, normalization_fn)

    if activation_fn is None:
        activation_fn = keras.layers.LeakyReLU
    elif isinstance(activation_fn, str):
        activation_fn = getattr(keras.layers, activation_fn)

    assert callable(
        normalization_fn
    ), f"normalization_fn must be a callable, but got {normalization_fn}"
    assert callable(
        activation_fn
    ), f"activation_fn must be a callable, but got {activation_fn}"

    x = keras.layers.ZeroPadding1D((kernel_size - 1, 0))(inp)  # type: ignore
    x = keras.layers.Conv1D(num_kernel, kernel_size)(x)
    x = normalization_fn()(x)
    x = activation_fn()(x)
    return keras.models.Model(inputs=inp, outputs=x, name=name)


def vlaai(
    nb_blocks: int = 4,
    extractor_model: Callable[..., Model] | None = None,
    output_context_model: Callable[..., Model] | None = None,
    use_skip: bool = True,
    input_channels: int = 64,
    extractor_args={},
    output_context_args={},
    name="vlaai",
):
    params = VlaaiParams(
        nb_blocks=nb_blocks,
        extractor_model=extractor_model,
        output_context_model=output_context_model,
        use_skip=use_skip,
        input_channels=input_channels,
        extractor_args=extractor_args,
        output_context_args=output_context_args,
        name=name,
    )
    """Construct the VLAAI model.

    Parameters
    ----------
    nb_blocks: int
        Number of repeated blocks to use.
    extractor_model: Callable[[tf.Tensor], tf.Tensor]
        The extractor model to use.
    output_context_model: Callable[[tf.Tensor], tf.Tensor]
        The output context model to use.
    use_skip: bool
        Whether to use skip connections.
    input_channels: int
        Number of EEG channels in the input.
    output_dim: int
        Number of output dimensions.
    name: str
        Name of the model.

    Returns
    -------
    tf.keras.models.Model
        The VLAAI model.
    """
    if isinstance(nb_blocks, str):
        nb_blocks = int(nb_blocks)

    if extractor_model is None:
        extractor_model = extractor
    extractor_model = extractor_model(input_channels=input_channels, **extractor_args)

    if output_context_model is None:
        output_context_model = output_context
    output_context_model = output_context_model(
        input_channels=(
            extractor_args["num_kernels"][-1]
            if "num_kernels" in extractor_args
            else 128
        ),
        **output_context_args,
    )

    eeg = keras.layers.Input((None, input_channels))

    x = eeg

    # Iterate over the blocks
    for i in range(nb_blocks):
        if use_skip:
            if i == 0:
                x = extractor_model(x)
            else:
                x = extractor_model(eeg + x)
        else:
            x = extractor_model(x)
        x = output_context_model(x)
        x = keras.layers.Dense(input_channels)(x)

    return keras.models.Model(inputs=eeg, outputs=x, name=name)


if __name__ == "__main__":
    nch = 16
    model = vlaai(
        nb_blocks=4,
        input_channels=nch,
        extractor_args={
            "num_kernels": [128] * 2 + [64] * 2,
            "kernel_sizes": [8] * 4,
        },
        output_context_args={"num_kernel": 32, "kernel_size": 16},
    )
    pass
