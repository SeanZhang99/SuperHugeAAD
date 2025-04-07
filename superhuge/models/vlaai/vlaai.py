"""Code to construct the VLAAI network."""

import os
import einops

os.environ["KERAS_BACKEND"] = "torch"
from typing import Annotated, Callable, Sequence

import keras
from pydantic import BaseModel, Field

keras.config.set_image_data_format("channels_first")
from keras import Model


class ExtractorParams(BaseModel):
    num_kernels: Annotated[Sequence[int], Field(min_length=1)]
    kernel_sizes: Annotated[Sequence[int], Field(min_length=1)]
    num_layers: Annotated[int | None, Field(gt=0)] = None
    normalization_fn: Callable | str | None = None
    activation_fn: Callable | str | None = None
    name: str = "extractor"


class OutputContextParams(BaseModel):
    num_kernel: Annotated[int, Field(gt=0)] = 64
    kernel_size: Annotated[int, Field(gt=0)] = 32
    normalization_fn: Callable | str | None = None
    activation_fn: Callable | str | None = None
    name: str = "output_context_model"


class VlaaiParams(BaseModel):
    nb_blocks: Annotated[int, Field(gt=0)] = 4
    use_skip: bool = True
    input_channels: Annotated[int, Field(gt=0)] = 64
    extractor_args: dict = {}
    output_context_args: dict = {}
    name: str = "vlaai"


def extractor(
    num_kernels: Sequence[int] = (256, 256, 256, 128, 128),
    kernel_sizes: Sequence[int] = (8,) * 5,
    num_layers: int | None = None,
    input_shape: tuple[int] | None = None,
    input_tensor: keras.KerasTensor | None = None,
    normalization_fn: Callable | str | None = None,
    activation_fn: Callable | str | None = None,
    name="extractor",
):
    params = ExtractorParams(
        num_kernels=num_kernels,
        kernel_sizes=kernel_sizes,
        num_layers=num_layers,
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
    if input_shape is not None:
        eeg = keras.layers.Input(input_shape)
    elif input_tensor is not None:
        eeg = keras.layers.Input(input_tensor.shape[1:])

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
        x = keras.layers.ZeroPadding1D((kernel_size - 1, 0))(x)
        x = keras.layers.Conv1D(num_kernel, kernel_size, padding="valid")(x)
        x = normalization_fn()(x)
        x = activation_fn()(x)

    return keras.models.Model(inputs=eeg, outputs=x, name=name)


def output_context(
    num_kernel: int = 64,
    kernel_size: int = 32,
    input_shape: tuple[int] | None = None,
    input_tensor: keras.KerasTensor | None = None,
    normalization_fn: Callable | str | None = None,
    activation_fn: Callable | str | None = None,
    name="output_context_model",
):
    params = OutputContextParams(
        num_kernel=num_kernel,
        kernel_size=kernel_size,
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
    if input_shape is not None:
        inp = keras.layers.Input(input_shape)
    elif input_tensor is not None:
        inp = keras.layers.Input(input_tensor.shape[1:])

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
    window_length: int = 10,
    fs: int = 128,
    use_skip: bool = True,
    input_channels: int = 64,
    extractor_args: dict = {},
    output_context_args: dict = {},
    name="vlaai",
):
    params = VlaaiParams(
        nb_blocks=nb_blocks,
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

    eeg = keras.layers.Input((window_length * fs, input_channels))

    reshaped_eeg = keras.layers.Lambda(
        einops.rearrange, arguments={"pattern": "b t c -> b c t"}
    )(eeg)

    if isinstance(nb_blocks, str):
        nb_blocks = int(nb_blocks)

    x = reshaped_eeg

    # Iterate over the blocks
    for i in range(nb_blocks):
        if use_skip:
            if i == 0:
                x = extractor(input_tensor=x, **extractor_args, name=f"extractor_{i}")(
                    x
                )
            else:
                x = extractor(input_tensor=x, **extractor_args, name=f"extractor_{i}")(
                    reshaped_eeg + x
                )
        else:
            x = extractor(input_tensor=x, **extractor_args, name=f"extractor_{i}")(x)
        x = output_context(
            input_tensor=x, **output_context_args, name=f"output_context_{i}"
        )(x)
        x = keras.layers.Lambda(
            einops.rearrange, arguments={"pattern": "b c t -> b t c"}
        )(x)
        x = keras.layers.Dense(input_channels)(x)
        if i < nb_blocks - 1:
            x = keras.layers.Lambda(
                einops.rearrange, arguments={"pattern": "b t c -> b c t"}
            )(x)

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
