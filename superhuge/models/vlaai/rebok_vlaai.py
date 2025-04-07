"""Code to construct the VLAAI network."""

from typing import Annotated, Callable

import einops
import keras
from keras import Model
from pydantic import BaseModel, Field

from .vlaai import extractor, output_context


class VlaaiParams(BaseModel):
    nb_blocks: Annotated[int, Field(gt=0)] = 4
    extractor_model: Callable[..., Model] | None = None
    output_context_model: Callable[..., Model] | None = None
    use_skip: bool = True
    input_channels: Annotated[int, Field(gt=0)] = 64
    extractor_args: dict = {}
    output_context_args: dict = {}
    name: str = "vlaai"


def vlaai(
    nb_blocks: int = 4,
    window_length: int = 10,
    fs: int = 128,
    extractor_model: Callable[..., Model] | None = None,
    output_context_model: Callable[..., Model] | None = None,
    use_skip: bool = True,
    input_channels: int = 64,
    extractor_args: dict = {},
    output_context_args: dict = {},
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

    eeg = keras.layers.Input((window_length * fs, input_channels))

    reshaped_eeg = keras.layers.Lambda(
        einops.rearrange, arguments={"pattern": "b t c -> b c t"}
    )(eeg)

    if isinstance(nb_blocks, str):
        nb_blocks = int(nb_blocks)

    if extractor_model is None:
        extractor_model = extractor
    extractor_model = extractor_model(input_tensor=reshaped_eeg, **extractor_args)

    if output_context_model is None:
        output_context_model = output_context
    output_context_model = output_context_model(
        input_shape=(
            (
                extractor_args["num_kernels"][-1]
                if "num_kernels" in extractor_args
                else 128
            ),
            window_length * fs,
        ),
        **output_context_args,
    )

    # Iterate over the blocks
    for i in range(nb_blocks):
        if use_skip:
            if i == 0:
                x = extractor_model(reshaped_eeg)
            else:
                x = extractor_model(reshaped_eeg + x)
        else:
            x = extractor_model(x)
        x = output_context_model(x)
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
