from importlib import import_module
from typing import Callable
from keras import layers, Input, Model, KerasTensor, constraints, Layer
from einops import rearrange
import inspect


def preconv(input: KerasTensor, num_kernels: int, temporal_kernel_size: int):
    """
    Applies two convolutional layers followed by batch normalization and ELU activation.

    Args:
        input (KerasTensor): Input tensor.
        num_kernels (int): Number of kernels for the convolutional layers.
        temporal_kernel_size (int): Size of the temporal kernel.

    Returns:
        Model: Keras Model with the preconv layers applied.
    """
    x = layers.Conv2D(
        num_kernels,
        (1, temporal_kernel_size),
        padding="same",
        kernel_constraint=constraints.max_norm(2),
    )(input)
    x = layers.Conv2D(
        num_kernels,
        (input.shape[-1], 1),
        padding="valid",
        kernel_constraint=constraints.max_norm(2),
    )(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.ELU()(x)

    return Model(inputs=input, outputs=x)


def fg_cnn(input: KerasTensor, temporal_kernel_size: int, dp_rate: float):
    """
    Applies a dropout layer followed by a 1D convolutional layer, batch normalization, and ELU activation.

    Args:
        input (KerasTensor): Input tensor.
        temporal_kernel_size (int): Size of the temporal kernel.
        dp_rate (float): Dropout rate.

    Returns:
        Model: Keras Model with the fg_cnn layers applied.
    """
    x = layers.Dropout(dp_rate)(input)
    x = layers.Conv1D(
        filters=input.shape[1], kernel_size=temporal_kernel_size, padding="same"
    )(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.ELU()(x)

    return Model(inputs=input, outputs=x)


def feedforward(input: KerasTensor, hidden_dim: int, dp_rate: float):
    """
    Applies a feedforward neural network with dropout and ELU activation.

    Args:
        input (KerasTensor): Input tensor.
        hidden_dim (int): Dimension of the hidden layer.
        dp_rate (float): Dropout rate.

    Returns:
        Model: Keras Model with the feedforward layers applied.
    """
    x = layers.Dense(hidden_dim)(input)
    x = layers.ELU()(x)
    x = layers.Dropout(dp_rate)(x)
    x = layers.Dense(input.shape[-1])(x)
    x = layers.Dropout(dp_rate)(x)

    return Model(inputs=input, outputs=x)


def transformer(
    input: KerasTensor,
    depth: int,
    num_heads: int,
    dim_heads: int,
    fg_cnn_temporal_kernel_size: int,
    ff_hidden_dims: int,
    dp_rate: float,
):
    """
    Applies a transformer model with multi-head attention, fg_cnn, and feedforward layers.

    Args:
        input (KerasTensor): Input tensor.
        depth (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        dim_heads (int): Dimension of each attention head.
        fg_cnn_temporal_kernel_size (int): Size of the temporal kernel for fg_cnn.
        ff_hidden_dims (int): Dimension of the hidden layer in the feedforward network.
        dp_rate (float): Dropout rate.

    Returns:
        Model: Keras Model with the transformer layers applied.
    """
    x = input
    for i in range(depth):
        x_cg = x
        x_cg = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dim_heads, dropout=dp_rate
        )(x_cg)
        x_cg = layers.LayerNormalization()(x_cg + x)

        x_fg = fg_cnn(
            x, temporal_kernel_size=fg_cnn_temporal_kernel_size, dp_rate=dp_rate
        )(x)

        x = layers.LayerNormalization()(
            feedforward(x_cg, hidden_dim=ff_hidden_dims, dp_rate=dp_rate)(x_cg) + x_fg
        )

    return Model(inputs=input, outputs=x)


class pos_embedding(Layer):
    """
    Positional embedding layer that adds a trainable positional embedding to the input tensor.
    """

    def __init__(self):
        """
        Initializes the pos_embedding layer.
        """
        super().__init__()

    def build(self, input_shape):
        """
        Builds the pos_embedding layer by adding a trainable weight.

        Args:
            input_shape (tuple): Shape of the input tensor.
        """
        self.pos_embedding = self.add_weight(
            shape=input_shape[1:], trainable=True, name="pos_embedding"
        )

    def call(self, input):
        """
        Adds the positional embedding to the input tensor.

        Args:
            input (KerasTensor): Input tensor.

        Returns:
            KerasTensor: Tensor with positional embedding added.
        """
        return input + self.pos_embedding


def deformer(
    window_length: int,
    fs: int,
    num_kernels: int,
    temporal_kernel_size: int,
    mha_depth: int,
    mha_num_heads: int,
    mha_dim_heads: int,
    ff_hidden_dim: int,
    num_electrodes: int,
    dp_rate: float,
    preconv_callable: Callable[..., Model] | str | None = None,
    transformer_callable: Callable[..., Model] | str | None = None,
):
    """
    Constructs the deformer model with preconv, positional embedding, and transformer layers.

    Args:
        window_length (int): Length of the input window.
        fs (int): Sampling frequency.
        num_kernels (int): Number of kernels for the preconv layers.
        temporal_kernel_size (int): Size of the temporal kernel.
        mha_depth (int): Number of transformer layers.
        mha_num_heads (int): Number of attention heads.
        mha_dim_heads (int): Dimension of each attention head.
        ff_hidden_dim (int): Dimension of the hidden layer in the feedforward network.
        num_electrodes (int): Number of electrodes in the input data.
        dp_rate (float): Dropout rate.
        preconv_callable (Callable[..., Model] | None): Custom preconv function.
        transformer_callable (Callable[..., Model] | None): Custom transformer function.

    Returns:
        Model: Keras Model with the deformer architecture.
    """

    def validate_callable(callable_func: Callable, reference_func: Callable):
        if not callable(callable_func):
            raise TypeError(f"The provided argument {callable_func} is not callable")
        ref_signature = inspect.signature(reference_func)
        callable_signature = inspect.signature(callable_func)
        if ref_signature != callable_signature:
            raise TypeError(
                f"The provided callable {callable_func.__name__} does not match the required signature. \nGiven signature: {callable_signature}, expected signature {ref_signature}."
            )

    if preconv_callable is None:
        preconv_callable = preconv
    # elif isinstance(preconv_callable, str):
    #     module_name, func_name = preconv_callable.rsplit(".", 1)
    #     preconv_callable = getattr(import_module(module_name), func_name)
    if transformer_callable is None:
        transformer_callable = transformer
    # elif isinstance(transformer_callable, str):
    #     module_name, func_name = transformer_callable.rsplit(".", 1)
    #     transformer_callable = getattr(import_module(module_name), func_name)
    validate_callable(preconv_callable, preconv)
    validate_callable(transformer_callable, transformer)

    input = Input((window_length * fs, num_electrodes))
    x = layers.Lambda(lambda x: rearrange(x, "b t c -> b 1 c t"))(input)
    x = preconv_callable(x, num_kernels, temporal_kernel_size)(x)
    x = layers.Lambda(lambda x: rearrange(x, "b k c t -> b k (c t)"))(x)
    x = pos_embedding()(x)
    x = transformer_callable(
        x,
        mha_depth,
        mha_num_heads,
        mha_dim_heads,
        temporal_kernel_size,
        ff_hidden_dim,
        dp_rate,
    )(x)

    return Model(inputs=input, outputs=x)
