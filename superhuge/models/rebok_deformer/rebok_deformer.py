from .deformer import *


def rebok_transformer(
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

    mha = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=dim_heads, dropout=dp_rate
    )
    fg_cnn_module = fg_cnn(
        input, temporal_kernel_size=fg_cnn_temporal_kernel_size, dp_rate=dp_rate
    )
    ff = feedforward(input, hidden_dim=ff_hidden_dims, dp_rate=dp_rate)
    x = input
    for i in range(depth):
        x_cg = x
        x_cg = mha(x_cg, x_cg)
        x_cg = layers.LayerNormalization()(x_cg + x)

        x_fg = fg_cnn_module(x)

        x = layers.LayerNormalization()(ff(x_cg) + x_fg)

    return Model(inputs=input, outputs=x)


def rebok_deformer(
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
    Constructs a deformer model with a transformer architecture. The transformer layer use a single mha module multiple times.

    Args:
        window_length (int): Length of the window.
        fs (int): Sampling frequency.
        num_kernels (int): Number of kernels.
        temporal_kernel_size (int): Size of the temporal kernel.
        mha_depth (int): Number of transformer layers.
        mha_num_heads (int): Number of attention heads.
        mha_dim_heads (int): Dimension of each attention head.
        ff_hidden_dim (int): Dimension of the hidden layer in the feedforward network.
        num_electrodes (int): Number of electrodes.
        dp_rate (float): Dropout rate.
        preconv_callable (Callable[..., Model]|str|None): Callable or string to create the preconv model.
        transformer_callable (Callable[..., Model]|str|None): Callable or string to create the transformer model.
    """
    if preconv_callable is None:
        preconv_callable = preconv
    if transformer_callable is None:
        transformer_callable = rebok_transformer

    return deformer(
        window_length=window_length,
        fs=fs,
        num_kernels=num_kernels,
        temporal_kernel_size=temporal_kernel_size,
        mha_depth=mha_depth,
        mha_num_heads=mha_num_heads,
        mha_dim_heads=mha_dim_heads,
        ff_hidden_dim=ff_hidden_dim,
        num_electrodes=num_electrodes,
        dp_rate=dp_rate,
        preconv_callable=preconv_callable,
        transformer_callable=transformer_callable,
    )
