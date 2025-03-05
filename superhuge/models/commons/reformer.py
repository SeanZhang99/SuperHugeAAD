# This is the script of EEG-Deformer
# This is the network script
from collections.abc import Iterable

import einops
import torch
from einops import rearrange
from keras import layers, Model, Input
import keras

from .model_template import ModelTemplate


def cnn_block(in_chan, out_chan, kernel_size, num_chan, last_layer):
    input_layer = Input(shape=(None, None, in_chan))
    x = layers.Conv2D(
        out_chan,
        kernel_size,
        padding="same",
        kernel_constraint=keras.constraints.max_norm(2),
    )(input_layer)
    x = layers.Conv2D(
        out_chan,
        (num_chan, 1),
        padding="valid",
        kernel_constraint=keras.constraints.max_norm(2),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = (
        layers.Lambda(lambda x: einops.rearrange(x, "b f c t -> b c f t"))(x)
        if not last_layer
        else x
    )
    return Model(inputs=input_layer, outputs=x)


def create_models(
    skip_pool,
    symmetric_qk,
    dropout,
    num_kernels,
    temporal_kernel_size,
    mha_depth,
    mha_num_heads,
    mha_dim_heads,
    mha_hidden_dim,
    fc_num_neurons,
    fs,
    window_length,
    num_channel,
    num_class,
):
    input_eeg = Input(shape=(fs * window_length, num_channel))

    eeg = layers.Lambda(lambda x: einops.rearrange(x, "b t (f c) -> b f c t", f=1))(
        input_eeg
    )
    x = cnn_block(
        1, num_kernels, (1, temporal_kernel_size), num_channel, last_layer=True
    )(eeg)

    x = layers.Lambda(lambda x: einops.rearrange(x, "b k c f -> b k (c f)"))(x)
    pos_embedding = layers.Embedding(
        input_dim=num_kernels, output_dim=int(0.5 * fs * window_length)
    )(x)
    x = layers.Add()([x, pos_embedding])

    x = build_transformer(
        x,
        dropout,
        mha_depth,
        mha_num_heads,
        mha_dim_heads,
        mha_hidden_dim,
        num_kernels,
        temporal_kernel_size,
        skip_pool,
    )

    x = layers.Flatten()(x)
    x = layers.Dense(fc_num_neurons, activation="elu")(x)
    output = layers.Dense(num_class)(x)

    return Model(inputs=input_eeg, outputs=output)


def build_transformer(
    x,
    dropout,
    depth,
    num_heads,
    dim_heads,
    hidden_dim,
    num_kernels,
    temporal_kernel_size,
    skip_pool,
):
    dense_feature = []
    for i in range(depth):
        time_dim = (
            int(0.5 * x.shape[-1]) if ((i % 2 != 0) or not skip_pool) else x.shape[-1]
        )
        attn = attention_block(time_dim, num_heads, dim_heads, dropout, symmetric_qk)(x)
        ff = feedforward_block(time_dim, hidden_dim, dropout)(attn)
        cnn = cnn_block(
            num_kernels,
            temporal_kernel_size,
            dropout,
            use_max_pool=(False if ((i % 2 == 0) and skip_pool) else True),
        )(x)
        x_info = layers.Lambda(lambda x: torch.log(torch.mean(x.pow(2), dim=-1)))(cnn)
        dense_feature.append(x_info)
        x = layers.LayerNormalization()(ff + cnn)
    x_dense = layers.Concatenate()(dense_feature)
    return layers.Concatenate()([x, x_dense])


def feedforward_block(dim, hidden_dim, dropout=0.0):
    input_layer = Input(shape=(dim,))
    x = layers.Dense(hidden_dim)(input_layer)
    x = layers.GELU()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dim)(x)
    x = layers.Dropout(dropout)(x)
    return Model(inputs=input_layer, outputs=x)


def attention_block(dim, num_heads, dim_heads, dropout, symmetric_qk):
    input_layer = Input(shape=(dim,))
    inner_dim = dim_heads * num_heads
    project_out = not (num_heads == 1 and dim_heads == dim)

    scale = dim_heads**-0.5

    to_qkv = layers.Dense(
        inner_dim * 3 if not symmetric_qk else inner_dim * 2, use_bias=False
    )(input_layer)

    if not symmetric_qk:
        qkv = tf.split(to_qkv, 3, axis=-1)
        q, k, v = [
            tf.reshape(t, (-1, num_heads, t.shape[-2], t.shape[-1] // num_heads))
            for t in qkv
        ]
        dots = tf.matmul(q, k, transpose_b=True) * scale
    else:
        qv = tf.split(to_qkv, 2, axis=-1)
        q, v = [
            tf.reshape(t, (-1, num_heads, t.shape[-2], t.shape[-1] // num_heads))
            for t in qv
        ]
        dots = tf.matmul(q, q, transpose_b=True) * scale

    attn = tf.nn.softmax(dots, axis=-1)
    out = tf.matmul(attn, v)
    out = tf.reshape(out, (-1, out.shape[-2], out.shape[-1] * num_heads))

    if project_out:
        out = layers.Dense(dim)(out)
        out = layers.Dropout(dropout)(out)

    return Model(inputs=input_layer, outputs=out)


def transformer_block(
    dim,
    depth,
    num_heads,
    dim_heads,
    dropout,
    hidden_dim,
    num_kernels,
    temporal_kernel_size,
    skip_pool,
):
    input_layer = Input(shape=(dim,))
    dense_feature = []
    x = input_layer
    for i in range(depth):
        time_dim = int(0.5 * dim) if ((i % 2 != 0) or not skip_pool) else dim
        attn = attention_block(time_dim, num_heads, dim_heads, dropout, symmetric_qk)(x)
        ff = feedforward_block(time_dim, hidden_dim, dropout)(attn)
        cnn = cnn_block(
            num_kernels,
            temporal_kernel_size,
            dropout,
            use_max_pool=(False if ((i % 2 == 0) and skip_pool) else True),
        )(x)
        x_info = layers.Lambda(lambda x: torch.log(torch.mean(x.pow(2), dim=-1)))(cnn)
        dense_feature.append(x_info)
        x = layers.LayerNormalization()(ff + cnn)
    x_dense = layers.Concatenate()(dense_feature)
    x = layers.Reshape((x.shape[0], -1))(x)
    emd = layers.Concatenate()([x, x_dense])
    return Model(inputs=input_layer, outputs=emd)


def get_hidden_size(input_size, num_layer, skip_pool):
    return [
        int(input_size * (0.5 ** (i if not skip_pool else i // 2)))
        for i in range(num_layer + 1)
    ]


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def conv2d_with_constraint(*args, doWeightNorm=True, max_norm=1, **kwargs):
    input_layer = Input(shape=(None, None, args[0]))
    x = layers.Conv2D(*args, **kwargs)(input_layer)
    if doWeightNorm:
        x = layers.Lambda(lambda x: tf.clip_by_norm(x, max_norm, axes=[0]))(x)
    return Model(inputs=input_layer, outputs=x)
