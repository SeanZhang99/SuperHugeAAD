import torch
from torch import nn
from typing import Type, Union


# 类型限制：支持哪些卷积类
NN_CONV_CLASS = Union[
    Type[nn.Conv1d],
    Type[nn.Conv2d],
    Type[nn.Conv3d],
    Type[nn.ConvTranspose1d],
    Type[nn.ConvTranspose2d],
    Type[nn.ConvTranspose3d],
]
ND_TRANSPOSE_TO_CONV_CLASS = {
    (1, False): nn.Conv1d,
    (2, False): nn.Conv2d,
    (3, False): nn.Conv3d,
    (1, True): nn.ConvTranspose1d,
    (2, True): nn.ConvTranspose2d,
    (3, True): nn.ConvTranspose3d,
}


class ConvNdWithConstraint(nn.Module):
    """
    通用卷积模块，支持 max_norm 权重范数约束。
    内部自动包装 PyTorch 的 Conv1d/2d/3d 或 Transposed 卷积。
    """

    def __init__(
        self, /, conv_cls: NN_CONV_CLASS, max_norm: int | None = None, **kwargs
    ):
        super().__init__()
        self._conv = conv_cls(**kwargs)
        self._max_norm = max_norm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._max_norm is not None:
            self._conv.weight.data = torch.renorm(
                self._conv.weight.data, p=2, dim=0, maxnorm=self._max_norm
            )
        return self._conv(input)

    def __getattr__(self, name):
        # 委托属性到 self._conv，但排除特殊字段防止递归
        if name != "_conv" and name in self._conv.__dict__:
            return getattr(self._conv, name)
        return super().__getattr__(name)


def convNd_with_constraint(
    nd: int, transpose: bool = False, max_norm: int | None = None, **kwargs
) -> ConvNdWithConstraint:
    """
    工厂函数，自动返回带 max_norm 限制的卷积模块。
    Args:
        nd (int): 1, 2, or 3 表示卷积维度
        transpose (bool): 是否为转置卷积
        max_norm (int | None): 权重最大范数
    Returns:
        ConvNdWithConstraint: 包装后的通用卷积模块
    """

    assert nd in (1, 2, 3), f"nd must be int 1, 2, or 3, but got {nd}"
    assert (
        nd,
        transpose,
    ) in ND_TRANSPOSE_TO_CONV_CLASS.keys(), (
        f"Unsupported Conv{'Transpose' if transpose else ''}{int(nd)}d combination."
    )

    return ConvNdWithConstraint(
        conv_cls=ND_TRANSPOSE_TO_CONV_CLASS[(nd, transpose)],
        max_norm=max_norm,
        **kwargs,
    )
