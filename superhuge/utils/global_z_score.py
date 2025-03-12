import einops
import torch
import torch.nn as nn


class GlobalZScore(nn.Module):
    """
    计算整个训练集的全局均值和标准差 (跨所有时间和通道)
    在训练时更新统计量，在测试时冻结。
    """

    def __init__(self, momentum=0.1):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("running_mean", torch.tensor(0.0))  # 标量均值
        self.register_buffer("running_var", torch.tensor(1.0))  # 标量方差

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, C) -> (批次, 时间步, 通道数)
        """
        x_flat = einops.rearrange(x, "b t c -> b (t c)")

        if self.training:
            batch_mean = x_flat.mean(dim=(0, 1))  # 计算整个 batch 的均值
            batch_var = x_flat.var(dim=(0, 1), unbiased=False)  # 计算方差

            # 使用指数加权平均更新全局统计量
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * batch_var

        # 标准化
        x = (x - self.running_mean) / (torch.sqrt(self.running_var) + 1e-6)
        return x
