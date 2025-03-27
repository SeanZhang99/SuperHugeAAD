from torch import nn


class MultiHeadAttention(nn.MultiheadAttention):

    def forward(self, x):
        return super().forward(x, x, x)[0]
