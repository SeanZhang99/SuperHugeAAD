# This is the script of EEG-Deformer
# This is the network script
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from typing import Iterable, Union


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, config, keyword = "chan"):
        super().__init__()
        heads=config['model']['trans'][keyword+'_num_heads']
        dim_head=config['model']['trans'][keyword+'_dim_heads']
        dropout=config['model']['dropout']
        symmetric_qk=config['model']['symmetric_qk']
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3 if not symmetric_qk else inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.symmetric_qk = symmetric_qk

    def forward(self, x):
        if not self.symmetric_qk:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        else:
            qv = self.to_qkv(x).chunk(2,dim=-1)
            q,  v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qv)
            dots = torch.matmul(q, q.transpose(-1, -2)) * self.scale

        attn = torch.softmax(dots,dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    

class Transformer(nn.Module):
    def cnn_block(self, in_chan, kernel_size, dp, use_max_pool = True):
        return nn.Sequential(
            nn.Dropout(p=dp),
            nn.Conv1d(in_channels=in_chan, out_channels=in_chan,
                      kernel_size=kernel_size, padding=self.get_padding_1D(kernel=kernel_size)),
            nn.BatchNorm1d(in_chan),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2) if use_max_pool else nn.Identity(),
        )

    def __init__(self, dim, config,):
        super().__init__()
        depth = config['model']['trans']['depth']
        skip_pool = config['model']['skip_pool']
        self.chan_attn_layers: Iterable = nn.ModuleList([])
        self.time_attn_layers: Iterable = nn.ModuleList([])
        time_dim = dim
        for i in range(depth):
            time_dim = time_dim if ((i%2 == 0) and skip_pool) else int(time_dim*0.5)
            self.chan_attn_layers.append(nn.ModuleList([
                Attention(
                    dim=time_dim,
                    config=config,
                    keyword="chan",
                ),
                FeedForward(
                    dim=time_dim, 
                    hidden_dim=config['model']['trans']['mlp_heads'],
                    dropout=config['model']['dropout']),
                self.cnn_block(in_chan=config['model']['cnn']['num_kernels'], kernel_size=config['model']['cnn']['temporal_kernel'], dp=config['model']['dropout'],use_max_pool=False if ((i % 2 == 0) and (skip_pool)) else True),
                nn.LayerNorm([time_dim,]),
                nn.LayerNorm([time_dim,]),
            ]))
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.skip_pool = skip_pool

    def forward(self, x):
        dense_feature = []
        for i, (attn, ff, cnn, ln1, ln2) in enumerate(self.chan_attn_layers):
            x_cg = x if ((i % 2 == 0) and (self.skip_pool)) else self.pool(x)
            x_cg = ln1(attn(x_cg) + x_cg)
            x_fg = cnn(x)
            x_info = self.get_info(x_fg)  # (b, in_chan)
            dense_feature.append(x_info)
            x = ln2(ff(x_cg) + x_fg)
        x_dense = torch.cat(dense_feature, dim=-1)
        x = torch.reshape(x,(x.size(0), -1))
        emd = torch.cat((x, x_dense), dim=-1)
        return emd

    def get_info(self, x):
        # x: b, k, l
        x = torch.log(torch.mean(x.pow(2), dim=-1))
        return x

    def get_padding_1D(self, kernel):
        return int(0.5 * (kernel - 1))


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class Deformer(nn.Module):
    def cnn_block(self, in_chan, out_chan, kernel_size, num_chan, last_layer, ):
        return nn.Sequential(
            Conv2dWithConstraint(in_chan, out_chan, kernel_size, padding=self.get_padding(kernel_size[-1]), max_norm=2),
            Conv2dWithConstraint(out_chan, out_chan, (num_chan, 1), padding=0, max_norm=2,),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)) if last_layer else Rearrange('b f c t -> b c f t'),
        )

    def __init__(self, *, config,):
        super().__init__()

        num_kernel = config['model']['cnn']['num_kernels']
        temporal_kernel = config['model']['cnn']['temporal_kernel']

        num_time = config['dataset']['window_length'] * config['dataset']['fs']
        num_chan = config['dataset']['num_chan']

        depth = config['model']['trans']['depth']


        self.skip_pool = config['model']['skip_pool']
        self.cnn_encoder1 = self.cnn_block(in_chan=1, out_chan=num_kernel, kernel_size=(1, temporal_kernel), num_chan=num_chan,last_layer=False)

        self.aux_encoder = nn.Sequential(
            nn.Linear(in_features=config['dataset']['n_ssl_pts'],out_features=config['model']['aux']['aux_hidden_dim'],bias=True),
            # nn.Linear(in_features=config['dataset']['nclass']//2,out_features=config['model']['aux']['aux_hidden_dim'],bias=True),
            nn.ELU(),
            nn.Linear(in_features=config['model']['aux']['aux_hidden_dim'],out_features=num_kernel,bias=True),
            nn.Softmax(dim=-1),
        )

        self.aux_norm = nn.BatchNorm2d(num_features=2)

        self.cnn_encoder2 = self.cnn_block(in_chan=2, out_chan=num_kernel, kernel_size=(1, temporal_kernel), num_chan=num_kernel,last_layer=True)

        dim = int(0.5*num_time)  # embedding size after the first cnn encoder

        self.to_patch_embedding = Rearrange('b k c f -> b k (c f)')

        self.pos_embedding = nn.Parameter(torch.randn(1, num_kernel, dim))

        self.transformer = Transformer(
            dim=dim,
            config=config,
        )

        L = self.get_hidden_size(input_size=dim, num_layer=depth)
        # L[-1] = input_size * 0.5 ** depth
        # or: (skip pooling)
        # L[-1] = input_size * 0.5 ** (depth // 2)

        # out_size: part one from transformer, part two from skip convolution (HCT)
        # double num_kernel -> double out_size
        # double depth -> undetermined. Affected by depth and input_size. 
        out_size = int(num_kernel * L[-1]) + int(num_kernel * depth)

        self.mlp_head = nn.Sequential(
            nn.Linear(out_size, config['model']['post_mlp']['post_mlp_dim']),
            nn.ELU(),
            nn.Linear(config['model']['post_mlp']['post_mlp_dim'],len(config['dataset']['classlabels'])),
        )


    def forward(self, eeg, aux):
        # eeg: (b, chan, time)
        # aux: (b, theta, 1)
        eeg = torch.unsqueeze(eeg, dim=1)  # (b, 1, chan, time)
        x = self.cnn_encoder1(eeg)  # (b, num_kernel, 1, num_time)

        aux = self.aux_encoder(aux.float())
        aux = repeat(aux, 'b f -> b unused f t',t=x.shape[-1],unused=1)
        x = torch.cat([x,aux],dim=1)
        x = self.aux_norm(x)

        x = self.cnn_encoder2(x)
        x = self.to_patch_embedding(x)

        b, n, _ = x.shape
        x += self.pos_embedding
        x = self.transformer(x)
        return self.mlp_head(x)

    def get_padding(self, kernel):
        return (0, int(0.5 * (kernel - 1)))

    def get_hidden_size(self, input_size, num_layer):
        return [int(input_size * (0.5 ** (i if not self.skip_pool else i // 2))) for i in range(num_layer + 1)]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
