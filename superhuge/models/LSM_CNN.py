from einops import repeat
import torch
import einops.layers.torch as layers
from typing import Tuple
from torch import nn

def create_lsm_model(
        input: torch.Tensor,
        config: dict,
) -> Tuple[torch.nn.Module, torch.Tensor]:
    cnn_kernel = (1, config['dataset']['num_chan'],)
    lsm_chan_dim = config['model']['lsm']['chan_dim']

    model = torch.nn.Sequential()

    res = layers.Rearrange("batch channel time -> batch 1 time channel")
    input = res.forward(input)
    model.append(res)

    pad = torch.nn.ZeroPad2d((cnn_kernel[0]-1,0,0,0,))
    input = pad.forward(input)
    model.append(pad)

    conv = torch.nn.Conv2d(in_channels=input.shape[1],out_channels=lsm_chan_dim*lsm_chan_dim,kernel_size=cnn_kernel,bias=False)
    input = conv.forward(input)
    model.append(conv)

    bn = torch.nn.BatchNorm2d(input.shape[1])
    input = bn.forward(input)
    model.append(bn)

    res = layers.Rearrange("batch (c1 c2) time feature -> batch feature time c1 c2",c1=lsm_chan_dim)
    input = res.forward(input)
    model.append(res)

    return model, input

def create_cnn_model(
        input: torch.Tensor,
        config: dict,
        on_aux: bool = False,
) -> Tuple[torch.nn.Module,torch.Tensor]:
    num_layers = config['model']['cnn']['num_layers']
    cnn_kernel_size = (
        config['model']['cnn']['temporal_kernel'],
        config['model']['cnn']['chan_kernel'],
        config['model']['cnn']['chan_kernel'],
        )
    num_kernels = config['model']['cnn']['num_kernels']
    if isinstance(num_kernels,int):
        num_kernels = [num_kernels,] * num_layers

    model = torch.nn.Sequential()
    for ll in range(num_layers):
        cnn = torch.nn.Sequential(
            torch.nn.Conv3d(
            in_channels=input.shape[1] + (1 if on_aux and ll == 0 else 0),
            out_channels=num_kernels[ll],
            kernel_size=cnn_kernel_size,
            padding="same",
            ),
            torch.nn.ELU(),
            torch.nn.BatchNorm3d(num_kernels[ll]) if config['model']['use_norm'] else torch.nn.Identity(),
            torch.nn.Dropout(config['model']['dropout']) if not config['model']['dropout'] else torch.nn.Identity(),
        )
        input = cnn.forward(torch.cat([input,torch.randn(input.shape)],dim=1) if on_aux and ll == 0 else input)
        model.append(cnn)

    return model, input

def create_fc_model(
        input: torch.Tensor,
        config: dict,
) -> Tuple[torch.nn.Module,torch.Tensor]:
    hidden_dim = config['model']['post_mlp']['post_mlp_dim']
    out_dim = config['dataset']['nclass']

    model = torch.nn.Sequential()

    pooling = torch.nn.AvgPool3d(
        kernel_size=(input.shape[2],1,1)
    )
    input = pooling.forward(input)
    model.append(pooling)

    flatten = torch.nn.Flatten()
    input = flatten.forward(input)
    model.append(flatten)

    fc1 = torch.nn.Linear(
        in_features=input.shape[-1],
        out_features=hidden_dim,
        )
    input = fc1.forward(input)
    model.append(fc1)

    sg = torch.nn.Sigmoid()
    input = sg.forward(input)
    model.append(sg)

    fc2 = torch.nn.Linear(
        in_features=hidden_dim,
        out_features=out_dim,
    )
    input = fc2.forward(input)
    model.append(fc2)

    return model, input


def create_aux_encoder(
        config: dict
) -> torch.nn.Module:
    return nn.Sequential(
            nn.Linear(in_features=config['dataset']['n_ssl_pts'],out_features=config['model']['aux']['aux_hidden_dim'],bias=True),
            # nn.Linear(in_features=config['dataset']['nclass']//2,out_features=config['model']['aux']['aux_hidden_dim'],bias=True),
            nn.ELU(),
            nn.Linear(in_features=config['model']['aux']['aux_hidden_dim'],out_features=config['model']['lsm']['chan_dim']**2,bias=True),
            nn.Softmax(dim=-1),
            layers.Rearrange("batch (c1 c2) -> batch c1 c2",c1=config['model']['lsm']['chan_dim'],c2=config['model']['lsm']['chan_dim'])
        )

class lsm_cnn(torch.nn.Module):

    def __init__(self,
                config: dict,
                on_aux:bool = False,) -> None:
        super().__init__()

        input=torch.randn(
            size=(
                config['dataset']['batch_size'],
                config['dataset']['num_chan'],
                int(config['dataset']['fs']*config['dataset']['window_length']),
                )
        )

        self.lsm, input = create_lsm_model(
            input,
            config=config,
        )

        self.cnn, input = create_cnn_model(
            input,
            config,
            on_aux,
        )

        self.fc, input = create_fc_model(
            input,
            config,
        )

        self.aux_encoder = create_aux_encoder(
            config
        ) if on_aux else None



    def forward(self, *inputs):
        if self.aux_encoder is not None:
            eeg, aux = inputs
        else:
            eeg = inputs[0]
        eeg = self.lsm(eeg)
        if self.aux_encoder is not None:
            aux = self.aux_encoder(aux.to(torch.float))
            aux = repeat(aux,"batch c1 c2 -> batch feature time c1 c2",feature=1,time=eeg.shape[2])
            eeg = torch.cat([aux,eeg,],dim=1)
        eeg = self.cnn(eeg)
        eeg = self.fc(eeg)
        return eeg