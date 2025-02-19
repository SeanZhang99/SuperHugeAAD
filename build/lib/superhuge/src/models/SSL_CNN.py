import torch
from model.simpleCNN import simpleCNN_NJU
from torch import nn
from einops import rearrange, repeat


class SSL_CNN(simpleCNN_NJU):
    def __init__(self,config):
        super().__init__(config)
        self.aux_encoder = nn.Sequential(
            nn.Linear(in_features=config['dataset']['n_ssl_pts'],out_features=config['model']['aux']['aux_hidden_dim'],bias=True),
            # nn.Linear(in_features=config['dataset']['nclass']//2,out_features=config['model']['aux']['aux_hidden_dim'],bias=True),
            nn.ELU(),
            nn.Linear(in_features=config['model']['aux']['aux_hidden_dim'],out_features=config['model']['cnn']['num_kernels'],bias=True),
            nn.Softmax(dim=-1),
        )

        self.cnn = torch.nn.Conv2d(2,config['model']['cnn']['num_kernels'],(config['model']['cnn']['temporal_kernel'],config['dataset']['num_chan']))


    def forward(self, eeg, aux):
        eeg = torch.reshape(eeg,[eeg.shape[0],1,*eeg.shape[1:]])
        eeg = torch.permute(eeg,(0,1,3,2,))
        eeg = self.padding(eeg)
        aux = self.aux_encoder(aux.float())
        aux = repeat(aux, 'b f -> b unused t f',t=eeg.shape[-2],unused=1)
        eeg = torch.cat([eeg,aux],dim=1)
        eeg = self.cnn(eeg)
        eeg = self.relu(eeg)
        eeg = self.avgp(eeg)
        eeg = torch.squeeze(eeg)
        eeg = self.fc1(eeg)
        eeg = self.sigmoid(eeg)
        eeg = self.fc2(eeg)
        return eeg