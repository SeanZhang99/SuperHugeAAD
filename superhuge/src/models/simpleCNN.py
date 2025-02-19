import torch


class simpleCNN_NJU(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.padding = torch.nn.ZeroPad2d((0,0,0,config['model']['cnn']['temporal_kernel']-1))
        self.cnn = torch.nn.Conv2d(1,config['model']['cnn']['num_kernels'],(config['model']['cnn']['temporal_kernel'],config['dataset']['num_chan']))
        # time * 1
        self.relu = torch.nn.ELU()
        # time * 1
        self.avgp = torch.nn.AvgPool2d((int(config['dataset']['fs']*config['dataset']['window_length']),1))
        self.fc1 = torch.nn.Linear(config['model']['cnn']['num_kernels'],config['model']['post_mlp']['post_mlp_dim'])
        self.sigmoid = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(config['model']['post_mlp']['post_mlp_dim'],config['dataset']['nclass'])

    def forward(self,eeg):
        eeg = torch.reshape(eeg,[eeg.shape[0],1,*eeg.shape[1:]])
        eeg = torch.permute(eeg,(0,1,3,2,))
        eeg = self.padding(eeg)
        eeg = self.cnn(eeg)
        eeg = self.relu(eeg)
        eeg = self.avgp(eeg)
        eeg = torch.squeeze(eeg)
        eeg = self.fc1(eeg)
        eeg = self.sigmoid(eeg)
        eeg = self.fc2(eeg)
        return eeg