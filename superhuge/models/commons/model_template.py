from torch.nn import Module


class ModelTemplate(Module):
    def __init__(self, fs: int, window_length: int):
        super().__init__()
        self.fs = fs
        self.window_length = window_length
