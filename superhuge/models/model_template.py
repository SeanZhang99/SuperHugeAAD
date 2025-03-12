from torch.nn import Module


class ModelTemplate(Module):
    @staticmethod
    def create_models(**kwargs):
        return ModelTemplate(**kwargs)
