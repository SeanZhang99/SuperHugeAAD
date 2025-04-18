from .final_mixin import lsBaseFinalMixin


class lsBaseModel(lsBaseFinalMixin):
    """
    This class is the base class for all linear models. It inherits from the following classes:
    lsBaseABCModel: This class is an abstract class that defines the structure of the linear model.
    lsBaseFinalMixin: This class defines the final methods of the linear model.
    pl2.LightningModule: This class is a PyTorch Lightning module that defines the training and evaluation methods of the linear model.

    For users who would like to write his/her own linear model, he/she should inherit from this class.
    You shall overwrite all abstract methods stated in lsBaseABCModel.
    You are free to overwrite other methods or add new ones if needed.
    """
