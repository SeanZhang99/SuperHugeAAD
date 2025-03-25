from .eeg_dataset import EegDataset


def create_datasets(*, cls: type[EegDataset], **kwargs):
    return cls.create_datasets(**kwargs)


def get_item(obj: EegDataset, idx: int):
    return obj.__getitem__(idx)
