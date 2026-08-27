import inspect
import random
from collections.abc import Callable, Mapping
from typing import Any, Protocol, Sequence, TypeAlias, TypeVar

from pydantic import BaseModel
from pydantic_core import core_schema


class MetadataElement(BaseModel, extra="allow"):
    dataset_id: int | None = 0
    subject_id: int | None = 0
    trial_id: int | None = 0
    num_channel: int | None = 0
    signal_length: int | None = 0
    fs: int | None = 0
    dataset_name: str | None = None
    channel_infos: Mapping[int, Mapping[str, Any]] | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.entry = f"dataset-{self.dataset_id:03d}-subject-{self.subject_id:03d}-trial-{self.trial_id:03d}"
        if self.channel_infos is None and self.num_channel is not None:
            self.channel_infos = {
                i: {"name": f"ch-{i+1:02d}", "type": "EEG"}
                for i in range(self.num_channel)
            }


class RegressionMetadataElement(MetadataElement):
    env: str | int | None = None
    mel: str | int | None = None


class ClassifyMetadataElement(MetadataElement):
    label: str | int | None


class RegressionClassifyMetadataElement(
    RegressionMetadataElement, ClassifyMetadataElement
):
    pass


MetadataElementType = TypeVar("MetadataElementType", bound=MetadataElement)

DatasetSubjectTrialEntry: TypeAlias = str
MetadataField: TypeAlias = str
FoldIndicator: TypeAlias = str
MetadataValue: TypeAlias = Any
# Metadata: TypeAlias = dict[DatasetSubjectTrialEntry, MetadataElement]
Metadata = Mapping[DatasetSubjectTrialEntry, MetadataElementType]
CrossValidationEntry: TypeAlias = dict[
    FoldIndicator,
    Sequence[DatasetSubjectTrialEntry] | Sequence[float | int | Sequence[float | int]],
]


def generate_test_metadata(
    num_datasets=10,
    num_subjects_per_dataset=10,
    min_trials_per_subject=10,
    max_trials_per_subject=20,
) -> Metadata:
    metadata = {}
    random.seed(42)
    for i in range(num_datasets):
        for j in range(num_subjects_per_dataset):
            num_trials = random.randint(min_trials_per_subject, max_trials_per_subject)
            for k in range(num_trials):
                metadata[f"dataset-{i+1:03d}-subject-{j+1:03d}-trial-{k+1:03d}"] = (
                    MetadataElement(
                        dataset_id=i + 1,
                        subject_id=j + 1,
                        trial_id=k + 1,
                        num_channel=32,
                        signal_length=10000,
                        fs=128,
                        label=random.randint(0, 1),  # Random binary label
                    )
                )

    return metadata
