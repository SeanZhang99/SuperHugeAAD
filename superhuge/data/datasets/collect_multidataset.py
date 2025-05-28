from collections import defaultdict

from torch.utils.data._utils.collate import default_collate


def collect_multidataset(batch: list[dict]):
    """
    Collects the outputs of multiple datasets into a single batch.

    Args:
        batched (dict): A dictionary of batched data from multiple datasets.

    Returns:
        dict: A dictionary of the collected batched data.
    """
    """Groups data by source type and applies default collation per group."""

    grouped_batches = defaultdict(list)

    # Group by 'dataset_id'
    for item in batch:
        grouped_batches[item["meta"]["dataset_id"]].append(item)

    # Apply default collation separately per source
    collated_batch = {}
    for dataset_id, items in grouped_batches.items():
        collated_batch[dataset_id] = default_collate(items)

    return collated_batch
