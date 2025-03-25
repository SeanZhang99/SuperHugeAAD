from torch.utils.data._utils.collate import default_collate


def custom_collate_fn(batch):
    """
    Custom collate function to process input batch.
    Aggregates 'channel_infos' from 'meta' into a list.
    Uses default behavior for other fields and keys.
    """
    aggregated_channel_infos = []
    for sample in batch:
        if "meta" in sample and "channel_infos" in sample["meta"]:
            aggregated_channel_infos.extend(sample["meta"]["channel_infos"])
            del sample["meta"]["channel_infos"]

    # Use default_collate for the rest of the batch
    collated_batch = default_collate(batch)

    # Replace 'channel_infos' in 'meta' with the aggregated list
    if "meta" in collated_batch:
        collated_batch["meta"]["channel_infos"] = aggregated_channel_infos

    return collated_batch
