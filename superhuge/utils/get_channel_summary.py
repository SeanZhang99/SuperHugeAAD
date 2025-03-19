import pickle
from typing import Any


from superhuge.datasets.metadata_processing.data import MetaData


def get_channel_summary(metadata_path: str):
    with open(metadata_path, "rb") as f:
        metadata: dict[str, Any] = pickle.load(f)
    channel_summary = {}
    for entry, meta in metadata.items():
        for k, v in meta["channel_infos"].items():
            chan_name = v["name"]
            if (
                all(pattern not in chan_name for pattern in ["EX", "EO", "EC"])
                and chan_name not in channel_summary.keys()
            ):
                channel_summary[chan_name] = len(channel_summary)
    return dict(sorted(channel_summary.items(), key=lambda x: x[0]))


if __name__ == "__main__":
    channel_summary = get_channel_summary(r"E:\derivatives\SuperHuge\meta\metadata.pkl")
    print("CHANNEL1D_ENUM = {")
    for k, v in channel_summary.items():
        print(f'    "{k}": {v},')
    print("}")
    print("\n")
    print(f"num_electrodes: {len(channel_summary)}")
    pass
