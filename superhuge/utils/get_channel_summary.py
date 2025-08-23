from collections import OrderedDict
import os
import pickle
import numpy as np
from typing import Any, Dict, List, Tuple
import re

pattern = re.compile(r"([ACFINOPT][CFOpPT]?)([\dz]\d?)")


def get_channel_summary(metadata_path: str) -> List[str]:
    """
    解析 EEG 通道数据并获取通道名称列表
    """
    with open(metadata_path, "rb") as f:
        metadata: Dict[str, Any] = pickle.load(f)

    channel_names = set()
    for entry, meta in metadata.items():
        for k, v in meta["channel_infos"].items():
            chan_name = v["name"]
            if all(pattern not in chan_name for pattern in ["EX", "EO", "EC"]):
                channel_names.add(chan_name)

    return sorted(channel_names)


def create_channel_row_mapping(channels: List[str]) -> Dict[str, int]:
    """
    生成通道字母标识到行号的映射表，合并FT, FC，T, C, TP, CP等情况
    """
    channel_name_to_row = OrderedDict(
        {
            "N": False,
            "Fp": False,
            "AF": False,
            "F": False,
            ("FC", "FT"): False,
            ("A", "C", "T"): False,
            ("CP", "TP"): False,
            "P": False,
            "PO": False,
            "O": False,
            "I": False,
        }
    )

    for ch in channels:
        match = pattern.match(ch)
        if match:
            region = match.group(1)
            for channel_name in channel_name_to_row.keys():
                if isinstance(channel_name, tuple):
                    if region in channel_name:
                        channel_name_to_row[channel_name] = True
                        break
                elif region == channel_name:
                    # 1 indicate reserve the row
                    channel_name_to_row[channel_name] = True
                    break

    new_channel_name_to_row = OrderedDict()

    i = 0
    for k, v in channel_name_to_row.items():
        if v:
            if isinstance(k, tuple):
                for kk in k:
                    new_channel_name_to_row[kk] = i
            else:
                new_channel_name_to_row[k] = i
            i += 1

    return new_channel_name_to_row


def create_channel_col_mapping(channels: List[str]) -> Dict[str, int]:
    channel_name_to_col = OrderedDict(
        {
            "11": False,
            "9": False,
            "7": False,
            "5": False,
            "3": False,
            "1": False,
            "z": False,
            "2": False,
            "4": False,
            "6": False,
            "8": False,
            "10": False,
            "12": False,
        }
    )

    for ch in channels:
        if ch.endswith("1"):
            pass
        match = pattern.match(ch)
        if match:
            region = match.group(1)
            number = match.group(2)
            if region == "A":
                number = str(10 + int(number))
            if number in channel_name_to_col.keys():
                channel_name_to_col[number] = True

    new_channel_name_to_col = OrderedDict()

    i = 0
    for k, v in channel_name_to_col.items():
        if v:
            if isinstance(k, tuple):
                for kk in k:
                    new_channel_name_to_col[kk] = i
            else:
                new_channel_name_to_col[k] = i
            i += 1

    return new_channel_name_to_col


def infer_channel_positions(channels: List[str]) -> Dict[str, Tuple[int, int]]:
    """
    计算通道在二维网格中的位置
    """
    row_mapping = create_channel_row_mapping(channels)
    col_mapping = create_channel_col_mapping(channels)
    channel_to_position = {}
    for channel in channels:
        match = pattern.match(channel)
        if match:
            region = match.group(1)
            number = match.group(2)
            if region == "A":
                number = str(10 + int(number))
            row = row_mapping[region]
            column = col_mapping[number]
            channel_to_position[f"{region}{number}"] = (row, column)

    return channel_to_position


def map_channels_to_grid(metadata_path: str, output_path: str = ""):
    """
    读取元数据并映射 EEG 通道到二维排列
    """
    channels = get_channel_summary(metadata_path)
    positions = infer_channel_positions(channels)

    max_row = max(p[0] for p in positions.values()) + 1
    max_col = max(p[1] for p in positions.values()) + 1

    grid = np.empty((max_row, max_col), dtype=object)
    grid.fill(None)

    for ch, (r, c) in positions.items():
        grid[r, c] = ch

    print("\nSummary of 2D grid mapping of EEG channels:\n")
    for row in grid:
        print(" ".join([f"{ch:>5}" if ch else "  ---" for ch in row]))

    print("\nPlease paste the following dict into your code:\n")
    print("CHANNEL2D_ENUM = {\n")
    for k, v in positions.items():
        print(f'    "{k}": {v},')
    print("}")
    print("\n")
    print("num_electrodes = ", len(positions))

    # find out potential duplicated grid positions
    duplicate_positions = {}
    for ch, pos in positions.items():
        if pos in duplicate_positions:
            duplicate_positions[pos].append(ch)
        else:
            duplicate_positions[pos] = [ch]
    print("\nPotential duplicated grid positions:")
    for pos, chs in duplicate_positions.items():
        if len(chs) > 1:
            print(f"Position {pos} has channels: {', '.join(chs)}")
    print("\n")
    print("Please check the above duplicated positions and fix them manually.\n")

    if output_path:
        write_channel_results_to_file(channels, positions, output_path)


def map_channel_to_vector(metadata_path: str, output_path: str = ""):
    channel_vector = get_channel_summary(metadata_path)

    print("\nPlease paste the following list into your code:\n")
    print("CHANNEL1D_ENUM = {\n")
    for i, ch in enumerate(channel_vector):
        print(f'    "{ch}": {i},')
    print("}")
    print("\n")
    print("num_electrodes = ", len(channel_vector))

    if output_path:
        write_channel_results_to_file(channel_vector, {}, output_path)


def write_channel_results_to_file(
    channel_vector: List[str],
    grid_positions: Dict[str, Tuple[int, int]],
    output_path: str,
):
    """
    Write the channel vector and grid results into a .py file as Enum types.
    """
    with open(output_path, "w") as f:
        f.write("from enum import Enum\n\n")
        f.write("class CHANNEL1D_ENUM(Enum):\n")
        for i, ch in enumerate(channel_vector):
            f.write(f"    {ch} = {i}\n")
        f.write("\n")
        f.write("class CHANNEL2D_ENUM(Enum):\n")
        for ch, pos in grid_positions.items():
            f.write(f"    {ch} = {pos}\n")
        f.write("\n")
        f.write(f"num_electrodes = {len(channel_vector)}\n")


if __name__ == "__main__":
    metadata_path = (
        "/data/nvme/ssd2/zhangyuanming/EEG/derivatives/SuperHuge/meta/metadata.pkl"
    )
    output_path = os.path.join(os.path.dirname(__file__), "channel_enum.py")
    map_channel_to_vector(metadata_path, output_path)
    map_channels_to_grid(metadata_path, output_path)
