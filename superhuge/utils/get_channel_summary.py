"""
EEG channel summary, 2D grid mapping, and naming convention normalization.

Reads a pickled metadata dict (produced by `get_dataset_info`) and:
  1. Extracts unique EEG channel names (filtering out EOG/EXG/ECG channels).
  2. Maps channels to a 2D grid layout (row = anterior-posterior band,
     column = left-right position).
  3. Handles naming convention variants such as A1/A2 ↔ M1/M2,
     FP1 → Fp1, and non-standard prefixes (CB, M, etc.).
  4. Detects duplicate grid positions and warns about ambiguous layout.
"""

from collections import OrderedDict
import os
import pickle
import warnings

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import re

# ---------------------------------------------------------------------------
# Regex for standard EEG 10-10 channel names
#
#   Group 1: prefix     — 1 or 2 letters from the known 10-10 set
#   Group 2: number     — digit(s) or 'z' (midline), optionally with a
#                          trailing digit for numbers like "10", "11", "12"
#
# Extended to cover non-standard prefixes occasionally used:
#   - M1/M2    (mastoid, modern naming for what was A1/A2)
#   - CB1/CB2  (cerebellar channels, used by USTC dataset)
#   - FP1/FPZ  (some datasets write "FP" in uppercase instead of "Fp")
#
# First-letter set:   A C F I M N O P T
# Second-letter set:  B C F O p P T     (lowercase 'p' for "Fp" prefix)
# ---------------------------------------------------------------------------
PREFIX_REGEX = re.compile(r"([ACFIMNOPT][BCFOpPT]?)([\dz]\d?)")

# Channels whose names match the regex but are NOT EEG channels
EXG_PATTERNS = ("EX", "EO", "EC")


# ---------------------------------------------------------------------------
# Naming-convention normalisation
# ---------------------------------------------------------------------------

# Mapping from old / non-standard prefixes to the canonical form used in
# the row/column tables below.
_PREFIX_ALIASES: Dict[str, str] = {
    "FP": "Fp",  # uppercase P → lowercase p
}


def normalize_channel_name(name: str, *, swap_am: bool = False) -> str:
    """Normalise a channel name to its canonical form.

    Parameters
    ----------
    name : str
        Raw channel name (e.g. ``"FP1"``, ``"M1"``, ``"A2"``).
    swap_am : bool
        If *True*, treat A1/A2 and M1/M2 as the same position.
        The modern standard (M1/M2) is preferred, so A1 → M1, A2 → M2.
        The 10-20 system originally called the mastoid reference A1/A2;
        the 10-10 system renamed it M1/M2.  Many datasets use the two
        interchangeably.

    Returns
    -------
    str
        Normalised channel name.
    """
    # Uppercase P → lowercase p (e.g. "FP1" → "Fp1")
    for old, new in _PREFIX_ALIASES.items():
        if name.startswith(old):
            name = new + name[len(old) :]
            break

    if swap_am:
        # A1/A2 → M1/M2 (mastoid reference, modern naming)
        if name in ("A1", "A2"):
            name = "M" + name[1:]
        # M1/M2 → A1/A2 is deliberately NOT supported — modern standard is M.

    return name


# ---------------------------------------------------------------------------
# Row mapping  (anterior → posterior)
#
# Each key is either a single prefix string or a tuple of prefixes that
# should share the same grid *row* because they lie at approximately the
# same anterior-posterior level on the scalp.
#
# Maps from-region → row-index (0 = most anterior).
# ---------------------------------------------------------------------------

_ROW_TEMPLATE: "OrderedDict[str | Tuple[str, ...], bool]" = OrderedDict(
    {
        "N": False,  # Nasion
        "Fp": False,  # Frontopolar
        "AF": False,  # Anterior frontal
        "F": False,  # Frontal
        ("FC", "FT"): False,  # Fronto-central, fronto-temporal
        ("C", "T", "A"): False,  # Central, temporal, earlobe reference
        ("CP", "TP", "M"): False,  # Centro-parietal, temporo-parietal, mastoid
        "P": False,  # Parietal
        "PO": False,  # Parieto-occipital
        "O": False,  # Occipital
        "CB": False,  # Cerebellar (non-standard)
        "I": False,  # Inion
    }
)


def create_channel_row_mapping(channels: List[str]) -> Dict[str, int]:
    """Build a prefix → row-index dictionary from the channels present.

    Only rows that appear in *channels* are assigned an index; unused rows
    are omitted so the grid does not have empty rows.

    Parameters
    ----------
    channels : list of str
        Normalised channel names (e.g. ``["Fp1", "Cz", "O2"]``).

    Returns
    -------
    dict[str, int]
        Mapping ``prefix → row_index``.
    """
    present: Dict[str | Tuple[str, ...], bool] = OrderedDict(
        (k, False) for k in _ROW_TEMPLATE
    )

    for ch in channels:
        match = PREFIX_REGEX.match(ch)
        if not match:
            continue
        region = match.group(1)

        for key in present:
            if isinstance(key, tuple):
                if region in key:
                    present[key] = True
                    break
            elif region == key:
                present[key] = True
                break

    # Compact: assign consecutive integers to present rows.
    new_mapping: Dict[str, int] = {}
    idx = 0
    for key, is_present in present.items():
        if is_present:
            if isinstance(key, tuple):
                for prefix in key:
                    new_mapping[prefix] = idx
            else:
                new_mapping[key] = idx
            idx += 1

    return new_mapping


# ---------------------------------------------------------------------------
# Column mapping  (left → right)
#
# Column keys are the lateral number or 'z' for midline.
# Ordered from leftmost (11) to rightmost (12).
# ---------------------------------------------------------------------------

_COL_TEMPLATE: "OrderedDict[str, bool]" = OrderedDict(
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


def create_channel_col_mapping(channels: List[str]) -> Dict[str, int]:
    """Build a lateral-number → column-index dictionary.

    Parameters
    ----------
    channels : list of str
        Normalised channel names.

    Returns
    -------
    dict[str, int]
        Mapping ``number_or_z → column_index``.
    """
    present: Dict[str, bool] = OrderedDict((k, False) for k in _COL_TEMPLATE)

    for ch in channels:
        match = PREFIX_REGEX.match(ch)
        if not match:
            continue
        region = match.group(1)
        number = match.group(2)

        # Non-standard prefixes that sit at the far edges of the scalp:
        #   A1/A2  (earlobe reference)
        #   M1/M2  (mastoid reference, modern naming)
        #
        # CB1/CB2 (cerebellar) are deliberately NOT included here — they
        # should be aligned below O1/O2, not at the far edges.
        if region in ("A", "M"):
            number = str(10 + int(number))

        if number in present:
            present[number] = True

    # Compact.
    new_mapping: Dict[str, int] = {}
    idx = 0
    for number, is_present in present.items():
        if is_present:
            new_mapping[number] = idx
            idx += 1

    return new_mapping


# ---------------------------------------------------------------------------
# Position inference
# ---------------------------------------------------------------------------


def infer_channel_positions(
    channels: List[str],
    *,
    swap_am: bool = False,
) -> Dict[str, Tuple[int, int]]:
    """Compute (row, column) grid positions for every channel.

    Parameters
    ----------
    channels : list of str
        Raw channel names from metadata (may contain naming variants).
    swap_am : bool
        If *True*, treat A1/A2 ↔ M1/M2 as the same position (see
        :func:`normalize_channel_name` for rationale).

    Returns
    -------
    dict[str, tuple[int, int]]
        ``channel_name → (row, column)``.
    """
    # Normalise names before deriving mappings.
    norm_channels = [normalize_channel_name(ch, swap_am=swap_am) for ch in channels]

    row_mapping = create_channel_row_mapping(norm_channels)
    col_mapping = create_channel_col_mapping(norm_channels)

    channel_to_position: Dict[str, Tuple[int, int]] = {}
    for ch in channels:
        # Use the normalised form for grid lookup.
        nch = normalize_channel_name(ch, swap_am=swap_am)

        match = PREFIX_REGEX.match(nch)
        if not match:
            warnings.warn(f"Unrecognised channel name '{ch}' — skipping grid mapping.")
            continue

        region = match.group(1)
        number = match.group(2)

        # Earlobe (A) and mastoid (M) references sit at the far edges
        # of the scalp → map to columns 11/12.  CB (cerebellar) keeps
        # its original number (1/2) so it sits below O1/O2.
        if region in ("A", "M"):
            number = str(10 + int(number))

        if region not in row_mapping:
            warnings.warn(f"Unknown region '{region}' for channel '{ch}' — skipping.")
            continue
        if number not in col_mapping:
            warnings.warn(
                f"Unknown column number '{number}' for channel '{ch}' — skipping."
            )
            continue

        row = row_mapping[region]
        col = col_mapping[number]

        # Store under the *original* name so the caller sees the original
        # spelling.
        channel_to_position[ch] = (row, col)

    return channel_to_position


# ---------------------------------------------------------------------------
# High-level public API
# ---------------------------------------------------------------------------


def get_channel_summary(
    metadata_path: str,
    *,
    swap_am: bool = False,
    verbose: bool = False,
) -> List[str]:
    """Extract unique EEG channel names from a pickled metadata dict.

    Parameters
    ----------
    metadata_path : str
        Path to ``metadata.pkl``.
    swap_am : bool
        Normalise A1/A2 ↔ M1/M2 (see :func:`normalize_channel_name`).
    verbose : bool
        Log number of channels found.

    Returns
    -------
    list of str
        Sorted unique channel names.
    """
    with open(metadata_path, "rb") as f:
        metadata: Dict[str, Any] = pickle.load(f)

    channel_names: set = set()
    for entry, meta in metadata.items():
        for _k, v in meta["channel_infos"].items():
            chan_name: str = v["name"]
            # Skip EOG / EXG / ECG auxiliary channels.
            if any(ex in chan_name for ex in EXG_PATTERNS):
                continue
            chan_name = normalize_channel_name(chan_name, swap_am=swap_am)
            channel_names.add(chan_name)

    result = sorted(channel_names)
    if verbose:
        print(f"[get_channel_summary] Found {len(result)} unique EEG channels.")
    return result


# ---------------------------------------------------------------------------
# Grid and vector output
# ---------------------------------------------------------------------------


def map_channels_to_grid(
    metadata_path: str,
    output_path: str = "",
    *,
    swap_am: bool = False,
):
    """Read metadata, infer 2D grid layout, and print / write the result.

    Parameters
    ----------
    metadata_path : str
        Path to ``metadata.pkl``.
    output_path : str, optional
        If provided, write the result to a Python enum file.
    swap_am : bool
        Treat A1/A2 ↔ M1/M2 as the same electrode position.
    """
    channels = get_channel_summary(metadata_path, swap_am=swap_am)
    positions = infer_channel_positions(channels, swap_am=swap_am)

    if not positions:
        print("No valid channel positions could be inferred. Aborting.")
        return

    max_row = max(p[0] for p in positions.values()) + 1
    max_col = max(p[1] for p in positions.values()) + 1

    grid = np.empty((max_row, max_col), dtype=object)
    grid.fill(None)

    for ch, (r, c) in positions.items():
        grid[r, c] = ch

    print("\n" + "=" * 60)
    print("  2D GRID MAPPING OF EEG CHANNELS")
    print("  (anterior → posterior ↓,  left → right →)")
    print("=" * 60 + "\n")

    for row in grid:
        print(" ".join([f"{ch:>5}" if ch else "  ---" for ch in row]))

    print("\n" + "-" * 60)
    print("CHANNEL2D_ENUM mapping:\n")
    for k, v in positions.items():
        print(f'    "{k}": {v},')
    print()
    print(f"NUM_ELECTRODES = {len(positions)}")

    # Detect duplicate grid positions.
    dup: Dict[Tuple[int, int], List[str]] = {}
    for ch, pos in positions.items():
        dup.setdefault(pos, []).append(ch)

    collisions = {p: chs for p, chs in dup.items() if len(chs) > 1}
    if collisions:
        print("\n" + "!" * 60)
        print("  DUPLICATE GRID POSITIONS (check & fix manually):")
        print("!" * 60)
        for pos, chs in sorted(collisions.items()):
            print(f"    Position {pos}: {', '.join(chs)}")
        print()
    else:
        print("\nNo duplicate grid positions detected.\n")

    if output_path:
        write_channel_results_to_file(channels, positions, output_path)


def map_channel_to_vector(
    metadata_path: str,
    output_path: str = "",
    *,
    swap_am: bool = False,
):
    """Extract channel names as a flat sorted list (1D enumeration).

    Parameters
    ----------
    metadata_path : str
        Path to ``metadata.pkl``.
    output_path : str, optional
        If provided, write the result to a Python enum file.
    swap_am : bool
        Treat A1/A2 ↔ M1/M2 as the same electrode position.
    """
    channel_vector = get_channel_summary(metadata_path, swap_am=swap_am)

    print("\n" + "=" * 60)
    print("  1D CHANNEL ENUMERATION")
    print("=" * 60 + "\n")

    print("CHANNEL1D_ENUM = {")
    for i, ch in enumerate(channel_vector):
        print(f'    "{ch}": {i},')
    print("}")
    print(f"\nnum_electrodes = {len(channel_vector)}\n")

    if output_path:
        write_channel_results_to_file(channel_vector, {}, output_path)


# ---------------------------------------------------------------------------
# File writer
# ---------------------------------------------------------------------------


def write_channel_results_to_file(
    channel_vector: List[str],
    grid_positions: Dict[str, Tuple[int, int]],
    output_path: str,
):
    """Write channel vector and grid positions as Python Enum definitions.

    Parameters
    ----------
    channel_vector : list of str
        Sorted channel names.
    grid_positions : dict
        Channel → (row, column) mapping (may be empty for 1D-only output).
    output_path : str
        Destination ``.py`` file.
    """
    with open(output_path, "w") as f:
        f.write("from enum import Enum\n\n")
        f.write("class CHANNEL1D_ENUM(Enum):\n")
        for i, ch in enumerate(channel_vector):
            # Replace non-identifier characters so the enum member is valid.
            member = ch.replace("-", "_").replace(".", "_")
            f.write(f"    {member} = {i}\n")
        f.write("\n")
        f.write(f"NUM_ELECTRODES = {len(channel_vector)}\n")

        if grid_positions:
            f.write("\n")
            f.write("class CHANNEL2D_ENUM(Enum):\n")
            for ch, pos in grid_positions.items():
                member = ch.replace("-", "_").replace(".", "_")
                f.write(f"    {member} = {pos}\n")
        f.write("\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    metadata_path = rf"E:\derivatives\SuperHuge\preprocessed\meta\metadata.pkl"
    output_path = os.path.join(os.path.dirname(__file__), "channel_enum.py")

    map_channel_to_vector(metadata_path, output_path, swap_am=True)
    map_channels_to_grid(metadata_path, output_path, swap_am=True)
