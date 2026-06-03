from os import PathLike
from pathlib import Path


def rm_ckpt(ckpt_dir: PathLike):
    ckpt_path = Path(ckpt_dir)
    if ckpt_path.exists() and ckpt_path.is_dir():
        for child in ckpt_path.iterdir():
            if child.is_file() and child.name.endswith(".ckpt"):
                child.unlink()
            elif child.is_dir():
                rm_ckpt(child)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove checkpoint files in a directory."
    )
    parser.add_argument(
        "ckpt_dir",
        type=str,
        help="The directory containing checkpoint files to remove.",
    )
    args = parser.parse_args()

    assert hasattr(args, "ckpt_dir"), "The 'ckpt_dir' argument is required."
    assert args.ckpt_dir, "The 'ckpt_dir' argument cannot be empty."
    assert Path(
        args.ckpt_dir
    ).exists(), f"The directory '{args.ckpt_dir}' does not exist."

    rm_ckpt(args.ckpt_dir)
