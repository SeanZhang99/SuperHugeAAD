import pathlib
from typing import Literal
import pandas as pd


def gather_runs(
    run_dir: str | pathlib.Path,
    concat: bool = True,
    reduce: Literal["mean", "median"] | None = None,
    metrics_names: str | list[str] = ["test", "val"],
):
    run_dir = pathlib.Path(run_dir)
    runs = []
    for file in run_dir.glob("**/metrics.csv"):
        df = pd.read_csv(file)
        # Keep only the columns that starts with the metrics names
        if isinstance(metrics_names, str):
            metrics_names = [metrics_names]
        metrics_cols = [
            col
            for col in df.columns
            if any(col.startswith(name) for name in metrics_names)
        ]
        df = df[metrics_cols]
        # take the last two row of the dataframe and merge into one row
        target_df = pd.DataFrame()
        for col in df.columns:
            tmp = df[col]
            idx = tmp.last_valid_index()
            target_df[col] = [tmp.loc[idx]]
        runs.append(target_df)

    if concat:
        runs = pd.concat(runs, ignore_index=True)
        if reduce is not None:
            if reduce == "mean":
                runs = runs.mean()
                runs = runs.to_frame().T
            elif reduce == "median":
                runs = runs.median()
                runs = runs.to_frame().T
    return runs
