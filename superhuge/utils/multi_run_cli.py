from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from itertools import product
from math import isnan
import os
import pickle
from typing import Iterator, NamedTuple
from pathlib import Path

from lightning.pytorch.loggers.logger import Logger
import numpy as np

from lightning import LightningModule
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback

from .task_config_parser import TaskConfigParser

cv_fold = NamedTuple("cv_fold", [("val_fold_idx", str), ("test_fold_idx", str)])


@dataclass
class ExperimentStates:
    _task_configs: Iterator[tuple[list[str], str]]
    _cli_argv: list[str]
    _current_task_config: list[str] | None = None
    _current_experiment_name: str | None = None
    _cv_folds_iter: Iterator[tuple[str, str]] | None = None
    _current_val_fold_idx: str | None = None
    _current_test_fold_idx: str | None = None

    def __iter__(self):
        if isinstance(self._task_configs, Sequence):
            self._task_configs_iter = iter(list(self._task_configs))
        else:
            self._task_configs_iter = iter(self._task_configs)
        return self

    def __next__(self) -> tuple[list[str], str]:
        if self._cv_folds_iter is None:
            self._current_task_config, self._current_experiment_name = next(
                self._task_configs_iter
            )
            self._cv_folds_iter = iter(
                [
                    cv_fold(*fold)
                    for fold in self.__prepare_fold_idx(self._current_task_config)
                ]
            )
        try:
            self._current_val_fold_idx, self._current_test_fold_idx = next(
                self._cv_folds_iter
            )
            if self._current_val_fold_idx == self._current_test_fold_idx:
                return self.__next__()
        except StopIteration:
            self._cv_folds_iter = None
            return self.__next__()

        assert self._current_task_config is not None
        assert self._current_experiment_name is not None
        return (
            self._current_task_config
            + [
                "--data.init_args.val_fold_idx",
                self._current_val_fold_idx,
                "--data.init_args.test_fold_idx",
                self._current_test_fold_idx,
            ],
            self._current_experiment_name,
        )

    def __prepare_fold_idx(self, config_list: list[str]):
        assert "--data.init_args.n_folds" in config_list, (
            "MULTI_RUN_CLI:__PREPARE_FOLD_IDX:ARGUMENT_MISSING: "
            "Argument --data.init_args.n_folds is required to prepare fold indices"
        )
        n_folds = int(config_list[config_list.index("--data.init_args.n_folds") + 1])
        if "--data.init_args.val_fold_idx" not in self._cli_argv:
            val_fold_idx = [str(x) for x in range(n_folds)]
        else:
            val_fold_idx = self._cli_argv[
                self._cli_argv.index("--data.init_args.val_fold_idx") + 1
            ]

        if "--data.init_args.test_fold_idx" not in self._cli_argv:
            test_fold_idx = [str(x) for x in range(n_folds)]
        else:
            test_fold_idx = self._cli_argv[
                self._cli_argv.index("--data.init_args.test_fold_idx") + 1
            ]

        return list(product(val_fold_idx, test_fold_idx))


class MultiRunCLI:
    _experiment_states: ExperimentStates

    def __init__(
        self,
        *args: str,
        task_config_path: str,
        model_checkpoint_path: str | None = None,
        cli_checkpoint_path: str | None = None,
    ) -> None:
        self.cli_argv = list(args)
        self.task_config_path = task_config_path
        self.model_ckpt_path = model_checkpoint_path
        self.cli_ckpt_path = cli_checkpoint_path

        assert (
            self.task_config_path is not None
        ), "MULTI_RUN_CLI:__INIT__:TASK_CONFIG_ACQUIRING:ARGUMENT_MISSING: Task config is required by providing --task_config=<path> or --task_config <path>"
        self.task_config_parser = TaskConfigParser(self.task_config_path)

        if self.cli_ckpt_path is not None and os.path.isfile(self.cli_ckpt_path):
            with open(self.cli_ckpt_path, "rb") as f:
                loaded: MultiRunCLI = pickle.load(f)
            self._experiment_states = loaded._experiment_states
        else:
            self._experiment_states = ExperimentStates(
                _task_configs=iter(list(self.task_config_parser.generate_configs())),
                _cli_argv=self.cli_argv,
            )

    def __generate_config_hash(self, config_list: list[str]) -> int:
        """生成配置列表的确定性哈希种子"""
        # remove val_fold_idx and test_fold_idx from config_list
        config_list = [
            config
            for i, config in enumerate(config_list)
            if config_list[i - 1]
            not in ("--data.init_args.val_fold_idx", "--data.init_args.test_fold_idx")
            and config
            not in ("--data.init_args.val_fold_idx", "--data.init_args.test_fold_idx")
        ]
        # 创建稳定字符串表示
        config_str = "|".join(sorted(config_list)).encode("utf-8")
        # 生成SHA256哈希
        hash_digest = hashlib.sha256(config_str).digest()
        # 转换为0-2^32-1范围内的整数
        return int.from_bytes(hash_digest[:4], byteorder="big") % (2**32)

    def __extract_model_name(self):
        for i, arg in enumerate(self.cli_argv):
            if arg == "--model":
                if i + 1 < len(self.cli_argv):
                    config_path = self.cli_argv[i + 1]
                    model_name = os.path.splitext(os.path.basename(config_path))[0]
                    if model_name != "optimizer_config":
                        with open(config_path, "r") as f:

                            model_config = f.read()
                            model_hash = hashlib.sha256(
                                model_config.encode("utf-8")
                            ).hexdigest()[:4]

                        return (
                            model_name,
                            f"{model_hash}",
                        )
        return None, None

    def run(
        self,
        verbose: bool = True,
        save_config: bool = True,
        extra_experiment_name: str = "",
    ):

        accumulated_results: dict[str, list] = {}
        for (
            config_list,
            experiment_name,
        ) in self._experiment_states:
            model_name, model_hash = self.__extract_model_name()
            args = (
                self.cli_argv
                + config_list
                + [
                    "--experiment_name",
                    f"{experiment_name}-{extra_experiment_name}",
                    "--model_name",
                    f"{model_name}_v_{model_hash}",
                ]
            )
            seed = str(self.__generate_config_hash(args))
            cli = NamedParamsCLI(
                parser_kwargs={"parser_mode": "omegaconf"},
                args=(
                    args
                    + (
                        [
                            "--seed_everything",
                            seed,
                        ]
                        if "--seed_everything" not in self.cli_argv
                        else []
                    )
                    + [
                        "--experiment_hash",
                        seed,
                    ]
                ),
                run=False,
                save_config_callback=(SaveConfigCallback if save_config else None),
            )

            cli.trainer.fit(
                model=cli.model,
                datamodule=cli.datamodule,
                ckpt_path=self.model_ckpt_path,
            )

            for func, loader in zip(
                (cli.trainer.validate, cli.trainer.test),
                (
                    cli.datamodule.val_dataloader(),
                    cli.datamodule.test_dataloader(),
                ),
            ):
                results = func(
                    model=cli.model,
                    dataloaders=loader,
                    verbose=verbose,
                    ckpt_path=(
                        "best" if not hasattr(cli.model, "fake_parameter") else None
                    ),
                )
                for key, value in results[0].items():
                    assert not isnan(value), f"Evaluation metrics got NaN for {key}"
                    accumulated_results.setdefault(key, []).append(value)

            if (
                isinstance(cli.trainer.loggers, Sequence)
                and len(cli.trainer.loggers) > 0
            ):
                logger: Logger = cli.trainer.loggers[0]
                assert hasattr(logger, "_root_dir")
                assert hasattr(logger, "_name")
                assert hasattr(logger, "_version")
                cli_ckpt_path: Path = (
                    Path(logger._root_dir)  # type: ignore
                    / str(logger._name)  # type: ignore
                    / f"version_{logger._version}"  # type: ignore
                    / "cli_ckpt.pkl"
                )
                with cli_ckpt_path.open("wb") as f:
                    pickle.dump(self, f)

        return {k: np.mean(v) for k, v in accumulated_results.items()}


class NamedParamsCLI(LightningCLI):
    model: LightningModule

    def _get_parameters(self):  # type: ignore
        return self.model.named_parameters()

    def add_arguments_to_parser(self, parser):
        # When linking arguments, make sure the target argument is declared by the target class.
        parser.link_arguments(
            "data.fs",
            "model.init_args.model_common_args.fs",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.window_length",
            "model.init_args.model_common_args.window_length",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "model.init_args.num_mix_out_channels",
            "model.init_args.model_common_args.num_channels",
        )
        parser.link_arguments(
            "data.num_channels",
            "model.init_args.model_common_args.num_channels",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.init_args.root_path", "data.init_args.transform.root_path"
        )
        parser.link_arguments(
            "data.init_args.transform.init_args.fs",
            "data.init_args.fs",
        )
        parser.link_arguments(
            "data.init_args.preproc_stage",
            "data.init_args.transform.init_args.preproc_stage",
        )
        parser.link_arguments(
            "model.init_args.num_audio_features",
            "model.init_args.model_common_args.num_audio_features",
        )
        parser.link_arguments(
            "model.init_args.num_class",
            "model.init_args.model_common_args.num_class",
        )
        parser.link_arguments(
            "data.sample_weights",
            "model.init_args.multiclass_loss_weights",
            apply_on="instantiate",
        )

        def compute_experiment_path(*args: str | int) -> str:
            paths = []
            for arg in args:
                if isinstance(arg, str):
                    if "-" in arg:
                        paths.extend(arg.split("-"))
                    else:
                        paths.append(arg)
                elif isinstance(arg, int):
                    paths.append(str(arg))
                else:
                    raise ValueError(
                        "Only str and int types are supported for experiment path computation."
                    )
            path = os.path.join(*paths)
            return path

        parser.add_argument("--experiment_name", type=str)
        parser.add_argument("--model_name", type=str)
        parser.add_argument("--experiment_hash", type=str)

        parser.link_arguments(
            source=(
                "model_name",
                "experiment_name",
                "experiment_hash",
                "data.init_args.window_length",
            ),
            target="trainer.logger.init_args.name",
            compute_fn=compute_experiment_path,
        )
