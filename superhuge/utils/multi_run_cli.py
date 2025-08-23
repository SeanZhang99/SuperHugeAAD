import gc
import hashlib
from itertools import product
from math import isnan
import os
from glob import glob
from typing import Sequence

import numpy as np
import torch
from lightning import LightningModule
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
import tqdm

from .task_config_parser import TaskConfigParser


class MultiRunCLI:
    def __init__(self, *args: str) -> None:
        self.cli_argv = list(args)
        self.task_config_path, self.cli_argv = self.__extract_task_config()
        self.ckpt_path, self.cli_argv = self.__extract_ckpt_path()
        assert (
            self.task_config_path is not None
        ), "MULTI_RUN_CLI:__INIT__:TASK_CONFIG_ACQUIRING:ARGUMENT_MISSING: Task config is required by providing --task_config=<path> or --task_config <path>"
        self.task_config_parser = TaskConfigParser(self.task_config_path)

    def __extract_ckpt_path(self) -> tuple[str | None, list[str]]:
        cli_argv: list[str] = self.cli_argv
        ckpt_path = None
        for i, arg in enumerate(self.cli_argv):
            if arg.startswith("--ckpt_path"):
                if "=" in arg:
                    ckpt_path = arg.split("=")[1]
                    cli_argv.pop(i)
                elif i + 1 < len(self.cli_argv):
                    ckpt_path = self.cli_argv[i + 1]
                    cli_argv.pop(i)
                    cli_argv.pop(i)
                break
        if ckpt_path is not None and not os.path.isfile(self.ckpt_path):  # type: ignore
            raise FileNotFoundError(
                f"MULTI_RUN_CLI:__INIT__:CKPT_VALIDATION:FILE_NOT_FOUND: "
                f"Checkpoint file {self.ckpt_path} does not exist"
            )
        return ckpt_path, cli_argv

    def __extract_task_config(self) -> tuple[str | None, list[str]]:
        cli_argv: list[str] = self.cli_argv
        task_config_path = None
        for i, arg in enumerate(self.cli_argv):
            if arg.startswith("--task_config"):
                if "=" in arg:
                    task_config_path = arg.split("=")[1]
                    cli_argv.pop(i)
                elif i + 1 < len(self.cli_argv):
                    task_config_path = self.cli_argv[i + 1]
                    cli_argv.pop(i)
                    cli_argv.pop(i)
                break
        assert (
            task_config_path is not None
        ), "MULTI_RUN_CLI:__INIT__:TASK_CONFIG_ACQUIRING:ARGUMENT_MISSING: Task config is required by providing --task_config=<path> or --task_config <path>"

        return task_config_path, cli_argv

    def __generate_config_hash(self, config_list: list[str]) -> int:
        """生成配置列表的确定性哈希种子"""
        # 创建稳定字符串表示
        config_str = "|".join(sorted(config_list)).encode("utf-8")
        # 生成SHA256哈希
        hash_digest = hashlib.sha256(config_str).digest()
        # 转换为0-2^32范围内的整数
        return int.from_bytes(hash_digest[:4], byteorder="big") % (2**32)

    def __prepare_fold_idx(self, config_list: list[str]):
        assert "--data.init_args.n_folds" in config_list, (
            "MULTI_RUN_CLI:__PREPARE_FOLD_IDX:ARGUMENT_MISSING: "
            "Argument --data.init_args.n_folds is required to prepare fold indices"
        )
        n_folds = int(config_list[config_list.index("--data.init_args.n_folds") + 1])
        if "--data.init_args.val_fold_idx" not in self.cli_argv:
            val_fold_idx = [str(x) for x in range(n_folds)]
        else:
            val_fold_idx = self.cli_argv[
                self.cli_argv.index("--data.init_args.val_fold_idx") + 1
            ]

        if "--data.init_args.test_fold_idx" not in self.cli_argv:
            test_fold_idx = [str(x) for x in range(n_folds)]
        else:
            test_fold_idx = self.cli_argv[
                self.cli_argv.index("--data.init_args.test_fold_idx") + 1
            ]

        yield from product(val_fold_idx, test_fold_idx)

    def run(
        self,
        verbose: bool = True,
        save_config: bool = True,
        extra_experiment_name: str = "",
    ):
        accumulated_results: dict[str, list] = {}
        for config_list, experiment_name in self.task_config_parser.generate_configs():
            for val_fold_idx, test_fold_idx in self.__prepare_fold_idx(config_list):
                if val_fold_idx == test_fold_idx:
                    # 如果验证集和测试集折叠索引相同，则跳过
                    continue
                # 替换配置列表中的折叠索引
                cli = NamedParamsCLI(
                    parser_kwargs={"parser_mode": "omegaconf"},
                    args=(
                        self.cli_argv
                        + config_list
                        + (
                            [
                                "--seed_everything",
                                str(self.__generate_config_hash(config_list)),
                            ]
                            if "--seed_everything" not in self.cli_argv
                            else []
                        )
                        + ["--data.init_args.val_fold_idx", val_fold_idx]
                        + ["--data.init_args.test_fold_idx", test_fold_idx]
                        + [
                            "--experiment_name",
                            f"{experiment_name}-{extra_experiment_name}",
                        ]
                    ),
                    run=False,
                    save_config_callback=SaveConfigCallback if save_config else None,
                )

                cli.trainer.fit(
                    model=cli.model, datamodule=cli.datamodule, ckpt_path=self.ckpt_path
                )
                if hasattr(cli.model, "fake_parameter"):
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
                        )
                        for key, value in results[0].items():
                            assert not isnan(
                                value
                            ), f"Evaluation metrics got NaN for {key}"
                            accumulated_results.setdefault(key, []).append(value)
                else:
                    results = cli.trainer.test(
                        model=cli.model,
                        datamodule=cli.datamodule,
                        ckpt_path="best",
                        verbose=verbose,
                    )
                    for key, value in results[0].items():
                        assert not isnan(value), f"Evaluation metrics got NaN for {key}"
                        accumulated_results.setdefault(key, []).append(value)
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
            "data.sample_weights",
            "model.init_args.loss_args.weight",
            apply_on="instantiate",
        )
        parser.add_argument("--experiment_name", type=str)
        parser.link_arguments(
            ("experiment_name", "data.init_args.window_length"),
            "trainer.logger.init_args.name",
            compute_fn=lambda exp_name, window_length: os.path.join(
                *exp_name.split("-"), str(window_length)
            ),
        )
