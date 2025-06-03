import os, hashlib, gc
from lightning import LightningModule
import numpy as np
import torch
from lightning.pytorch.cli import LightningCLI

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
        self.__run_cli()

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
        if ckpt_path is not None and not os.path.isfile(self.ckpt_path):
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

    def __run_cli(self):
        for config_list in self.task_config_parser.generate_configs():
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
                        and "--seed_everything" not in self.cli_argv
                        else []
                    )
                ),
                run=False,
            )

            cli.trainer.fit(
                model=cli.model, datamodule=cli.datamodule, ckpt_path=self.ckpt_path
            )
            cli.trainer.test(
                model=cli.model,
                datamodule=cli.datamodule,
                ckpt_path="best",
                verbose=True,
            )
            self.__release_resources(cli)

    def __release_resources(self, cli: "NamedParamsCLI"):
        """资源释放策略"""
        # 释放模型引用
        del cli.model
        del cli.datamodule
        del cli.trainer

        # 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 双保险清理
            torch.cuda.ipc_collect()

        # 强制垃圾回收
        gc.collect()


class NamedParamsCLI(LightningCLI):
    model: LightningModule

    def _get_parameters(self):
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

    # parser.link_arguments(
    #     "data.init_args.fs",
    #     "data.init_args.dataset_args.meta_filter_func.init_args.init_args.fs",
    # )
