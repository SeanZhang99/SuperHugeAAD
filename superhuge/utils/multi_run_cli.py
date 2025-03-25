import sys

import lightning
import lightning.pytorch
import lightning.pytorch.callbacks
from lightning.pytorch.cli import LightningCLI
import torch
import tqdm

from .task_config_parser import TaskConfigParser


class MultiRunCLI:
    def __init__(self, *args: str) -> None:
        self.cli_argv = list(args)
        self.task_config_path, self.cli_argv = self.__extract_task_config()
        assert (
            self.task_config_path is not None
        ), "MULTI_RUN_CLI:__INIT__:TASK_CONFIG_ACQUIRING:ARGUMENT_MISSING: Task config is required by providing --task_config=<path> or --task_config <path>"
        self.task_config_parser = TaskConfigParser(self.task_config_path)
        self.__run_cli()

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
        return task_config_path, cli_argv

    def __run_cli(self):
        def to_device(data, device):
            if isinstance(data, dict):
                return {k: to_device(v, device) for k, v in data.items()}
            elif isinstance(data, (list, tuple)):
                return [to_device(v, device) for v in data]
            elif isinstance(
                data, (torch.Tensor, lightning.pytorch.LightningDataModule)
            ):
                return data.to(device)
            return data

        for config_list in self.task_config_parser.generate_configs():
            cli = LightningCLI(
                # parser_kwargs={"parser_mode": "omegaconf"},
                args=self.cli_argv + config_list,
                run=False,
            )
            # batch = cli.datamodule.train_dataloader()._get_iterator().__next__()
            # batch = to_device(batch, "cuda")
            # cli.model.to("cuda")
            # for i in tqdm.trange(100):
            #     cli.model.forward(batch)
            cli.trainer.fit(
                cli.model,
                datamodule=cli.datamodule,
            )
            cli.trainer.test(
                cli.model,
                datamodule=cli.datamodule,
                ckpt_path="best",
                verbose=True,
            )
            break
