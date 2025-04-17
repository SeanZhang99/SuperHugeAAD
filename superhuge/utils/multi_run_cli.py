import torch
import lightning.pytorch.callbacks
from lightning.pytorch.cli import LightningCLI
import tqdm

from ..datasets.commons.data_interface import DInterface

from ..models.commons.m_interface import MInterface

from ..models.rebok_deformer.deformer import deformer

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
        for config_list in self.task_config_parser.generate_configs():
            cli = NamedParamsCLI(
                parser_kwargs={"parser_mode": "omegaconf"},
                args=self.cli_argv + config_list,
                run=False,
                model_class=MInterface,
                datamodule_class=DInterface,
                subclass_mode_data=True,
                subclass_mode_model=True,
            )
            cli.trainer.fit(
                model=cli.model,
                datamodule=cli.datamodule,
            )
            cli.trainer.test(
                model=cli.model,
                datamodule=cli.datamodule,
                # dataloaders=cli.datamodule.val_dataloader(),
                ckpt_path="best",
                verbose=True,
            )


class NamedParamsCLI(LightningCLI):
    model: torch.nn.Module

    def _get_parameters(self):
        return self.model.named_parameters()

    def add_arguments_to_parser(self, parser):
        # When linking arguments, make sure the target argument is declared by the target class.
        parser.link_arguments(
            "data.fs",
            "model.init_args.module.init_args.fs",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.window_length",
            "model.init_args.module.init_args.window_length",
            apply_on="instantiate",
        )

    # parser.link_arguments(
    #     "data.init_args.fs",
    #     "data.init_args.dataset_args.meta_filter_func.init_args.init_args.fs",
    # )
