import sys

from lightning.pytorch.cli import LightningCLI

from .task_config_parser import TaskConfigParser


class MultiRunCLI:
    def __init__(self, *args: str) -> None:
        self.cli_argv = list(args) + sys.argv[1:]
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
            cli = LightningCLI(
                parser_kwargs={"parser_mode": "omegaconf"},
                args=self.cli_argv + config_list,
            )
