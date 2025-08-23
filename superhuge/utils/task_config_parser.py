from collections.abc import Generator
from copy import deepcopy
from itertools import product
import os
from typing import Any

import yaml


class TaskConfigParser:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config: dict[str, dict[str, dict[str, Any]]] = self._load_config()
        self.tasks = self._get_enabled_tasks()
        self.cross_validations = self._get_enabled_cross_validations()
        self.last_temp_file = None

    def _load_config(self) -> dict:
        with open(self.config_path, "r") as file:
            return yaml.safe_load(file)

    def _get_enabled_tasks(self) -> dict[str, dict]:
        tasks: dict[str, dict] = {}
        for task_type, task_info in self.config.items():
            for task_name, task_details in task_info.get("tasks", {}).items():
                if task_details.get("enable", 0) == 1:
                    tasks.setdefault(task_type, {})[task_name] = task_details
        return tasks

    def _get_enabled_cross_validations(self) -> dict[str, dict]:
        cross_validations: dict[str, dict] = {}
        for task_type, task_info in self.config.items():
            for cv_name, cv_details in task_info.get("cross_validation", {}).items():
                if cv_details.get("enable", 0) == 1:
                    cross_validations.setdefault(task_type, {})[cv_name] = cv_details
        return cross_validations

    def _deep_merge_dicts(self, base: dict, *updates: dict) -> dict:
        merged = deepcopy(base)
        for update in updates:
            for key, value in update.items():
                if isinstance(value, dict) and key in merged:
                    merged[key] = self._deep_merge_dicts(merged[key], value)
                else:
                    merged[key] = value
        return merged

    def _dict_to_cli_args(self, prefix: str, data: dict) -> list[str]:
        args = []
        for key, value in data.items():
            if isinstance(value, dict):
                args.extend(self._dict_to_cli_args(f"{prefix}.{key}", value))
            else:
                args.append(f"{prefix}.{key}")
                args.append(str(value))
        return args

    def generate_configs(self)  -> Generator[tuple[list[str], str], Any, None]:
        for task_type, task_details in self.tasks.items():
            general_data: dict = self.config[task_type]["general"]["data"]
            for task_name, task_detail in task_details.items():
                for cv_name, cv_details in self.cross_validations.get(
                    task_type, {}
                ).items():
                    n_folds = cv_details["data"]["init_args"].get("n_folds", None)
                    if n_folds is None:
                        n_folds = task_detail["data"]["init_args"].get("n_folds", None)
                    if n_folds is None:
                        n_folds = general_data["init_args"].get("n_folds", None)
                    assert n_folds is not None, "n_folds not specified."
                    task_data: dict = task_detail.get("data", {})
                    cv_data: dict = cv_details.get("data", {})
                    merged_data: dict = self._deep_merge_dicts(
                        general_data, task_data, cv_data
                    )

                    general_model: dict = self.config[task_type]["general"].get(
                        "model", {}
                    )
                    task_model: dict = task_detail.get("model", {})
                    cv_model: dict = cv_details.get("model", {})
                    merged_model: dict = self._deep_merge_dicts(
                        general_model, task_model, cv_model
                    )

                    config_copy: dict = {
                        "data": merged_data,
                        "model": merged_model,
                    }

                    cli_args: list[str] = []
                    cli_args.extend(
                        self._dict_to_cli_args("--data", config_copy["data"])
                    )
                    cli_args.extend(
                        self._dict_to_cli_args("--model", config_copy["model"])
                    )

                    yield cli_args, "-".join(
                        [
                            task_type,
                            task_name,
                            cv_name,
                        ]
                    )


if __name__ == "__main__":
    config_path = r"C:/Users/Sean/Documents/Seafile/ZYMdeDocument/24-12-SuperHugeAAD/SuperHugeAAD/scripts/dnn/configs/task_config.yaml"
    parser = TaskConfigParser(config_path)

    for config_file in parser.generate_configs():
        print(f"Generated config file: {config_file}")
