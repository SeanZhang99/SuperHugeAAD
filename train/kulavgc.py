# Copyright 2021 Zhongyang Zhang
# Contact: mirakuruyoo@gmai.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This package is adopted based on Pytorch Lightning Template project.
# Author: Yuanming Zhang

"""This main entrance of the whole project.

Most of the code should not be changed, please directly
add all the input arguments of your model's constructor
and the dataset file's constructor. The MInterface and
DInterface can be seen as transparent to all your args.
"""
import os


os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch

from superhuge.utils.multi_run_cli import MultiRunCLI

torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":
    project_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_path, "configs")
    model_config = os.path.join(project_path, "configs", "models", "wf.yaml")
    cli = MultiRunCLI(
        "--config",
        os.path.join(config_path, "trainer_config.yaml"),
        "--config",
        os.path.join(config_path, "linear_trainer.yaml"),
        "--data",
        os.path.join(
            config_path,
            "data_config.yaml" if os.name == "nt" else "wsl_data_config.yaml",
        ),
        "--model",
        model_config,
        "--model",
        os.path.join(config_path, "optimizer_config.yaml"),
        "--model.init_args.summary_verbose",
        "true",
        task_config_path=os.path.join(config_path, "task_config.yaml"),
    )
    cli.run()
