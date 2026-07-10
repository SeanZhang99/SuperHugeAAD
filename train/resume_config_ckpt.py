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

from superhuge.utils.multi_run_cli import NamedParamsCLI

torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":
    config_path = rf"C:\Users\sean\xwechat_files\wxid_o7yyhkckp1ya22_2314\msg\file\2026-01\config.yaml"
    model_ckpt_path = rf"C:\Users\sean\xwechat_files\wxid_o7yyhkckp1ya22_2314\msg\file\2026-01\last.ckpt"
    cli = NamedParamsCLI(
        args=["--config", config_path, "--model.init_args.ckpt_path", model_ckpt_path],
        run=False,
    )
    cli.trainer.test()
