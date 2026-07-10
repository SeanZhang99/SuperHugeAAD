import torch
import torch.nn as nn

from mamba_ssm import Mamba


class Bimamba_outer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state,
        d_conv,
        expand,
    ):
        super().__init__()
        self.forward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.backward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.output_proj = nn.Linear(2 * d_model, d_model)

    def forward(self, hidden_input: torch.Tensor) -> torch.Tensor:
        forward_output: torch.Tensor = self.forward_mamba(hidden_input)
        backward_output: torch.Tensor = self.backward_mamba(hidden_input.flip([1]))
        res = torch.cat((forward_output, backward_output.flip([1])), dim=-1)
        res = self.output_proj(res)
        return res
