"""Several modules"""
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor


class CausalCNN1d(nn.Module):
    """
    CausalCNN1d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding_mode: str = "replicate",
    ):
        super(CausalCNN1d, self).__init__()
        if padding_mode == "zeros":
            self.pad_layer = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
        elif padding_mode == "reflect":
            self.pad_layer = nn.ReflectionPad1d((kernel_size - 1, 0))
        else:
            self.pad_layer = nn.ReplicationPad1d((kernel_size - 1, 0))

        self.conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )

    def forward(self, inp: Tensor) -> Tensor:
        """
        inp: (batch, d_model, inp_len)
        """
        out = self.pad_layer(inp)
        out = self.conv1d(out)
        return out


class InvertCausalCNN1d(nn.Module):
    """
    InvertCausalCNN1d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding_mode: str = "replicate",
    ):
        super(InvertCausalCNN1d, self).__init__()
        if padding_mode == "zeros":
            self.pad_layer = nn.ConstantPad1d((0, kernel_size - 1), 0.0)
        elif padding_mode == "reflect":
            self.pad_layer = nn.ReflectionPad1d((0, kernel_size - 1))
        else:
            self.pad_layer = nn.ReplicationPad1d((0, kernel_size - 1))

        self.conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )

    def forward(self, inp: Tensor) -> Tensor:
        """
        inp: (batch, d_model, inp_len)
        """
        out = self.pad_layer(inp)
        out = self.conv1d(out)
        return out


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, inp: Tensor, att_mask: Optional[Tensor] = None):
        """
        N: batch size, T: sequence length, H: Hidden dimension
        input:
            inp : size (N, T, H)
        attention_weight:
            att_w : size (N, T, 1)
        return:
            utter_rep: size (N, H)
        """
        att_logits = self.W(inp).squeeze(-1)
        if att_mask is not None:
            att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(inp * att_w, dim=1)

        return utter_rep
