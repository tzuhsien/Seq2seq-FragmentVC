"""Useful utilities."""

from typing import Tuple, List, Optional
import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from fairseq.models.wav2vec import Wav2Vec2Model


def load_pretrained_wav2vec(ckpt_path):
    """Load pretrained Wav2Vec model."""
    ckpt = torch.load(ckpt_path)
    model = Wav2Vec2Model.build_model(ckpt["args"], task=None)
    model.load_state_dict(ckpt["model"])
    model.remove_pretraining_modules()
    model.eval()
    return model


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi *
                        float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def generate_position_embedding(shape: List[int]) -> torch.Tensor:
    """
    Geberate origin position embedding in 'Attention Is All Your Need'
    size: (len, batch, dimension)
    reference: https://github.com/wzlxjtu/PositionalEncoding2D
    """
    length, batch_size, d_model = shape
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dim (got dim={:d})".format(d_model)
        )
    position_embedding = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(
        (
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
    )
    # position_embedding: (length, d_model)
    position_embedding[:, 0::2] = torch.sin(position.float() * div_term)
    position_embedding[:, 1::2] = torch.cos(position.float() * div_term)

    # position_embedding: (batch_size, length, d_model)
    position_embedding = position_embedding.repeat(batch_size, 1, 1)
    # position_embedding: (length, batch_size, d_model)
    position_embedding = position_embedding.permute(1, 0, 2)

    return position_embedding


def generate_square_subsequent_mask(size: int) -> torch.Tensor:
    """
    Generate a square mask for the sequence.
    The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def generate_diagonal_mask(size: int) -> torch.Tensor:
    """
    Generate a diagonal mask for the sequence.
    The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = torch.zeros(size, size)
    mask = mask.fill_diagonal_(1)
    mask = mask.masked_fill(mask == 0, float(
        "-inf")).masked_fill(mask == 1, float(0.0))
    return mask
