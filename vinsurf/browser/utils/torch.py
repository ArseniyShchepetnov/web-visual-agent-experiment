"""Torch utils."""

import torch


def get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Calculate vector norm.

    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True)
    and used to make model `executorch` exportable.
    See issue https://github.com/pytorch/executorch/issues/3566.
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    return torch.sqrt(sum_tensor)
