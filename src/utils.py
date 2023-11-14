import torch


def mean_pool(x, mask):
    """
    x - [N, T, D]
    mask - [N, T]
    """
    return (x * mask[..., None].float()).sum(1) / mask.float().sum(1)[:, None]


def get_sequence_mask(lengths, max_length: int = None):
    max_length = max_length if max_length is not None else lengths.max().item()
    return torch.arange(max_length, device=lengths.device)[None] < lengths[:, None]
