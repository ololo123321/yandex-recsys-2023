import os
from typing import List, Union
import torch
from torch import nn

from omegaconf import OmegaConf
import hydra

from src.utils import mean_pool


def load_checkpoint(checkpoint_path):
    train_cfg = OmegaConf.load(os.path.join(os.path.dirname(checkpoint_path), "..", ".hydra", "config.yaml"))
    model = hydra.utils.instantiate(train_cfg.model)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return model


def get_paths(checkpoint_paths):
    if isinstance(checkpoint_paths, list):
        return checkpoint_paths
    elif isinstance(checkpoint_paths, str):
        res = []
        with open(checkpoint_paths) as f:
            for line in f:
                res.append(line.strip())
        return res
    else:
        raise


# проверил на примере двух моделей: не зашло :(
class EnsembleV1(nn.Module):
    """
    требования к эксперту:
    * атрибут config с атрибутом hidden_size
    * метод get_logits, возвращающий тензор [N, T, D]

    пока учим только адаптеры
    """
    def __init__(self, checkpoint_paths: Union[str, List[str]], input_dim: int = 768, hidden_dim: int = 512, dropout: float = 0.1, num_classes: int = 256):
        super().__init__()
        """
        checkpoint_paths - если str, то путь к файлу со списком путей
        """
        self.experts = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for checkpoint_path in get_paths(checkpoint_paths):
            model = load_checkpoint(str(checkpoint_path))
            # disable grad
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
            # maybe create adapter
            if model.config.hidden_size == hidden_dim:
                self.adapters.append(nn.Identity())
            else:
                self.adapters.append(nn.Linear(model.config.hidden_size, hidden_dim))
            # create dropout for each expert
            self.dropouts.append(nn.Dropout(dropout))
            self.experts.append(model)

        self.fc_gate = nn.Linear(input_dim, len(self.experts))
        self.pooler = mean_pool
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim

    def forward(self, x, mask, **kwargs):
        # gating
        g = torch.softmax(self.fc_gate(x) / self.hidden_dim ** 0.5, dim=-1)  # [N, T, num_experts]

        # weighted sum
        y = torch.zeros((x.shape[0], x.shape[1], self.hidden_dim), device=x.device, dtype=x.dtype)
        for i, (expert, adapter, dropout) in enumerate(zip(self.experts, self.adapters, self.dropouts)):
            with torch.no_grad():
                out = expert.get_logits(x, mask)
            out = dropout(out)
            out = adapter(out)
            y = y + out * g[:, :, i].unsqueeze(-1)
        x = self.pooler(y, mask)
        x = self.fc(x)
        return x


class EnsembleV2(nn.Module):
    """
    учим линейную комбинацию экспертов на уровне скоров классов
    """
    def __init__(self, checkpoint_paths: List[str], dropout: float = 0.2, num_classes: int = 256):
        super().__init__()
        self.experts = nn.ModuleList()
        for checkpoint_path in get_paths(checkpoint_paths):
            model = load_checkpoint(str(checkpoint_path))
            # disable grad
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
            self.experts.append(model)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(self.experts), 1)
        self.num_classes = num_classes

    def forward(self, x, mask, **kwargs):
        y = torch.zeros((x.shape[0], self.num_classes, len(self.experts)), device=x.device, dtype=x.dtype)
        for i, expert in enumerate(self.experts):
            with torch.no_grad():
                y[:, :, i] = expert(x, mask)
        x = self.dropout(y)
        x = self.fc(x).squeeze(-1)
        return x
