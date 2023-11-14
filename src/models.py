import os
from collections import OrderedDict
import torch
from torch import nn

from src.utils import mean_pool
from src.models_custom import ChannelDropout1d


### baselines


class BaselineModelV1(nn.Module):
    """
    бейзлайн от организаторов
    """
    def __init__(self, num_classes: int, input_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        self.num_classes = num_classes
        self.bn = nn.LayerNorm(hidden_dim)
        self.projector = nn.Linear(input_dim, hidden_dim)
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, mask, **kwargs):
        """
        x - [N, T, D]
        mask - [N, T]
        """
        x = self.projector(x)  # [N, T, H]
        x = mean_pool(x, mask)  # [N, H]
        x = self.bn(x)  # [N, H]
        x = self.lin(x)  # [N, H]
        outs = self.fc(x)  # [N, C]
        return outs


class BaselineModelV2(nn.Module):
    """
    pontwise-conv -> multihead-additive-attention - flattent - fc
    """
    def __init__(self, hidden_dim=768, dropout=0.1, num_heads=4, num_classes=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_in = nn.Dropout(dropout)
        self.pointwise_conv = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout_ff = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_heads)
        )
        self.ln2 = nn.LayerNorm(hidden_dim * num_heads)
        self.fc = nn.Linear(hidden_dim * num_heads, num_classes)

    def forward(self, x, mask, **kwargs):
        y = self.dropout_in(x)
        y = self.pointwise_conv(y)
        y = self.dropout_ff(y)
        x = self.ln1(x + y)
        w = self.attn(x)  # [N, T, H]

        w = w + (1.0 - mask[:, :, None].float()) * -10000.0
        w = torch.softmax(w / self.hidden_dim ** 0.5, dim=1)  # [N, T, H]
        x = x.unsqueeze(-2) * w.unsqueeze(-1)  # [N, T, H, D]
        x = x.sum(1)  # [N, H, D]
        x = self.ln2(x.reshape(x.shape[0], -1))  # [N, H * D]
        x = self.fc(x)

        return x


### custom encoders


class EncoderOnly(nn.Module):
    def __init__(
            self,
            config_cls,
            encoder_cls,
            config_params,
            num_classes,
            input_dim: int = 768,
            add_pos_emb: bool = False,
            pooling_type: str = "mean",  # {mean, attention}  # TODO: отдельный конфиг под пулер
            attn_pooler_v2_hidden: int = 128,  # TODO: отдельный конфиг под пулер
            attn_pooler_v2_dropout: float = 0.3,  # TODO: отдельный конфиг под пулер
            extend_attn_mask: bool = True,
            add_channel_dropout: bool = False,  # для обратной совместимости
            p_channel_dropout: float = 0.1
    ):
        super().__init__()
        assert pooling_type in {"mean", "attention_v1", "attention_v2"}, pooling_type
        self.pooling_type = pooling_type
        config = config_cls(**config_params)
        self.encoder = encoder_cls(config)

        # output layer
        self.fc = nn.Linear(config.hidden_size, num_classes)

        # channel dropout
        self.channel_dropout = None
        if add_channel_dropout:
            self.channel_dropout = ChannelDropout1d(p_channel_dropout)

        # input projection
        self.fc_in = None
        if config.hidden_size != input_dim:
            self.fc_in = nn.Linear(input_dim, config.hidden_size)

        # pos embeddings
        self.pos_emb = None
        if add_pos_emb:
            t_max = 512  # TODO: в конфиг
            x = torch.zeros((t_max, config.hidden_size))
            bound = 1.0 / config.hidden_size ** 0.5
            x.uniform_(-bound, bound)
            self.pos_emb = nn.Parameter(x)  # [T_max, D]

        # pooler
        if self.pooling_type == "mean":
            self.pooler = mean_pool
        elif self.pooling_type == "attention_v1":
            self.fc_pool = nn.Linear(config.hidden_size, 1)
            self.pooler = self._attention_pool_v1
        elif self.pooling_type == "attention_v2":
            self.attn_pooler_v2_hidden = attn_pooler_v2_hidden
            self.fc_pool1 = nn.Linear(config.hidden_size, self.attn_pooler_v2_hidden)
            self.fc_pool1_dropout = nn.Dropout(attn_pooler_v2_dropout)
            self.fc_pool2 = nn.Linear(self.attn_pooler_v2_hidden, 1)
            self.pooler = self._attention_pool_v2
        else:
            raise

        self.extend_attn_mask = extend_attn_mask
        self.config = config

    def forward(self, x, mask, **kwargs):
        x = self.get_logits(x, mask)
        x = self.pooler(x, mask)
        x = self.fc(x)
        return x

    def get_logits(self, x, mask):
        # maybe dropout
        if self.channel_dropout is not None:
            x = self.channel_dropout(x)

        # maybe project
        x = self.fc_in(x) if self.fc_in is not None else x

        # maybe add pos emb
        if self.pos_emb is not None:
            x = x + self.pos_emb[:x.shape[1]][None]  # [N, T, D] + [1, T, D]

        # maybe extend mask
        if self.extend_attn_mask:
            extended_attention_mask = (1.0 - mask[:, None, None, :].to(dtype=x.dtype)) * -10000.0  # [N, 1, 1, T]
        else:
            extended_attention_mask = mask

        # get logits
        encoder_outputs = self.encoder(x, attention_mask=extended_attention_mask)
        x = encoder_outputs[0]  # [N, T, D]
        return x

    def _attention_pool_v1(self, x, mask):
        """
        https://arxiv.org/abs/1409.0473

        x: [N, T, D]
        mask: [N, T]
        return: [N, D]
        """
        w = self.fc_pool(x)  # [N, T, 1]
        w = torch.tanh(w)  # [N, T, 1]
        w = w + (1.0 - mask[:, :, None].float()) * -10000.0
        w = torch.softmax(w, dim=1)  # [N, T, 1]
        x = (x * w).sum(1)  # [N, D]
        return x

    def _attention_pool_v2(self, x, mask):
        """
        https://arxiv.org/abs/1409.0473

        x: [N, T, D]
        mask: [N, T]
        return: [N, D]
        """
        w = self.fc_pool1(x)  # [N, T, h]
        w = torch.relu(w)
        w = self.fc_pool1_dropout(w)
        w = self.fc_pool2(w)  # [N, T, 1]
        w = w * 1.0 / self.attn_pooler_v2_hidden ** 0.5
        w = w + (1.0 - mask[:, :, None].float()) * -10000.0
        w = torch.softmax(w, dim=1)  # [N, T, 1]
        x = (x * w).sum(1)  # [N, D]
        return x


class EncoderOnlyFromPretrained(nn.Module):
    def __init__(self, config_cls, encoder_cls, pretrained_dir, num_classes):
        super().__init__()
        config = config_cls.from_pretrained(os.path.join(pretrained_dir, "config.json"))
        self.encoder = encoder_cls(config)
        d = torch.load(os.path.join(pretrained_dir, "pytorch_model.bin"), map_location="cpu")
        prefix = "roberta.encoder."  # TODO: нужно вынести в параметры, так как зависит от претреина
        self.encoder.load_state_dict(OrderedDict({k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}))
        self.fc = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x, mask, **kwargs):
        extended_attention_mask = (1.0 - mask[:, None, None, :].to(dtype=x.dtype)) * -10000.0
        encoder_outputs = self.encoder(x, attention_mask=extended_attention_mask)
        x = encoder_outputs[0]
        x = mean_pool(x, mask)
        x = self.fc(x)
        return x
