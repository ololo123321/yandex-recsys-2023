import torch
from torch import nn

from transformers import BertConfig, DebertaConfig
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.models.deberta.modeling_deberta import DisentangledSelfAttention, build_relative_position

from src.utils import mean_pool

# models


class TransformerEncoderCustomV1(nn.Module):
    """
    * pre-ln
    * no pos emb
    * bert self attention with relative embeddings
    * mean pooling
    """
    def __init__(
            self,
            config: BertConfig,
            input_dim: int,
            num_classes: int,
            position_embedding_type: str = None,
            pooling_type: str = "mean"
    ):
        assert (position_embedding_type is None) \
               or position_embedding_type in {"absolute", "relative_key", "relative_key_query"}, position_embedding_type
        assert pooling_type in {"mean", "attn"}, pooling_type
        super().__init__()

        # input dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # input projection
        self.fc_in = None
        if config.hidden_size != input_dim:
            self.fc_in = nn.Linear(input_dim, config.hidden_size)

        # transformer
        self.layers = nn.ModuleList([
            TransformerLayerPreLN(
                mha=BertSelfAttention(config, position_embedding_type),
                d=config.hidden_size,
                intermediate_size=config.intermediate_size,
                act="gelu",
                dropout=config.hidden_dropout_prob
            )
            for _ in range(config.num_hidden_layers)
        ])

        if pooling_type == "mean":
            self.pooler = mean_pool
        elif pooling_type == "attn":
            self.mlp_pool = nn.Sequential(
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, 1)
            )
            self.pooler = self._attention_pool
        else:
            raise

        # output layer
        self.fc = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x, mask, **kwargs):
        """
        x - [N, T, D]
        mask - [N, T]
        """
        # input dropout
        x = self.dropout(x)

        # input proj
        if self.fc_in is not None:
            x = self.fc_in(x)

        extended_attention_mask = (1.0 - mask[:, None, None, :].to(dtype=x.dtype)) * -10000.0  # [N, 1, 1, T]
        for layer in self.layers:
            x = layer(x, extended_attention_mask)  # [N, T, D]
        x = mean_pool(x, mask)  # [N, D]
        x = self.fc(x)  # [N, C]
        return x

    def _attention_pool(self, x, mask):
        """
        x: [N, T, D]
        mask: [N, T]
        return: [N, D]
        """
        d_model = x.shape[-1]
        w = self.mlp_pool(x) / d_model ** 0.5  # [N, T, 1]
        w = w + (1.0 - mask[:, :, None].float()) * -10000.0
        w = torch.softmax(w, dim=1)  # [N, T, 1]
        x = (x * w).sum(1)  # [N, D]
        return x


class TransformerEncoderCustomV2(nn.Module):
    """
    * pre-ln
    * no pos emb
    * deberta attention
    * mean pooling
    """
    def __init__(
            self,
            config: DebertaConfig,
            input_dim: int,
            num_classes: int
    ):
        super().__init__()

        # input dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # input projection
        self.fc_in = None
        if config.hidden_size != input_dim:
            self.fc_in = nn.Linear(input_dim, config.hidden_size)

        # relative embeddings
        # шарятся между слоями
        assert config.max_relative_positions > 1, config.max_relative_positions
        self.rel_embeddings = nn.Embedding(config.max_relative_positions * 2, config.hidden_size)

        # transformer
        self.layers = nn.ModuleList([
            TransformerLayerPreLN(
                mha=DisentangledSelfAttention(config),
                d=config.hidden_size,
                intermediate_size=config.intermediate_size,
                act="gelu",
                dropout=config.hidden_dropout_prob
            )
            for _ in range(config.num_hidden_layers)
        ])

        # output layer
        self.fc = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x, mask, **kwargs):
        """
        x - [N, T, D]
        mask - [N, T]
        """
        # input dropout
        x = self.dropout(x)

        # input proj
        if self.fc_in is not None:
            x = self.fc_in(x)

        # build mask
        # [N, 1, 1, T] * [N, 1, T, 1] = [N, 1, T, T]
        attention_mask = mask[:, None, None, :] * mask[:, None, :, None]

        # get relative positions
        relative_pos = build_relative_position(x.shape[1], x.shape[1], x.device)

        # transformer
        for layer in self.layers:
            x = layer(
                x,
                attention_mask,
                relative_pos=relative_pos,
                rel_embeddings=self.rel_embeddings.weight,
                output_attentions=True  # to return tuple (for compatibility with other mha layers)
            )  # [N, T, D]

        # output
        x = mean_pool(x, mask)  # [N, D]
        x = self.fc(x)  # [N, C]
        return x


class TransformerEncoderCustomV3(nn.Module):
    def __init__(
            self,
            num_experts: int = 3,
            num_layers: int = 1,
            num_heads: int = 12,
            d_model: int = 384,
            dff: int = 1536,
            input_dim: int = 768,
            num_classes: int = 256,
            dropout: float = 0.1,
            learnable_norm: bool = False
    ):
        super().__init__()
        self.input_dropout = nn.Dropout(dropout)
        self.input_proj = None
        if input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)
        # self.experts = nn.ModuleList([
        #     nn.TransformerEncoder(
        #         encoder_layer=nn.TransformerEncoderLayer(
        #             d_model=d_model,
        #             nhead=num_heads,
        #             dim_feedforward=dff,
        #             activation="gelu",
        #             batch_first=True,
        #             norm_first=True
        #         ),
        #         num_layers=num_layers
        #     ) for _ in range(num_experts)
        # ])
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = nn.ModuleList([
                TransformerLayerPreLN(
                    mha=MHA(num_heads=num_heads, head_dim=d_model // num_heads),
                    d=d_model,
                    intermediate_size=dff,
                    act="gelu",
                    dropout=dropout
                )
                for _ in range(num_layers)
            ])
            self.experts.append(expert)
        self.num_experts = num_experts
        self.pooler = AttentionPoolerV1(d_model)
        self.fc_out = nn.Linear(d_model, num_classes)

        self.ln = None
        if learnable_norm:
            self.ln = nn.LayerNorm(d_model)

    def forward(self, x, mask, **kwargs):
        x = self.input_dropout(x)
        if self.input_proj is not None:
            x = self.input_proj(x)

        # accumulate hiddens
        y = torch.zeros_like(x)
        for expert in self.experts:
            out = x
            for layer in expert:
                out = layer(out, mask)
            y = y + out

        # normalize
        if self.ln is not None:
            x = self.ln(y)
        else:
            x = y / self.num_experts

        # pool, fc
        x = self.pooler(x, mask)
        x = self.fc_out(x)
        return x


class TransformerEncoderCustomV4(nn.Module):
    """v3 + gating"""
    def __init__(self, num_experts=3, num_layers=1, num_heads=12, d_model=384, dff=1536, input_dim=768, num_classes=256):
        super().__init__()
        self.input_dropout = nn.Dropout(0.1)
        self.input_proj = None
        if input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = nn.ModuleList([
                TransformerLayerPreLN(
                    mha=MHA(num_heads=num_heads, head_dim=d_model // num_heads),
                    d=d_model,
                    intermediate_size=dff,
                    act="gelu",
                    dropout=0.1
                )
                for _ in range(num_layers)
            ])
            self.experts.append(expert)
        self.fc_gate = nn.Linear(d_model, num_experts)
        self.pooler = AttentionPoolerV1(d_model)
        self.fc_out = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, x, mask, **kwargs):
        x = self.input_dropout(x)
        if self.input_proj is not None:
            x = self.input_proj(x)
        g = torch.softmax(self.fc_gate(x) / self.d_model ** 0.5, dim=-1)  # [N, T, num_experts]
        y = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            out = x
            for layer in expert:
                out = layer(out, mask)
            y = y + out * g[:, :, i].unsqueeze(-1)
        x = self.pooler(x, mask)
        x = self.fc_out(x)
        return x


class TransformerEncoderCustomV5(nn.Module):
    """
    v1 + sliding window mask
    """
    def __init__(
            self,
            config: BertConfig,
            input_dim: int,
            num_classes: int,
            position_embedding_type: str = None,
            window: int = 16,
            add_pos_emb: bool = False
    ):
        super().__init__()

        # input dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # input projection
        self.fc_in = None
        if config.hidden_size != input_dim:
            self.fc_in = nn.Linear(input_dim, config.hidden_size)

        self.pos_emb = None
        if add_pos_emb:
            t_max = 512  # TODO: в конфиг
            x = torch.zeros((t_max, config.hidden_size))
            bound = 1.0 / config.hidden_size ** 0.5
            x.uniform_(-bound, bound)
            self.pos_emb = nn.Parameter(x)  # [T_max, D]

        # transformer
        self.layers = nn.ModuleList([
            TransformerLayerPreLN(
                mha=BertSelfAttention(config, position_embedding_type),
                d=config.hidden_size,
                intermediate_size=config.intermediate_size,
                act="gelu",
                dropout=config.hidden_dropout_prob
            )
            for _ in range(config.num_hidden_layers)
        ])

        self.pooler = AttentionPoolerV1(config.hidden_size)

        # output layer
        self.fc = nn.Linear(config.hidden_size, num_classes)

        self.window = window
        self.config = config

    def forward(self, x, mask, **kwargs):
        """
        x - [N, T, D]
        mask - [N, T]
        """
        x = self.get_logits(x, mask)

        # pooler, fc
        x = self.pooler(x, mask)  # [N, D]
        x = self.fc(x)  # [N, C]
        return x

    def get_logits(self, x, mask):
        # input dropout
        x = self.dropout(x)

        # input proj
        if self.fc_in is not None:
            x = self.fc_in(x)

        # pos emb
        if self.pos_emb is not None:
            x = x + self.pos_emb[:x.shape[1]][None]  # [N, T, D] + [1, T, D]

        # transformer
        band_mask = self._get_mask(x.shape[1]).to(mask.device)  # [T, T]
        merged_mask = torch.logical_and(mask[:, :, None].bool(), band_mask[None])
        extended_attention_mask = (1.0 - merged_mask[:, None].to(dtype=x.dtype)) * -10000.0  # [N, 1, T, T]
        for layer in self.layers:
            x = layer(x, extended_attention_mask)  # [N, T, D]
        return x

    def _get_mask(self, n):
        xs = torch.arange(n)
        ys = torch.arange(n)
        diff = xs[None] - ys[:, None]
        k = self.window // 2 if self.window % 2 == 0 else self.window // 2 + 1
        mask = diff.abs() < k
        # все токены всегда аттендятся на первый; первый токен всегда аттендится на все
        mask[0, :] = True
        mask[:, 0] = True
        return mask


class TransformerEncoderCustomV6(nn.Module):
    """
    v6 + pointwise conv before transformer stack
    """
    def __init__(
            self,
            config: BertConfig,
            input_dim: int,
            num_classes: int,
            position_embedding_type: str = None,
            window: int = 16
    ):
        super().__init__()

        # input dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # input projection
        if config.hidden_size == input_dim:
            self.conv = IntermediateFeedForward(config.hidden_size, config.hidden_size * 4, "gelu")
        else:
            self.conv = IntermediateFeedForward(input_dim, input_dim * 4, config.hidden_size, "gelu")

        # transformer
        self.layers = nn.ModuleList([
            TransformerLayerPreLN(
                mha=BertSelfAttention(config, position_embedding_type),
                d=config.hidden_size,
                intermediate_size=config.intermediate_size,
                act="gelu",
                dropout=config.hidden_dropout_prob
            )
            for _ in range(config.num_hidden_layers)
        ])

        self.pooler = AttentionPoolerV1(config.hidden_size)

        # output layer
        self.fc = nn.Linear(config.hidden_size, num_classes)

        self.window = window

    def forward(self, x, mask, **kwargs):
        """
        x - [N, T, D]
        mask - [N, T]
        """
        # input dropout
        x = self.dropout(x)

        # conv
        x = self.conv(x)

        band_mask = self._get_mask(x.shape[1]).to(mask.device)  # [T, T]
        merged_mask = torch.logical_and(mask[:, :, None].bool(), band_mask[None])
        extended_attention_mask = (1.0 - merged_mask[:, None].to(dtype=x.dtype)) * -10000.0  # [N, 1, T, T]
        for layer in self.layers:
            x = layer(x, extended_attention_mask)  # [N, T, D]
        x = self.pooler(x, mask)  # [N, D]
        x = self.fc(x)  # [N, C]
        return x

    def _get_mask(self, n):
        xs = torch.arange(n)
        ys = torch.arange(n)
        diff = xs[None] - ys[:, None]
        k = self.window // 2 if self.window % 2 == 0 else self.window // 2 + 1
        mask = diff.abs() < k
        # все токены всегда аттендятся на первый; первый токен всегда аттендится на все
        mask[0, :] = True
        mask[:, 0] = True
        return mask


class TransformerEncoderCustomV7(nn.Module):
    """
    cross-attention (tags, tags)
    cross-attention (tokens, tags)
    """
    def __init__(
            self,
            config: BertConfig,
            input_dim: int,
            num_classes: int,
            position_embedding_type: str = None,
            window: int = 16,
            num_decoder_layers: int = 1
    ):
        super().__init__()
        self.num_classes = num_classes

        # input dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # input projection
        self.fc_in = None
        if config.hidden_size != input_dim:
            self.fc_in = nn.Linear(input_dim, config.hidden_size)

        # transformer
        self.encoder_layers = nn.ModuleList([
            TransformerLayerPreLN(
                mha=BertSelfAttention(config, position_embedding_type),
                d=config.hidden_size,
                intermediate_size=config.intermediate_size,
                act="gelu",
                dropout=config.hidden_dropout_prob
            )
            for _ in range(config.num_hidden_layers)
        ])

        self.decoder = DecoderPreLN(
            num_layers=num_decoder_layers,
            hidden_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            vocab_size=num_classes,
            dropout=config.hidden_dropout_prob
        )

        # output layer
        self.fc = nn.Linear(config.hidden_size, 1)

        self.window = window

    def forward(self, x, mask, **kwargs):
        # input dropout
        x = self.dropout(x)

        # input proj
        if self.fc_in is not None:
            x = self.fc_in(x)

        # encoder
        band_mask = self._get_mask(x.shape[1]).to(mask.device)  # [T, T]
        merged_mask = torch.logical_and(mask[:, :, None].bool(), band_mask[None])
        encoder_mask = (1.0 - merged_mask[:, None].to(dtype=x.dtype)) * -10000.0  # [N, 1, T, T]
        for layer in self.encoder_layers:
            x = layer(x, encoder_mask)  # [N, T, D]

        # decoder
        tag_ids = torch.tile(torch.arange(self.num_classes, device=x.device)[None], [x.shape[0], 1])  # [N, C]
        # маскируем только source на паддинг, должна быть broadcastable с [N, H, C, T]
        cross_attn_mask = (1.0 - mask[:, None, None, :].to(dtype=x.dtype)) * -10000.0  # [N, 1, 1, T]
        x = self.decoder(tag_ids, x, self_attn_mask=None, cross_attn_mask=cross_attn_mask)  # [N, C, D]

        # логиты классов
        logits = self.fc(x).squeeze(-1)  # [N, C, 1] -> [N, C]
        return logits

    def _get_mask(self, n):
        xs = torch.arange(n)
        ys = torch.arange(n)
        diff = xs[None] - ys[:, None]
        k = self.window // 2 if self.window % 2 == 0 else self.window // 2 + 1
        mask = diff.abs() < k
        # все токены всегда аттендятся на первый; первый токен всегда аттендится на все
        mask[0, :] = True
        mask[:, 0] = True
        return mask


class TransformerEncoderCustomV8(nn.Module):
    """
    v5 + свой пулер для каждого класса
    """
    def __init__(
            self,
            config: BertConfig,
            input_dim: int,
            num_classes: int,
            position_embedding_type: str = None,
            window: int = 16
    ):
        super().__init__()

        # input dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # input projection
        self.fc_in = None
        if config.hidden_size != input_dim:
            self.fc_in = nn.Linear(input_dim, config.hidden_size)

        # transformer
        self.layers = nn.ModuleList([
            TransformerLayerPreLN(
                mha=BertSelfAttention(config, position_embedding_type),
                d=config.hidden_size,
                intermediate_size=config.intermediate_size,
                act="gelu",
                dropout=config.hidden_dropout_prob
            )
            for _ in range(config.num_hidden_layers)
        ])

        self.pooler = MultiHeadAdditiveAttention(d=config.hidden_size, num_heads=num_classes)

        # output layer
        self.fc = nn.Linear(config.hidden_size, 1)

        self.window = window

    def forward(self, x, mask, **kwargs):
        """
        x - [N, T, D]
        mask - [N, T]
        """
        # input dropout
        x = self.dropout(x)

        # input proj
        if self.fc_in is not None:
            x = self.fc_in(x)

        band_mask = self._get_mask(x.shape[1]).to(mask.device)  # [T, T]
        merged_mask = torch.logical_and(mask[:, :, None].bool(), band_mask[None])
        extended_attention_mask = (1.0 - merged_mask[:, None].to(dtype=x.dtype)) * -10000.0  # [N, 1, T, T]
        for layer in self.layers:
            x = layer(x, extended_attention_mask)  # [N, T, D]
        x = self.pooler(x, mask)  # [N, C, D]
        x = self.fc(x).squeeze(-1)  # [N, C, 1] -> [N, C]  # only change
        return x

    def _get_mask(self, n):
        xs = torch.arange(n)
        ys = torch.arange(n)
        diff = xs[None] - ys[:, None]
        k = self.window // 2 if self.window % 2 == 0 else self.window // 2 + 1
        mask = diff.abs() < k
        # все токены всегда аттендятся на первый; первый токен всегда аттендится на все
        mask[0, :] = True
        mask[:, 0] = True
        return mask


class TransformerEncoderCustomV9(nn.Module):
    """
    cross-attention (tokens, tags)
    """
    def __init__(
            self,
            config: BertConfig,
            input_dim: int,
            num_classes: int,
            position_embedding_type: str = None,
            window: int = 16,
            num_decoder_layers: int = 1
    ):
        super().__init__()
        self.num_classes = num_classes

        # input dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # input projection
        self.fc_in = None
        if config.hidden_size != input_dim:
            self.fc_in = nn.Linear(input_dim, config.hidden_size)

        # transformer
        self.encoder_layers = nn.ModuleList([
            TransformerLayerPreLN(
                mha=BertSelfAttention(config, position_embedding_type),
                d=config.hidden_size,
                intermediate_size=config.intermediate_size,
                act="gelu",
                dropout=config.hidden_dropout_prob
            )
            for _ in range(config.num_hidden_layers)
        ])

        self.decoder = DecoderPreLN2(
            num_layers=num_decoder_layers,
            hidden_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            vocab_size=num_classes,
            dropout=config.hidden_dropout_prob
        )

        # output layer
        self.fc = nn.Linear(config.hidden_size, 1)

        self.window = window

    def forward(self, x, mask, **kwargs):
        # input dropout
        x = self.dropout(x)

        # input proj
        if self.fc_in is not None:
            x = self.fc_in(x)

        # encoder
        band_mask = self._get_mask(x.shape[1]).to(mask.device)  # [T, T]
        merged_mask = torch.logical_and(mask[:, :, None].bool(), band_mask[None])
        encoder_mask = (1.0 - merged_mask[:, None].to(dtype=x.dtype)) * -10000.0  # [N, 1, T, T]
        for layer in self.encoder_layers:
            x = layer(x, encoder_mask)  # [N, T, D]

        # decoder
        tag_ids = torch.tile(torch.arange(self.num_classes, device=x.device)[None], [x.shape[0], 1])  # [N, C]
        # маскируем только source на паддинг, должна быть broadcastable с [N, H, C, T]
        cross_attn_mask = (1.0 - mask[:, None, None, :].to(dtype=x.dtype)) * -10000.0  # [N, 1, 1, T]
        x = self.decoder(tag_ids, x, cross_attn_mask)  # [N, C, D]

        # логиты классов
        logits = self.fc(x).squeeze(-1)  # [N, C, 1] -> [N, C]
        return logits

    def _get_mask(self, n):
        xs = torch.arange(n)
        ys = torch.arange(n)
        diff = xs[None] - ys[:, None]
        k = self.window // 2 if self.window % 2 == 0 else self.window // 2 + 1
        mask = diff.abs() < k
        # все токены всегда аттендятся на первый; первый токен всегда аттендится на все
        mask[0, :] = True
        mask[:, 0] = True
        return mask


class TransformerEncoderCustomV10(nn.Module):
    """
    v5 + cross attention pooler
    """
    def __init__(
            self,
            config: BertConfig,
            input_dim: int,
            num_classes: int,
            position_embedding_type: str = None,
            window: int = 16,
            add_pos_emb: bool = False
    ):
        super().__init__()

        # input dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # input projection
        self.fc_in = None
        if config.hidden_size != input_dim:
            self.fc_in = nn.Linear(input_dim, config.hidden_size)

        self.pos_emb = None
        if add_pos_emb:
            t_max = 512  # TODO: в конфиг
            x = torch.zeros((t_max, config.hidden_size))
            bound = 1.0 / config.hidden_size ** 0.5
            x.uniform_(-bound, bound)
            self.pos_emb = nn.Parameter(x)  # [T_max, D]

        # transformer
        self.layers = nn.ModuleList([
            TransformerLayerPreLN(
                mha=BertSelfAttention(config, position_embedding_type),
                d=config.hidden_size,
                intermediate_size=config.intermediate_size,
                act="gelu",
                dropout=config.hidden_dropout_prob
            )
            for _ in range(config.num_hidden_layers)
        ])

        # pooler
        x = torch.zeros((num_classes, config.hidden_size))
        bound = 1.0 / config.hidden_size ** 0.5
        x.uniform_(-bound, bound)
        self.Q = nn.Parameter(x)  # [C, D]
        self.pooler = CrossAttentionPooler(config.hidden_size)
        self.num_classes = num_classes

        # output layer
        self.fc = nn.Linear(config.hidden_size, 1)

        self.window = window

    def forward(self, x, mask, **kwargs):
        """
        x - [N, T, D]
        mask - [N, T]
        """
        # input dropout
        x = self.dropout(x)

        # input proj
        if self.fc_in is not None:
            x = self.fc_in(x)

        # pos emb
        if self.pos_emb is not None:
            x = x + self.pos_emb[:x.shape[1]][None]  # [N, T, D] + [1, T, D]

        # transformer
        band_mask = self._get_mask(x.shape[1]).to(mask.device)  # [T, T]
        merged_mask = torch.logical_and(mask[:, :, None].bool(), band_mask[None])
        extended_attention_mask = (1.0 - merged_mask[:, None].to(dtype=x.dtype)) * -10000.0  # [N, 1, T, T]
        for layer in self.layers:
            x = layer(x, extended_attention_mask)  # [N, T, D]

        # pooler, fc
        q = self.Q[torch.tile(torch.arange(self.num_classes, device=x.device)[None], [x.shape[0], 1])]  # [N, C, D]
        x = self.pooler(q, x, mask)  # [N, C, D]
        x = self.fc(x).squeeze(-1)  # [N, C, 1] -> [N, C]
        return x

    def _get_mask(self, n):
        xs = torch.arange(n)
        ys = torch.arange(n)
        diff = xs[None] - ys[:, None]
        k = self.window // 2 if self.window % 2 == 0 else self.window // 2 + 1
        mask = diff.abs() < k
        # все токены всегда аттендятся на первый; первый токен всегда аттендится на все
        mask[0, :] = True
        mask[:, 0] = True
        return mask


class TransformerEncoderCustomV11(nn.Module):
    """
    v5 + cross attention pooler (mha2). отличие от v9:
    * нет кросс-аттеншена (таг, таг)
    * нет ff в декодере
    к сожалению, это всё равно не лучше, чем v5
    """
    def __init__(
            self,
            config: BertConfig,
            input_dim: int,
            num_classes: int,
            position_embedding_type: str = None,
            window: int = 16,
            add_pos_emb: bool = False
    ):
        super().__init__()

        # input dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # input projection
        self.fc_in = None
        if config.hidden_size != input_dim:
            self.fc_in = nn.Linear(input_dim, config.hidden_size)

        self.pos_emb = None
        if add_pos_emb:
            t_max = 512  # TODO: в конфиг
            x = torch.zeros((t_max, config.hidden_size))
            bound = 1.0 / config.hidden_size ** 0.5
            x.uniform_(-bound, bound)
            self.pos_emb = nn.Parameter(x)  # [T_max, D]

        # transformer
        self.layers = nn.ModuleList([
            TransformerLayerPreLN(
                mha=BertSelfAttention(config, position_embedding_type),
                d=config.hidden_size,
                intermediate_size=config.intermediate_size,
                act="gelu",
                dropout=config.hidden_dropout_prob
            )
            for _ in range(config.num_hidden_layers)
        ])

        # pooler
        x = torch.zeros((num_classes, config.hidden_size))
        bound = 1.0 / config.hidden_size ** 0.5
        x.uniform_(-bound, bound)
        self.Q = nn.Parameter(x)  # [C, D]
        self.pooler = MHA2(num_heads=config.num_attention_heads, head_dim=config.hidden_size // config.num_attention_heads)
        self.num_classes = num_classes

        # output layer
        self.fc = nn.Linear(config.hidden_size, 1)

        self.window = window

    def forward(self, x, mask, **kwargs):
        """
        x - [N, T, D]
        mask - [N, T]
        """
        # input dropout
        x = self.dropout(x)

        # input proj
        if self.fc_in is not None:
            x = self.fc_in(x)

        # pos emb
        if self.pos_emb is not None:
            x = x + self.pos_emb[:x.shape[1]][None]  # [N, T, D] + [1, T, D]

        # transformer
        band_mask = self._get_mask(x.shape[1]).to(mask.device)  # [T, T]
        merged_mask = torch.logical_and(mask[:, :, None].bool(), band_mask[None])
        extended_attention_mask = (1.0 - merged_mask[:, None].to(dtype=x.dtype)) * -10000.0  # [N, 1, T, T]
        for layer in self.layers:
            x = layer(x, extended_attention_mask)  # [N, T, D]

        # pooler, fc
        q = self.Q[torch.tile(torch.arange(self.num_classes, device=x.device)[None], [x.shape[0], 1])]  # [N, C, D]
        cross_attn_mask = (1.0 - mask[:, None, None, :].to(dtype=x.dtype)) * -10000.0  # [N, 1, 1, T]
        x = self.pooler(q, x, x, cross_attn_mask)  # [N, C, D]
        x = self.fc(x).squeeze(-1)  # [N, C, 1] -> [N, C]
        return x

    def _get_mask(self, n):
        xs = torch.arange(n)
        ys = torch.arange(n)
        diff = xs[None] - ys[:, None]
        k = self.window // 2 if self.window % 2 == 0 else self.window // 2 + 1
        mask = diff.abs() < k
        # все токены всегда аттендятся на первый; первый токен всегда аттендится на все
        mask[0, :] = True
        mask[:, 0] = True
        return mask


class TransformerEncoderCustomV12(nn.Module):
    """
    v11 + попытка ускорить сходимость
    сработало: так рили гораздо быстрее сходится, чем v11
    """
    def __init__(
            self,
            config: BertConfig,
            input_dim: int,
            num_classes: int,
            position_embedding_type: str = None,
            window: int = 16,
            add_pos_emb: bool = False
    ):
        super().__init__()

        # input dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # input projection
        self.fc_in = None
        if config.hidden_size != input_dim:
            self.fc_in = nn.Linear(input_dim, config.hidden_size)

        self.pos_emb = None
        if add_pos_emb:
            t_max = 512  # TODO: в конфиг
            x = torch.zeros((t_max, config.hidden_size))
            bound = 1.0 / config.hidden_size ** 0.5
            x.uniform_(-bound, bound)
            self.pos_emb = nn.Parameter(x)  # [T_max, D]

        # transformer
        self.layers = nn.ModuleList([
            TransformerLayerPreLN(
                mha=BertSelfAttention(config, position_embedding_type),
                d=config.hidden_size,
                intermediate_size=config.intermediate_size,
                act="gelu",
                dropout=config.hidden_dropout_prob
            )
            for _ in range(config.num_hidden_layers)
        ])

        # pooler
        x = torch.zeros((num_classes, config.hidden_size))
        bound = 1.0 / config.hidden_size ** 0.5
        x.uniform_(-bound, bound)
        self.Q = nn.Parameter(x)  # [C, D]
        self.pooler = MHA2(num_heads=config.num_attention_heads, head_dim=config.hidden_size // config.num_attention_heads)
        self.num_classes = num_classes
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.ln3 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

        # output layer
        self.fc = nn.Linear(config.hidden_size, 1)

        self.window = window

    def forward(self, x, mask, **kwargs):
        """
        x - [N, T, D]
        mask - [N, T]
        """
        # input dropout
        x = self.dropout(x)

        # input proj
        if self.fc_in is not None:
            x = self.fc_in(x)

        # pos emb
        if self.pos_emb is not None:
            x = x + self.pos_emb[:x.shape[1]][None]  # [N, T, D] + [1, T, D]

        # transformer
        band_mask = self._get_mask(x.shape[1]).to(mask.device)  # [T, T]
        merged_mask = torch.logical_and(mask[:, :, None].bool(), band_mask[None])
        extended_attention_mask = (1.0 - merged_mask[:, None].to(dtype=x.dtype)) * -10000.0  # [N, 1, T, T]
        for layer in self.layers:
            x = layer(x, extended_attention_mask)  # [N, T, D]

        # pooler, fc
        cross_attn_mask = (1.0 - mask[:, None, None, :].to(dtype=x.dtype)) * -10000.0  # [N, 1, 1, T]
        q = self.Q[torch.tile(torch.arange(self.num_classes, device=x.device)[None], [x.shape[0], 1])]  # [N, C, D]
        q = self.dropout1(q)
        x = self.ln1(x)  # нормируем keys
        y = self.ln2(q)  # нормируем queries
        y = self.pooler(y, x, x, cross_attn_mask)  # [N, C, D]
        y = self.dropout2(y)
        x = self.ln3(q + y)
        x = self.fc(x).squeeze(-1)  # [N, C, 1] -> [N, C]
        return x

    def _get_mask(self, n):
        xs = torch.arange(n)
        ys = torch.arange(n)
        diff = xs[None] - ys[:, None]
        k = self.window // 2 if self.window % 2 == 0 else self.window // 2 + 1
        mask = diff.abs() < k
        # все токены всегда аттендятся на первый; первый токен всегда аттендится на все
        mask[0, :] = True
        mask[:, 0] = True
        return mask


# layers


class TransformerLayer(nn.Module):
    """поддержка любой реализации селф аттеншена"""
    def __init__(self, mha, d, intermediate_size, act, dropout):
        super().__init__()
        self.mha = mha
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = IntermediateFeedForward(d, intermediate_size, activation=act)

    def forward(self, x, mask):
        raise NotImplementedError


class TransformerLayerPostLN(TransformerLayer):
    def forward(self, x, mask, **kwargs):
        y = self.mha(x, mask, **kwargs)[0]
        y = self.dropout1(y)
        x = x + y
        x = self.ln1(x)
        y = self.ffn(x)
        y = self.dropout2(y)
        x = x + y
        x = self.ln2(x)
        return x


class TransformerLayerPreLN(TransformerLayer):
    def forward(self, x, mask, **kwargs):
        y = self.ln1(x)
        y = self.mha(y, mask, **kwargs)[0]
        y = self.dropout1(y)
        x = x + y
        y = self.ln2(x)
        y = self.ffn(y)
        y = self.dropout2(y)
        x = x + y
        return x


class DecoderPreLN(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_heads, intermediate_size, vocab_size, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            DecoderLayerPleLN(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                intermediate_size=intermediate_size,
                activation="gelu"
            ) for _ in range(num_layers)
        ])

    def forward(self, input_ids, encoder_state, self_attn_mask, cross_attn_mask):
        x = self.emb(input_ids)
        for layer in self.layers:
            x = layer(x, encoder_state, self_attn_mask, cross_attn_mask)
        return x


class DecoderPreLN2(nn.Module):
    """
    - кросс-аттеншн (токены, лейблы)
    - входной дропаут
    """
    def __init__(self, num_layers, hidden_dim, num_heads, vocab_size, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderLayerPleLN2(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            ) for _ in range(num_layers)
        ])

    def forward(self, input_ids, encoder_state, cross_attn_mask):
        x = self.emb(input_ids)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_state, cross_attn_mask)
        return x


class DecoderLayerPleLN(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, intermediate_size, activation="gelu"):
        super().__init__()
        self.mha1 = MHA2(num_heads=num_heads, head_dim=hidden_dim // num_heads)
        self.mha2 = MHA2(num_heads=num_heads, head_dim=hidden_dim // num_heads)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.ffn = IntermediateFeedForward(hidden_dim, intermediate_size, activation=activation)

    def forward(self, q, enc, self_attn_mask, cross_attn_mask):
        q = self.ln1(q)
        x = self.mha1(q, q, q, self_attn_mask)
        x = self.dropout1(x)
        q = self.ln2(q + x)
        x = self.mha2(q, enc, enc, cross_attn_mask)
        x = self.dropout2(x)
        q = self.ln3(q + x)
        x = self.ffn(q)
        x = self.dropout3(x)
        q = q + x
        return q


class DecoderLayerPleLN2(nn.Module):
    """
    просто крос-аттенш (токены, лейблы)
    """
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.mha = MHA2(num_heads=num_heads, head_dim=hidden_dim // num_heads)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, enc, cross_attn_mask):
        x = self.ln(q)
        x = self.mha(x, enc, enc, cross_attn_mask)
        x = self.dropout(x)
        return x


class IntermediateFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, activation="gelu"):
        super().__init__()
        name2act = {
            "relu": nn.ReLU,
            "gelu": nn.GELU
        }
        output_dim = output_dim if output_dim is not None else input_dim
        self.dense_in = nn.Linear(input_dim, hidden_dim)
        self.activation = name2act[activation]()
        self.dense_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.dense_in(x)
        x = self.activation(x)
        x = self.dense_out(x)
        return x


class MHA(nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        d_model = num_heads * head_dim
        self.dense_input = nn.Linear(d_model, d_model * 3)

    def forward(self, x, mask=None):
        """
        https://arxiv.org/abs/1706.03762
        D = num_heads * head_dim
        :param x: tf.Tensor of shape [N, T, D]
        :param mask: tf.Tensor of shape [N, 1, T, T]. Ones at valid positions
        :return: tf.Tensor of shape [N, T, D]
        """
        batch_size = x.shape[0]
        qkv = self.dense_input(x)  # [N, T, H * D * 3]
        qkv = torch.reshape(qkv, [batch_size, -1, self.num_heads, self.head_dim, 3])  # [N, T, H, D, 3]
        qkv = torch.permute(qkv, [4, 0, 2, 1, 3])  # [3, N, H, T, D]
        q, k, v = torch.unbind(qkv, dim=0)  # 3 * [N, H, T, D]

        k = k.permute(0, 1, 3, 2)  # [N, H, D, T]
        logits = torch.matmul(q, k)  # [N, H, T, T]
        logits /= self.head_dim ** 0.5  # [N, H, T, T]

        if mask is not None:
            if len(mask.shape) == 2:
                extended_mask = mask[:, None, None, :]
            elif len(mask.shape) == 4:
                extended_mask = mask
            else:
                raise
            logits += (1.0 - extended_mask.float()) * -10000.0

        w = torch.softmax(logits, dim=-1)  # [N, H, T, T] (k-axis)
        x = torch.matmul(w, v)  # [N, H, T, D]
        x = torch.permute(x, [0, 2, 1, 3])  # [N, T, H, D]
        x = torch.reshape(x, [batch_size, -1, self.num_heads * self.head_dim])  # [N, T, D * H]
        return x


class MHA2(nn.Module):
    """
    поддержка разных q, k, v
    """
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        d_model = num_heads * head_dim
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, q, k, v, mask=None):
        """
        q - [N, Tq, D]
        k - [N, Tk, D]
        v - [N, Tk, D]
        mask - additive mask broadcastable to [N, H, Tq, Tk]
        """
        batch_size = q.shape[0]
        q = self.wq(q).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [N, H, Tq, d]
        k = self.wk(k).reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)  # [N, H, d, Tk]
        v = self.wv(v).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [N, H, Tk, d]

        logits = torch.matmul(q, k) / self.head_dim ** 0.5  # [N, H, Tq, Tk]
        if mask is not None:
            logits = logits + mask
        w = torch.softmax(logits, dim=-1)  # [N, H, Tq, Tk]

        # [N, H, Tq, d] -> [N, Tq, H, d] -> [N, Tq, D]
        x = torch.matmul(w, v).transpose(1, 2).reshape(batch_size, -1, self.d_model)
        return x


class AttentionPoolerV1(nn.Module):
    """
    самая простая реализация
    веса токенов одни и те же для всех лейблов.
    по сути только помогает игнорить какой-то общий слабый сигнал для всех классов: например, тишина.
    """
    def __init__(self, d):
        super().__init__()
        self.pooler = nn.Linear(d, 1)

    def forward(self, x, mask):
        d = x.shape[-1]
        w = self.pooler(x) / d ** 0.5  # [N, T, 1]
        w = w + (1.0 - mask[:, :, None].float()) * -10000.0
        w = torch.softmax(w, dim=1)  # [N, T, 1]
        x = (x * w).sum(1)  # [N, D]
        return x


class AttentionPoolerV2(nn.Module):
    """
    v1 + только в качестве mlp взята линейная модель с нелинейностью
    """
    def __init__(self, d):
        super().__init__()
        self.pooler = nn.Sequential(
            nn.Linear(d, d),
            nn.Tanh(),
            nn.Linear(d, 1)
        )

    def forward(self, x, mask):
        w = self.pooler(x)  # [N, T, 1]
        w = w + (1.0 - mask[:, :, None].float()) * -10000.0
        w = torch.softmax(w, dim=1)  # [N, T, 1]
        x = (x * w).sum(1)  # [N, D]
        return x


class MultiHeadAdditiveAttention(nn.Module):
    """
    для каждого класса учится своя весовая функция
    """
    def __init__(self, d, num_heads):
        """
        num_heads предполагается равным num_classes
        """
        super().__init__()
        self.fc = nn.Linear(d, num_heads)

    def forward(self, x, mask):
        d = x.shape[-1]
        w = self.fc(x) / d ** 0.5  # [N, T, H]
        w = w + (1.0 - mask[:, :, None].float()) * -10000.0
        w = torch.softmax(w, dim=1)  # [N, T, H]
        x = x.unsqueeze(-2) * w.unsqueeze(-1)  # [N, T, H, D]
        x = x.sum(1)  # [N, H, D]
        return x


class CrossAttentionPooler(nn.Module):
    """
    скоринговая функция одна и является функцией от контекста и эмбеддинга класса
    score(token) = f(tokens, query), где query - эмбеддинг класса

    https://arxiv.org/pdf/1409.0473.pdf
    """
    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, d)  # query projector
        self.U = nn.Linear(d, d)  # key projector
        self.V = nn.Linear(d, 1)  # scorer

    def forward(self, q, k, mask):
        """
        q - [N, Tq, D],
        k - [N, Tk, D]
        mask - [N, Tk] - нужно маскировать только keys
        return: [N, Tq, D]
        """
        # ([N, Tq, 1, D] + [N, 1, Tk, D]) x [D. 1] = [N, Tq, Tk, D] x [D, 1] = [N, Tq, Tk, 1]
        w = self.V(torch.tanh(self.W(q.unsqueeze(2)) + self.U(k.unsqueeze(1)))).squeeze(-1)  # [N, Tq, Tk]
        w = w + (1.0 - mask[:, None].float()) * -10000.0  # masking
        w = torch.softmax(w, dim=1)  # [N, Tq, Tk]

        # [N, 1, Tk, D] * [N, Tq, Tk, 1] = [N, Tq, Tk, D] --sum(2)--> [N, Tq, D]
        # Tq поэлементных произведений [N, Tk, D] (входные токены) * [N, Tk, 1] (alignment веса j-ого классификатора)
        x = (k.unsqueeze(1) * w.unsqueeze(-1)).sum(2)
        return x


class ChannelDropout1d(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        out = x
        out = out.unsqueeze(-1)
        out = self.dropout(out)
        out = out.squeeze(-1)
        return out


### resnet


class ResNetBaseline(nn.Module):
    """
    https://arxiv.org/abs/1909.04939
    """
    def __init__(
            self,
            conv_block_version: int = 1,
            in_channels: int = 768,
            mid_channels: int = 64,
            num_classes: int = 256,
            kernels: str = "8,5,3",
            conv_dropout: float = 0.0,
            block_dropout: float = 0.0,
            add_channel_dropout: bool = False,  # для обратной совместимости
            p_channel_dropout: float = 0.1
    ):
        super().__init__()
        if conv_block_version == 1:
            conv_block_cls = ConvBlock
        elif conv_block_version == 2:
            conv_block_cls = ConvBlockV2
        else:
            raise

        # channel dropout
        self.channel_dropout = None
        if add_channel_dropout:
            self.channel_dropout = ChannelDropout1d(p_channel_dropout)

        # if statement for backward compatibility
        if block_dropout == 0:
            self.layers = nn.Sequential(
                ResNetBlock(conv_block_cls, in_channels=in_channels, out_channels=mid_channels, kernels=kernels, dropout=conv_dropout),
                ResNetBlock(conv_block_cls, in_channels=mid_channels, out_channels=mid_channels * 2, kernels=kernels, dropout=conv_dropout),
                ResNetBlock(conv_block_cls, in_channels=mid_channels * 2, out_channels=mid_channels * 2, kernels=kernels, dropout=conv_dropout)
            )
        else:
            self.layers = nn.Sequential(
                nn.Dropout(block_dropout),
                ResNetBlock(conv_block_cls, in_channels=in_channels, out_channels=mid_channels, kernels=kernels, dropout=conv_dropout),
                nn.Dropout(block_dropout),
                ResNetBlock(conv_block_cls, in_channels=mid_channels, out_channels=mid_channels * 2, kernels=kernels, dropout=conv_dropout),
                nn.Dropout(block_dropout),
                ResNetBlock(conv_block_cls, in_channels=mid_channels * 2, out_channels=mid_channels * 2, kernels=kernels, dropout=conv_dropout)
            )
        self.pooler = AttentionPoolerV1(mid_channels * 2)
        self.fc = nn.Linear(mid_channels * 2, num_classes)

    def forward(self, x, mask, **kwargs):
        """
        x - [N, T, D]
        """
        if self.channel_dropout is not None:
            x = self.channel_dropout(x)
        x = x.transpose(1, 2)  # [N, D, T]
        x = self.layers(x)  # [N, mid_channels * 2, T]
        x = x.transpose(1, 2)  # [N, T, mid_channels * 2]
        x = self.pooler(x, mask)
        x = self.fc(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, conv_block, in_channels: int, out_channels: int, kernels: str = "8,5,3", dropout: float = 0.0):
        super().__init__()

        if isinstance(kernels, str):
            if "," in kernels:
                sep = ","
            elif "-" in kernels:
                sep = "-"
            else:
                raise
            kernels = [int(x) for x in kernels.split(sep)]
        else:
            # may be ListConfig also
            # assert isinstance(kernels, (list, tuple)), type(kernels)
            kernels = [int(x) for x in kernels]
        channels = [in_channels] + [out_channels] * len(kernels)

        self.layers = nn.Sequential(*[
            conv_block(in_channels=channels[i], out_channels=channels[i + 1],
                      kernel_size=kernels[i], stride=1, dropout=dropout) for i in range(len(kernels))
        ])

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding="same"),
                nn.BatchNorm1d(num_features=out_channels)
            ])

    def forward(self, x):  # type: ignore
        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, dropout: float = 0.0):
        super().__init__()
        # if statement for backward compatibility
        if dropout == 0:
            self.layers = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding="same"
                ),
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU()
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding="same"
                ),
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

    def forward(self, x):
        return self.layers(x)


class ConvBlockV2(nn.Module):
    """eca"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size,),
            stride=(stride,),
            padding="same"
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size,),
            stride=(stride,),
            padding="same"
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.eca = eca_layer(out_channels, kernel_size)

    def forward(self, x):
        """
        x: [N, D, T]
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        return out


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=(k_size,), padding="same", bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)  # [N, C, 1]

        # Two different branches of ECA module
        # [N, C, 1] -transpose(-1, -2)-> [N, 1, C] -conv-> [N, 1, C] -transpose(-1, -2)-> [N, C, 1]
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


### inception


class InceptionModel(nn.Module):
    """A PyTorch implementation of the InceptionTime model.
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    num_blocks:
        The number of inception blocks to use. One inception block consists
        of 3 convolutional layers, (optionally) a bottleneck and (optionally) a residual
        connector
    in_channels:
        The number of input channels (i.e. input.shape[-1])
    out_channels:
        The number of "hidden channels" to use. Can be a list (for each block) or an
        int, in which case the same value will be applied to each block
    bottleneck_channels:
        The number of channels to use for the bottleneck. Can be list or int. If 0, no
        bottleneck is applied
    kernel_sizes:
        The size of the kernels to use for each inception block. Within each block, each
        of the 3 convolutional layers will have kernel size
        `[kernel_size // (2 ** i) for i in range(3)]`
    num_classes:
        The number of output classes
    """

    def __init__(
            self,
            num_blocks: int = 3,
            in_channels: int = 768,
            out_channels: int = 64,
            bottleneck_channels: int = 0,
            kernel_sizes: tuple = (31, 17, 9),
            use_residuals: bool = True,
            num_classes: int = 256
    ):
        super().__init__()
        channels = [in_channels] + self._expand_to_blocks(out_channels, num_blocks)
        bottleneck_channels = self._expand_to_blocks(bottleneck_channels, num_blocks)
        if isinstance(kernel_sizes, str):
            if "," in kernel_sizes:
                sep = ","
            elif "-" in kernel_sizes:
                sep = "-"
            else:
                raise
            kernel_sizes = [int(x) for x in kernel_sizes.split(sep)]
        else:
            kernel_sizes = list(kernel_sizes)
        kernel_sizes = self._expand_to_blocks(kernel_sizes, num_blocks)
        if use_residuals == 'default':
            use_residuals = [True if i % 3 == 2 else False for i in range(num_blocks)]
        use_residuals = self._expand_to_blocks(use_residuals, num_blocks)

        self.layers = nn.Sequential(*[
            InceptionBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                residual=use_residuals[i],
                bottleneck_channels=bottleneck_channels[i],
                kernel_size=kernel_sizes[i]) for i in range(num_blocks)
        ])
        self.pooler = AttentionPoolerV1(channels[-1])
        self.fc = nn.Linear(channels[-1], num_classes)

    @staticmethod
    def _expand_to_blocks(value, num_blocks: int):
        if isinstance(value, (list, tuple)):
            assert len(value) == num_blocks, \
                f'Length of inputs lists must be the same as num blocks, ' \
                f'expected length {num_blocks}, got {len(value)}'
        else:
            value = [value] * num_blocks
        return value

    def forward(self, x, mask, **kwargs):
        x = x.transpose(1, 2)  # [N, D, T]
        x = self.layers(x)  # [N, mid_channels * 2, T]
        x = x.transpose(1, 2)  # [N, T, mid_channels * 2]
        x = self.pooler(x, mask)
        x = self.fc(x)
        return x


class InceptionBlock(nn.Module):
    """An inception block consists of an (optional) bottleneck, followed
    by 3 conv1d layers. Optionally residual
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            residual: bool,
            stride: int = 1,
            bottleneck_channels: int = 32,
            kernel_size: int = 41
    ) -> None:
        assert kernel_size > 3, "Kernel size must be strictly greater than 3"
        super().__init__()

        self.use_bottleneck = bottleneck_channels > 0
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(
                in_channels,
                bottleneck_channels,
                kernel_size=1,
                bias=False,
                padding="same"
            )
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        start_channels = bottleneck_channels if self.use_bottleneck else in_channels
        channels = [start_channels] + [out_channels] * 3
        self.conv_layers = nn.Sequential(*[
            nn.Conv1d(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel_size_s[i],
                stride=stride,
                bias=False,
                padding="same"
            )
            for i in range(len(kernel_size_s))
        ])

        self.batchnorm = nn.BatchNorm1d(num_features=channels[-1])
        self.relu = nn.ReLU()

        self.use_residual = residual
        if residual:
            self.residual = nn.Sequential(*[
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        org_x = x
        if self.use_bottleneck:
            x = self.bottleneck(x)
        x = self.conv_layers(x)

        if self.use_residual:
            x = x + self.residual(org_x)
        return x


### rnn


class RNNModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, cell_type="gru", num_layers=2, num_classes=256, dropout=0.3):
        super().__init__()
        # input dropout
        self.dropout = nn.Dropout(dropout)

        # rnn
        if cell_type == "lstm":
            rnn_cls = nn.LSTM
        elif cell_type == "gru":
            rnn_cls = nn.GRU
        else:
            raise NotImplementedError(f"invalid cell_type: {cell_type}")

        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

        self.pooler = mean_pool
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

        self.cell_type = cell_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, mask, **kwargs):
        lengths = mask.sum(1)  # [N]
        packed_seq = nn.utils.rnn.pack_padded_sequence(x, lengths.to("cpu"), batch_first=True, enforce_sorted=False)
        h = self._init_hidden(x.shape[0], x.device)
        packed_seq, _ = self.rnn(packed_seq, h)
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_seq, batch_first=True, padding_value=0.0)
        x = self.pooler(x, mask)
        x = self.fc(x)
        return x

    def _init_hidden(self, batch_size, device):
        # 2 - потому bidirectional
        if self.cell_type == "lstm":
            c0 = torch.zeros((2 * self.num_layers, batch_size, self.hidden_dim), device=device)
            h0 = torch.zeros((2 * self.num_layers, batch_size, self.hidden_dim), device=device)
            return c0, h0
        elif self.cell_type == "gru":
            h = torch.zeros((2 * self.num_layers, batch_size, self.hidden_dim), device=device)
            return h
