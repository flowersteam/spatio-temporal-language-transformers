"""
This module implements the models used for the experiments.
"""
import math
import torch
import torch.nn as nn
from functools import reduce

from src.models.shared import Feedforward

### Helper fns

prod = lambda x, y: x * y
prod_reduce = lambda l: reduce(prod, l, 1)

### Helper modules

class PositionalEncoding(nn.Module):
    """
    Positional encoding matrix based on sinusoids.

    From pytorch language modeling example here:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000, device=torch.device('cpu')):

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(
                0, d_model, 2, device=device
            ).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is [T, B, (Nobj), H]
        if len(x.shape) == 4: # adaptation for tensors with 4 dimensions
            pe = self.pe.unsqueeze(1)
        else:
            pe = self.pe
        x = x + pe[:x.size(0)]
        return self.dropout(x)

class TransformerModule(nn.Module):
    """
    Wrapper around the TransformerEncoderLayer and TransformerEncoder.
    Also allows for easier processing of 4+ dimensional tensors; the first
    dimension in considered as the obj dim on which to perform self-attention;
    the middle dims (at least 1) are considered as batch dims and have no effect,
    the last dim is considered as the feature dim.
    """
    def __init__(self, hidden_size, num_heads, num_layers, dropout=0.1):

        super().__init__()

        tfm_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_heads,
            hidden_size,
            dropout=dropout,
        )
        layernorm = nn.LayerNorm(hidden_size)
        self.tfm = nn.TransformerEncoder(
            tfm_layer,
            num_layers,
            layernorm,
        )

    def forward(self, x):
        if len(x.shape) == 3:
            return self.tfm(x)
        elif len(x.shape) > 3:
            shape = list(x.shape)
            mid_shape = shape[1:-1]
            mid_size = prod_reduce(mid_shape)
            x = x.reshape(shape[0], mid_size, shape[-1])
            x = self.tfm(x)
            x = x.reshape(shape[:1] + mid_shape + shape[-1:])
            return x
        else:
            raise ValueError('Invalid shape for input')

class Transformer_UT(nn.Module):
    """
    This model concatenates all the tokens corresponding to temporal traces of
    objects and words in the sentence, and performs num_layers of self-attention
    over all tokens. A learned query vector is used to query the transformed
    tensors.

    Parameters:
        - body_size: number of body features;
        - obj_size: number of object features;
        - voc_size: size of the vocabulary used in descriptions of the scene
        - seq_length: maximum size of a sentence;
        - hidden_size: size of: the output of the core and object encoding
            layers, the hidden layers of both Transformers;
        - num_heads: number of attention heads in both Transformers;
        - num_layers: number of layers in both Transformers;
        - dropout: dropout used in the Transformers, defaults to 0.1;
        - device: device on which to load the model;
        - word_aggreg: whether to use word_aggregation, if True the model becomes
            UT-WA
    """
    def __init__(self, body_size, obj_size, voc_size, seq_length,
                hidden_size, num_heads, num_layers,
                dropout=0.1, device=torch.device('cpu'), word_aggreg=False, **kwargs):

        super().__init__()

        self.body_size = body_size
        self.obj_size = obj_size
        self.voc_size = voc_size
        self.seq_length = seq_length
        self.ff_size = hidden_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.word_aggreg = word_aggreg

        self.fc_cast_body = Feedforward(self.body_size, [self.ff_size, self.hidden_size])
        self.fc_cast_obj = Feedforward(self.obj_size, [self.ff_size, self.hidden_size])
        self.fc_cast_words = Feedforward(self.voc_size, [self.ff_size, self.hidden_size])

        # init cls token
        self.query = nn.Parameter(torch.zeros(self.hidden_size))

        # self.query_token = torch.zeros(self.hidden_size, device=device)
        self.sensory_oh = torch.tensor([1., 0.], device=device)
        self.linguistic_oh = torch.tensor([0., 1.], device=device)

        self.dim_adjust_proj = nn.Linear(self.hidden_size + 2, self.hidden_size)

        self.fc_out = Feedforward(self.hidden_size, [self.ff_size, 1])

        self.transformer = TransformerModule(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        if self.word_aggreg:
            self.word_transformer = TransformerModule(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            )

        self.pe = PositionalEncoding(self.hidden_size, device=device, dropout=dropout)

        self.device = device
        self.to(self.device)

    def forward(self, state_trace, body_trace, description):

        state_trace = self.fc_cast_obj(state_trace) # [B, T, Nobj, h]
        body_trace = self.fc_cast_body(body_trace) # [B, T, 1, h]
        words = self.fc_cast_words(description) # [B, Seq, h]
        query_token = self.query

        B, T, Nobj, h = state_trace.shape
        _, S, _ = words.shape

        state_trace = state_trace.permute(1, 2, 0, 3) # [T, Nobj, B, h]
        body_trace = body_trace.permute(1, 2, 0, 3) # [T, 1, B, h]
        words = words.transpose(0, 1) # [S, B, h]

        # positional encoding
        state_trace = self.pe(state_trace)
        body_trace = self.pe(body_trace)
        words = self.pe(words)

        state_trace = state_trace.reshape(T * Nobj, B, h)
        body_trace = body_trace.reshape(T, B, h)
        sensory_trace = torch.cat([state_trace, body_trace], 0)

        # add one hot identifiers and reproject to hidden dim
        sensory_trace = torch.cat([
            sensory_trace,
            self.sensory_oh.expand(T * (Nobj + 1), B, 2)
        ], -1)
        words = torch.cat([
            words,
            self.linguistic_oh.expand(S, B, 2)
        ], -1)
        sensory_trace = self.dim_adjust_proj(sensory_trace)
        words = self.dim_adjust_proj(words)

        if self.word_aggreg:
            words = torch.cat([words, query_token.expand(1, B, h)], 0)
            words = self.word_transformer(words)[-1:]

        tfm_input = torch.cat([
            sensory_trace,
            words,
            query_token.expand(1, B, h) # query is last token
        ], dim=0)

        tfm_output = self.transformer(tfm_input)

        return torch.sigmoid(self.fc_out(tfm_output[-1]))

class SpatialFirstTransformer(nn.Module):
    """
    This model performs a first round of self-attention in individual frames/timesteps,
    using linguistic information, then performs self-attention over the temporal
    dimension.

    Parameters:
        - body_size: number of body features;
        - obj_size: number of object features;
        - voc_size: size of the vocabulary used in descriptions of the scene
        - seq_length: maximum size of a sentence;
        - hidden_size: size of: the output of the core and object encoding
            layers, the hidden layers of both Transformers;
        - num_heads: number of attention heads in both Transformers;
        - num_layers: number of layers in both Transformers;
        - dropout: dropout used in the Transformers, defaults to 0.1;
        - device: device on which to load the model;
        - word_aggreg: whether to use word_aggregation, if True the model becomes
            UT-WA
    """

    def __init__(self, body_size, obj_size, voc_size, seq_length,
                hidden_size, num_heads, num_layers,
                dropout=0.1, device=torch.device('cpu'), word_aggreg=False, **kwargs):

        super().__init__()

        self.body_size = body_size
        self.obj_size = obj_size
        self.voc_size = voc_size
        self.seq_length = seq_length
        self.ff_size = hidden_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.word_aggreg = word_aggreg

        self.fc_cast_body = Feedforward(self.body_size, [self.ff_size, self.hidden_size])
        self.fc_cast_obj = Feedforward(self.obj_size, [self.ff_size, self.hidden_size])
        self.fc_cast_words = Feedforward(self.voc_size, [self.ff_size, self.hidden_size])

        # init cls token
        self.query = nn.Parameter(torch.zeros(self.hidden_size))

        # self.query_token = torch.zeros(self.hidden_size, device=device)
        self.sensory_oh = torch.tensor([1., 0.], device=device)
        self.linguistic_oh = torch.tensor([0., 1.], device=device)

        self.dim_adjust_proj = nn.Linear(self.hidden_size + 2, self.hidden_size)

        self.fc_out = Feedforward(self.hidden_size, [self.ff_size, 1])

        self.spatial_transformer = TransformerModule(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.temporal_transformer = TransformerModule(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        if self.word_aggreg:
            self.word_transformer = TransformerModule(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            )

        self.pe = PositionalEncoding(self.hidden_size, device=device, dropout=dropout)

        self.device = device
        self.to(self.device)

    def forward(self, state_trace, body_trace, description):

        state_trace = self.fc_cast_obj(state_trace) # [B, T, Nobj, h]
        body_trace = self.fc_cast_body(body_trace) # [B, T, 1, h]
        words = self.fc_cast_words(description) # [B, Seq, h]
        query_token = self.query

        B, T, Nobj, h = state_trace.shape
        _, S, _ = words.shape

        state_trace = state_trace.permute(1, 2, 0, 3)
        body_trace = body_trace.permute(1, 2, 0, 3)
        words = words.transpose(0, 1)

        # positional encoding
        state_trace = self.pe(state_trace) # [T, Nobj, B, h]
        body_trace = self.pe(body_trace) # [T, 1, B, h]
        words = self.pe(words) # [S, B, h]

        # add sensory and linguistic identifiers and reproject to dim
        state_trace = torch.cat([state_trace, self.sensory_oh.expand(T, Nobj, B, 2)], -1)
        state_trace = self.dim_adjust_proj(state_trace)

        body_trace = torch.cat([body_trace, self.sensory_oh.expand(T, 1, B, 2)], -1)
        body_trace = self.dim_adjust_proj(body_trace)

        words = torch.cat([words, self.linguistic_oh.expand(S, B, 2)], -1)
        words = self.dim_adjust_proj(words)

        # aggreg words if need be
        if self.word_aggreg:
            words = torch.cat([words, query_token.expand(1, B, h)], 0)
            words = self.word_transformer(words)[-1:]
            S = 1

        # make spatial transformer input
        words = words.expand(T, S, B, h)
        query_token_spatial = query_token.expand(T, 1, B, h)
        spatial_tfm_input = torch.cat([
            state_trace,
            body_trace,
            words,
            query_token_spatial
        ], 1)
        spatial_tfm_input.transpose(0, 1) # [Nobj + 1 + S + 1, T, B, h]

        # [T, B, h]
        temporal_tfm_input = self.spatial_transformer(spatial_tfm_input)[-1]
        query_token_temporal = query_token.expand([1, B, h])
        temporal_tfm_input = torch.cat([temporal_tfm_input, query_token_temporal], 0)

        return torch.sigmoid(self.fc_out(self.temporal_transformer(temporal_tfm_input)[-1]))

class TemporalFirstTransformer(nn.Module):
    """
    This model performs a first round of self-attention over individual object
    traces using linguitic information, then performs self attention on the
    resulting object summaries.

    Parameters:
        - body_size: number of body features;
        - obj_size: number of object features;
        - voc_size: size of the vocabulary used in descriptions of the scene
        - seq_length: maximum size of a sentence;
        - hidden_size: size of: the output of the core and object encoding
            layers, the hidden layers of both Transformers;
        - num_heads: number of attention heads in both Transformers;
        - num_layers: number of layers in both Transformers;
        - dropout: dropout used in the Transformers, defaults to 0.1;
        - device: device on which to load the model;
        - word_aggreg: whether to use word_aggregation, if True the model becomes
            UT-WA
    """

    def __init__(self, body_size, obj_size, voc_size, seq_length,
                hidden_size, num_heads, num_layers,
                dropout=0.1, device=torch.device('cpu'), word_aggreg=False, **kwargs):
        super().__init__()

        self.body_size = body_size
        self.obj_size = obj_size
        self.voc_size = voc_size
        self.seq_length = seq_length
        self.ff_size = hidden_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.word_aggreg = word_aggreg

        self.fc_cast_body = Feedforward(self.body_size, [self.ff_size, self.hidden_size])
        self.fc_cast_obj = Feedforward(self.obj_size, [self.ff_size, self.hidden_size])
        self.fc_cast_words = Feedforward(self.voc_size, [self.ff_size, self.hidden_size])

        # init cls token
        self.query = nn.Parameter(torch.zeros(self.hidden_size))

        # self.query_token = torch.zeros(self.hidden_size, device=device)
        self.sensory_oh = torch.tensor([1., 0.], device=device)
        self.linguistic_oh = torch.tensor([0., 1.], device=device)

        self.dim_adjust_proj = nn.Linear(self.hidden_size + 2, self.hidden_size)

        self.fc_out = Feedforward(self.hidden_size, [self.ff_size, 1])

        self.spatial_transformer = TransformerModule(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.temporal_transformer = TransformerModule(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        if self.word_aggreg:
            self.word_transformer = TransformerModule(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            )

        self.pe = PositionalEncoding(self.hidden_size, device=device, dropout=dropout)

        self.device = device
        self.to(self.device)

    def forward(self, state_trace, body_trace, description):

        state_trace = self.fc_cast_obj(state_trace) # [B, T, Nobj, h]
        body_trace = self.fc_cast_body(body_trace) # [B, T, 1, h]
        words = self.fc_cast_words(description) # [B, Seq, h]
        query_token = self.query

        B, T, Nobj, h = state_trace.shape
        _, S, _ = words.shape

        state_trace = state_trace.permute(1, 2, 0, 3)
        body_trace = body_trace.permute(1, 2, 0, 3)
        words = words.transpose(0, 1)

        # positional encoding
        state_trace = self.pe(state_trace) # [T, Nobj, B, h]
        body_trace = self.pe(body_trace) # [T, 1, B, h]
        sensory_trace = torch.cat([state_trace, body_trace], 1) # [T, Nobj+1, B, h]
        words = self.pe(words) # [S, B, h]

         # add sensory and linguistic identifiers and reproject to dim
        sensory_trace = torch.cat([sensory_trace, self.sensory_oh.expand(T, Nobj + 1, B, 2)], -1)
        sensory_trace = self.dim_adjust_proj(sensory_trace)

        words = torch.cat([words, self.linguistic_oh.expand(S, B, 2)], -1)
        words = self.dim_adjust_proj(words)

        # aggreg words if need be
        if self.word_aggreg:
            words = torch.cat([words, query_token.expand(1, B, h)], 0)
            words = self.word_transformer(words)[-1:]
            S = 1

        words = words.expand(Nobj + 1, S, B, h).transpose(0, 1)
        query_token_temporal = query_token.expand(1, Nobj + 1, B, h)
        temporal_tfm_input = torch.cat([
            sensory_trace,
            words,
            query_token_temporal
        ], 0)

        # out is [T, B, h]
        spatial_tfm_input = self.temporal_transformer(temporal_tfm_input)[-1]
        query_token_spatial = query_token.expand([1, B, h])
        spatial_tfm_input = torch.cat([spatial_tfm_input, query_token_spatial], 0)

        return torch.sigmoid(self.fc_out(self.spatial_transformer(spatial_tfm_input)[-1]))