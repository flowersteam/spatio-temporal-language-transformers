"""
This module follows the structure of the model module.
It implements the twin transformer reward function: a first
Transformer is applied on each scene (objects + core), a second
one across the different time steps.
"""
import math
import torch
import torch.nn as nn

from src.models.shared import Feedforward


class FlatLSTM(nn.Module):
    """
    Reward function model based on LSTMs.
    The scene is processed in a recurrent fasion, as well as the description.
    Both are concatenated and fed through an MLP to produce the final answer.
    """

    def __init__(self, body_size, obj_size, voc_size, seq_length,
                 hidden_size, num_layers, dropout=0.,
                 device=torch.device('cpu'), **kwargs):

        super().__init__()

        # self.state_size = state_size
        self.body_size = body_size
        self.obj_size = obj_size
        self.state_size = 3 * obj_size + body_size
        self.voc_size = voc_size
        self.seq_length = seq_length
        self.ff_size = hidden_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        # size of the core vector containing body and linguistic information

        self.lstm = nn.LSTM(
            self.voc_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.lstm_scene = nn.LSTM(
            self.state_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc_out = Feedforward(2 * self.hidden_size, [self.ff_size, 1])

        self.device = device
        self.to(self.device)

    def forward(self, objs_trace, body_trace, description):
        """
        Sizes of the inputs:
        state_trace: [B, T, Nobj, Fobj]
        body_trace: [B, T, Nbody, Fbody]
        description: [B, Seq, Voc_size]

        Note: for now we assume no masks and the temporal output is queried at the end
        of the temporal sequence.
        """
        B, T, Nobj, _ = objs_trace.shape
        N = Nobj + 1

        # TODO: mask and temporal index of the current state

        # compute language embedding and cat it to the body trace
        state_trace = torch.cat((
            objs_trace.view([B, T, Nobj * objs_trace.shape[-1]]),
            body_trace.squeeze()
        ), dim=-1)

        h_lstm_seq, _ = self.lstm(description)
        h_lstm = h_lstm_seq[:, -1, :]

        h_scene_seq, _ = self.lstm_scene(state_trace)
        h_scene = h_scene_seq[:, -1, :]

        h_cat = torch.cat([h_scene, h_lstm], dim=1)
        logit = self.fc_out(h_cat)
        return torch.sigmoid(logit)

class FactoredLSTM(nn.Module):
    """
    Reward function model based on LSTMs.
    All objects and the body are processed in a recurrent
    Both are concatenated and fed through an MLP to produce the final answer.
    """

    def __init__(self, body_size, obj_size, voc_size, seq_length,
                 hidden_size, num_layers, dropout=0.,
                 device=torch.device('cpu'), **kwargs):

        super().__init__()

        # self.state_size = state_size
        self.body_size = body_size
        self.obj_size = obj_size
        self.state_size = 3 * obj_size + body_size
        self.voc_size = voc_size
        self.seq_length = seq_length
        self.ff_size = hidden_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers

        self.object_proj = nn.Linear(obj_size, self.hidden_size)
        self.body_proj = nn.Linear(body_size, self.hidden_size)

        self.lstm_lang = nn.LSTM(
            self.voc_size,
            self.hidden_size,
            num_layers=self.num_layers,
            # batch_first=True
        )
        self.lstm_obj = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            num_layers=self.num_layers,
            # batch_first=True
        )
        self.fc_out = Feedforward((1 + 3 + 1) * self.hidden_size, [self.ff_size, 1])

        self.device = device
        self.to(self.device)

    def forward(self, objs_trace, body_trace, description):
        """
        Sizes of the inputs:
        state_trace: [B, T, Nobj, Fobj]
        body_trace: [B, T, Nbody, Fbody]
        description: [B, Seq, Voc_size]

        Note: for now we assume no masks and the temporal output is queried at the end
        of the temporal sequence.
        """
        B, T, Nobj, _ = objs_trace.shape
        N = Nobj + 1
        h = self.hidden_size

        objs_trace = self.object_proj(objs_trace.permute(1, 2, 0, 3)) # [T, Nobj, B, h]
        body_trace = self.body_proj(body_trace.permute(1, 2, 0, 3)) # [T, 1, B, h]
        state_trace = torch.cat([
            objs_trace,
            body_trace,
        ], 1) # [T, Nobj + 1, B, h]
        description = description.transpose(0, 1) # [S, B, h]

        state_trace = state_trace.reshape(T, N * B, h)
        h_state = self.lstm_obj(state_trace)[0][-1].reshape(N, B, h)
        h_state = h_state.transpose(0, 1)

        h_lstm_seq, _ = self.lstm_lang(description)
        h_lstm = h_lstm_seq[-1] # [B, h]

        h_cat = torch.cat([h_state.reshape(B, N * h), h_lstm], dim=-1)
        logit = self.fc_out(h_cat)
        return torch.sigmoid(logit)


if __name__ == "__main__":
    B = 128
    T = 51
    seq = 4
    voc_size = 62

    Fobj = 42
    Nobj = 3
    Fbody = 3
    Nbody = 1
    obj_trace = torch.rand(B, T, Nobj, Fobj)
    body_trace = torch.rand(B, T, Nbody, Fbody)
    description = torch.rand(B, seq, voc_size)
    state_size = Nobj * Fobj + Nbody * Fbody

    reward_fn = FlatLSTM(
        body_size=Fbody,
        obj_size=Fobj,
        voc_size=voc_size,
        seq_length=seq,
        hidden_size=20,
        num_layers=1,
    )
    out = reward_fn(obj_trace, body_trace, description)
