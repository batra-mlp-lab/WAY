import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import copy
from loader import Loader


class RNN(nn.Module):
    def __init__(
        self,
        input_size,
        embed_size,
        hidden_size,
        num_layers,
        dropout,
        bidirectional,
        embedding_type,
        embedding_dir,
    ):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.reduce = "last" if not bidirectional else "mean"
        self.embedding_type = embedding_type
        self.embedding_dir = embedding_dir

        self.embedding = nn.Embedding(input_size, embed_size)
        if self.embedding_type == "glove":
            glove_weights = torch.FloatTensor(
                np.load(
                    self.embedding_dir + "glove_weights_matrix.npy", allow_pickle=True
                )
            )
            self.embedding.from_pretrained(glove_weights)
        elif self.embedding_type == "word2vec":
            w2v_weights = torch.FloatTensor(
                np.load(
                    self.embedding_dir + "w2v_weights_matrix.npy", allow_pickle=True
                )
            )
            self.embedding.from_pretrained(w2v_weights)

        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, seq_lengths):
        embed = self.embedding(x)
        embed = self.dropout(embed)
        embed_packed = pack_padded_sequence(
            embed, seq_lengths, enforce_sorted=False, batch_first=True
        )

        out_packed = embed_packed
        self.lstm.flatten_parameters()
        out_packed, _ = self.lstm(out_packed)
        out, _ = pad_packed_sequence(out_packed)

        # reduce the dimension
        if self.reduce == "last":
            out = out[seq_lengths - 1, np.arange(len(seq_lengths)), :]
        elif self.reduce == "mean":
            seq_lengths_ = seq_lengths.unsqueeze(-1)
            out = torch.sum(out[:, np.arange(len(seq_lengths_)), :], 0) / seq_lengths_

        return out
