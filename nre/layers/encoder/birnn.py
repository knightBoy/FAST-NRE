"""
    Use BiRNN for instances encoder
"""

import torch
import torch.nn as nn
from nre.utils.gpu_utils import gpu_utils

class BiRNNEncoder(nn.Module):
    def __init__(self, opt):
        super(BiRNNEncoder, self).__init__()

        self.opt = opt
        self.hidden_size = opt.rnn_hidden_size
        self.num_layers = opt.rnn_layer_num
        self.input_size = opt.word_vec_size + 2 * opt.position_size

        assert opt.rnn_type in ['LSTM', 'GRU'], 'Option rnn_type is invalid'
        if opt.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=opt.bidirectional)
        elif opt.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=opt.bidirectional)

        self.dropout = nn.Dropout(opt.dropout_keep)

    def init_hidden(self, batch):
        """
        Initialize the hidden states of rnn
        """

        # Set initial states
        directions = 2 if self.opt.bidirectional else 1
        if self.opt.rnn_type == 'LSTM':
            h0 = gpu_utils.to_cuda(torch.zeros(self.num_layers*directions, batch, self.hidden_size)) # 2 for bidirectional
            c0 = gpu_utils.to_cuda(torch.zeros(self.num_layers*directions, batch, self.hidden_size))
            return (h0, c0)
        else:
            h0 = gpu_utils.to_cuda(torch.randn(self.num_layers*directions, batch, self.hidden_size))
            return h0

    def forward(self, embeddings, input_lens):
        """
        Encode embeddings

        Args:
            embeddings: [batch_size, num_step, embedding_size]
            input_lens: [batch_size, 1]
        Return:
            hidden state of each sentence: [batch_size, 2 * rnn_hidden_size]
        """

        # Sort the sequences
        sort_index = torch.argsort(input_lens, dim=0, descending=True)
        unsort_index = torch.argsort(sort_index, dim=0, descending=False)
        embeddings = torch.index_select(embeddings, 0, sort_index)
        input_lens = torch.index_select(input_lens, 0, sort_index)

        # Pack the sequence
        pack = nn.utils.rnn.pack_padded_sequence(input=embeddings, lengths=input_lens, batch_first=True)

        # Froward propagate
        init_hidden = self.init_hidden(embeddings.size()[0])
        if self.opt.rnn_type == 'LSTM':
            _, (hn, cn) = self.rnn(pack, init_hidden)
        else:
            _, hn = self.rnn(pack, init_hidden)

        """
        unpacked = nn.utils.rnn.pad_packed_sequence(sequence=out, batch_first=True)
        out, bz = unpacked[0], unpacked[1] # out: tensor of shape (batch_size, seq_length, hidden_size*2), bz: lengths of sequences
        batch_index = torch.arange(bz.size()[0], dtype=torch.int64)
        bz = (bz - 1)
        hn = out[batch_index, bz, :]
        """
        
        hn = hn.transpose(0, 1) # shape of (batch_size, direction * layer_num, hidden_size)
        hn = torch.index_select(hn, 0, unsort_index)
        hn = hn.contiguous().view(hn.size()[0], -1)

        return self.dropout(hn)
        