import math
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LENGTH = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EncoderRNN(nn.Module):
    def __init__(self, ninp, nhid, nlayers, dropout=0.2):
        super(EncoderRNN, self).__init__()
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(ninp, nhid)
        self.gru = nn.GRU(nhid, nhid)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for l in range(self.nlayers):
            output, hidden = self.gru(output, hidden)
            output = self.dropout(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.nhid, device=device)

    def getHyperparams(self):
        return self.ninp, self.nhid, self.nlayers, self.wtmatrix

class DecoderRNN(nn.Module):
    def __init__(self, nhid, nout, nlayers, dropout=0.2):
        super(DecoderRNN, self).__init__()
        self.nhid = nhid
        self.nlayers = nlayers
        self.embedding = nn.Embedding(nout, nhid)
        self.gru = nn.GRU(nhid, nhid)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(nhid, nout)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        for l in range(self.nlayers):
            output, hidden = self.gru(output, hidden)
            output = self.dropout(output)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.nhid, device=device)

    def getHyperparams(self):
        return self.ninp, self.nhid, self.nlayers

class AttnDecoderRNN(nn.Module):
    def __init__(self, nhid, nout, nlayers, dropout_p=0.1,max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.nlayers = nlayers
        self.embedding = nn.Embedding(self.nout, self.nhid)
        self.attn = nn.Linear(self.nhid * 2, self.max_length)
        self.attn_combine = nn.Linear(self.nhid * 2, self.nhid)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.nhid, self.nhid)
        self.out = nn.Linear(self.nhid, self.nout)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        for l in range(self.nlayers):
            output, hidden = self.gru(output, hidden)
            output = self.dropout(output)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.nhid, device=device)

    def getHyperparams(self):
        return self.ninp, self.nhid, self.nlayers
