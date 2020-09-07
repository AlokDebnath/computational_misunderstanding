import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import BertTokenizer, BertModel

bert = BertModel.from_pretrained('bert-base-uncased')

class AttentionModel(torch.nn.Module):
    def __init__(self, bert, batch_size, output_size, hidden_size):
        super(AttentionModel, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.word_embeddings = bert
        embedding_length = bert.config.to_dict()['hidden_size']
        
        self.lstmA = nn.LSTM(embedding_length, hidden_size)
        self.lstmB = nn.LSTM(embedding_length, hidden_size)
        self.jointlstm = nn.LSTM(2*hidden_size, hidden_size)
        self.combine = nn.Linear(2*hidden_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    def forward(self, input_sentenceA, input_sentenceB, batch_size=None):
        inputA = self.word_embeddings(input_sentenceA)
        inputB = self.word_embeddings(input_sentenceB)
        inputA = inputA[0].permute(1, 0, 2)
        inputB = inputB[0].permute(1, 0, 2)
        if batch_size is None:
            hA_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
            hA_1 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
            hB_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
            hB_1 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())

            cA_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
            cA_1 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
            cB_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
            cB_1 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())

        else:
            hA_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
            hA_1 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
            hB_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
            hB_1 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())

            cA_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
            cA_1 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
            cB_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
            cB_1 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())

        outputA, (final_hidden_stateA, final_cell_stateA) = self.lstmA(inputA, (hA_0, cA_0))
        outputB, (final_hidden_stateB, final_cell_stateB) = self.lstmB(inputB, (hB_0, cB_0))
        outputAB = torch.cat((outputA, outputB), 2)
        outputAB, (final_hidden_state, final_cell_state) = self.jointlstm(outputAB, (h_0, c_0))
        
        outputA = self.combine(torch.cat((outputA, outputAB), 2))
        outputB = self.combine(torch.cat((outputB, outputAB), 2))
        
        outputA, (final_hidden_state, final_cell_state) = self.lstm(outputA, (hA_1, cA_1))
        outputB, (final_hidden_state, final_cell_state) = self.lstm(outputB, (hB_1, cB_1))

        outputA = outputA.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)
        outputB = outputB.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)
        
        # fin_outputA = self.attention_net(outputA, final_hidden_stateA)
        # fin_outputB = self.attention_net(outputB, final_hidden_stateB)

        logitsA = self.label(outputA)
        logitsB = self.label(outputB)
        return logitsA, logitsB

