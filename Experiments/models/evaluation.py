import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import os
import random
import preprocess
import model
import embed
import main

import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
MAX_LENGTH = 20
SOS_token = 0
EOS_token = 1


def manual(ixgen, encoder, decoder, sentence, max_length=MAX_LENGTH):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = embed.tensorFromSentence(ixgen, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        decoded_words = list()
        encoder_outputs = torch.zeros(MAX_LENGTH, encoder.nhid, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        for di in range(max_length):
            if args.attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(ixgen.index2word[topi.item()])
            
        decoder_input = topi.squeeze().detach()
        decoded_sentence = ' '.join(decoded_words)
        return decoded_sentence

def cli(ixgen, encoder, decoder, lines, model_save_path):
    ckpt = torch.load(model_save_path)
    encoder.load_state_dict(ckpt['encoder'])
    decoder.load_state_dict(ckpt['decoder'])
    print(encoder)
    print(decoder)
    print('> Enter sentences at the prompt. To quit, type \'QUIT\' \n< The output will be provided as soon as the model computes')
    while True:
        sentence = input('> ')
        if sentence == 'QUIT':
            break
        elif sentence == 'RANDOM':
            evaluateRandomly(ixgen, encoder, decoder, lines, n=10)
        output = manual(ixgen, encoder, decoder, sentence)
        print('< ' + str(output))
        print('')

def evaluateRandomly(ixgen, encoder, decoder, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[1])
        print('=', pair[2])
        output_sentence = manual(ixgen, encoder, decoder, pair[1])
        print('<', output_sentence)
        print('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating edited version from original version')
    parser.add_argument('--data_path', type=str, default='../data/wikiHow_revisions_corpus.txt', help='location of dataset')
    parser.add_argument('--embeddings', type=str, default='./glove.bin', help='path to embedding file')
    parser.add_argument('--attention', action='store_true', help='use attention decoder? Default: True')
    parser.add_argument('--nlayers', type=int, default=2, help='number of hidden layers in the encoder and decoder')
    parser.add_argument('--nhid', type=int, default=300, help='hidden dimension of encoder and decoder')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout for the encode and decoder')
    parser.add_argument('--model_save_path', type=str, default='/tmp/model.ckpt', help='save location for the model for evaluation')
    parser.add_argument('--seed', type=int, default=1111, help='manual seed for reproducability')
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    
    ixgen, lines = preprocess.prepareData(args.data_path, args.embeddings)
    wtmatrix = preprocess.wtMatrix(ixgen) 
    encoder = model.EncoderRNN(ixgen.n_words, args.nhid, args.nlayers, wtmatrix, dropout=args.dropout).to(device)
    if not args.attention:
        decoder = model.DecoderRNN(args.nhid, ixgen.n_words, args.nlayers, dropout=args.dropout).to(device)
    else:
        decoder = model.AttnDecoderRNN(args.nhid, ixgen.n_words, args.nlayers, dropout_p=args.dropout).to(device)

    cli(ixgen, encoder, decoder, lines, args.model_save_path)
