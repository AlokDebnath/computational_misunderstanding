import preprocess
import model
import embed

import os
from tqdm import tqdm
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Predicting edited version from original version')
parser.add_argument('--data_path', type=str, default='../data/wikiHow_revisions_corpus.txt', help='location of dataset')
parser.add_argument('--test_path', type=str, default='../data/test_files.txt', help='path to test split file')
parser.add_argument('--dev_path', type=str, default='../data/dev_files.txt', help='path to dev split file')
parser.add_argument('--embeddings', type=str, default='./glove.bin', help='path to embedding file')
parser.add_argument('--attention', action='store_true', help='use attention decoder? Default: True')
parser.add_argument('--nlayers', type=int, default=2, help='number of hidden layers in the encoder and decoder')
parser.add_argument('--nhid', type=int, default=256, help='hidden dimension of encoder and decoder')
parser.add_argument('--batch_size', type=int, default=10000, help='size of the training data batches. Test set bsz is 0.1*train_bsz')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout for the encode and decoder')
parser.add_argument('--model_save_path', type=str, default='./model.ckpt', help='save location for the model for evaluation')
parser.add_argument('--lr', type=float, default=0.2, help='model learning rate')
parser.add_argument('--log_interval', type=int, default=1000, help='reporting interval')
parser.add_argument('--seed', type=int, default=1111, help='manual seed for reproducability')

args = parser.parse_args()
torch.manual_seed(args.seed)

ixgen, lines = preprocess.prepareData(args.data_path, args.embeddings)
train_data, test_data, dev_data = preprocess.read_train_test_dev(lines, args.test_path, args.dev_path)

eval_batch_size = int(0.1* args.batch_size)
train_batches = preprocess.get_batches(args.batch_size, train_data)
test_batches = preprocess.get_batches(eval_batch_size, test_data)
dev_batches = preprocess.get_batches(eval_batch_size, dev_data)
wtmatrix = preprocess.wtMatrix(ixgen)

MAX_LENGTH = 150

def train(input_tensor, target_tensor, encoder, decoder, encoder_optim, decoder_optim, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.nhid, device=device)

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[preprocess.SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        if args.attention:
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        else:
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == preprocess.EOS_token:
            break
        
    loss.backward()
    encoder_optim.step()
    decoder_optim.step()

    return loss.item() / target_length

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(ixgen, encoder, decoder, train_batches, dev_batches, log_interval=1, learning_rate=0.2, model_save_path='./model.ckpt'):
    print('Starting training...')
    criterion = nn.NLLLoss()
    encoder_optim = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optim = optim.SGD(encoder.parameters(), lr=learning_rate)
    best_val_loss = None
    for i in range(len(train_batches)):
        batch_loss = 0
        start = time.time()
        print('=' * 60) 
        print('Training batch: %d' % i)
        print('-' * 60)
        print('Elapsed (Left) \t\t Iter. \t\t Avg. Training Loss')
        print('-' * 60)
        count = 0
        print_loss = 0
        for pair in train_batches[i]:
            input_tensor = embed.tensorFromSentence(ixgen, pair[0])
            output_tensor = embed.tensorFromSentence(ixgen, pair[1])
            # input_tensor, output_tensor = embed.tensorsFromPair(ixgen, pair)
            loss = train(input_tensor, output_tensor, encoder, decoder, encoder_optim, decoder_optim, criterion)
            count += 1
            batch_loss += loss
            print_loss += loss
            if count % log_interval == 0:
                print('%s \t %d \t %d%% \t\t %.4f' % (timeSince(start, count / len(train_batches[i])), int(count), float(count/len(train_batches[i]) * 100), float(print_loss/log_interval)))
                print_loss = 0
        print('-' * 60)
        print('Validation batch: %d' % i)
        print('-' * 60)
        print('Elapsed \t Left \t Iter. \t Avg. Validation Loss')
        batch_loss = batch_loss / len(train_batches[i])
        count = 0
        print_loss = 0
        val_loss = 0
        with torch.no_grad():
            for pair in dev_batches[i]:
                input_tensor, output_tensor = embed.tensorsFromPair(pair)
                loss = train(input_tensor, output_tensor, encoder, decoder, encoder_optim, decoder_optim, criterion)
                count += 1
                print_loss += loss
                val_loss += loss
                if count % log_interval == 0:
                    print('%s \t %d \t %d%% \t\t %.4f' % (timeSince(start, count / len(train_batches)), int(count), float(count/len(train_batches) * 100), float(print_loss/log_interval)))
                    print_loss = 0
            val_loss = val_loss/len(dev_batches[i])
            if not best_val_loss or val_loss < best_val_loss:
                with open(model_save_path, 'wb') as f:
                    torch.save({
                        'encoder': encoder.state_dict(),
                        'encoder_optim': encoder_optim.state_dict(),
                        'decoder': decoder.state_dict(),
                        'decoder_optim': decoder_optim.state_dict(),
                        }, f)
            print('-' * 60)
            print('Avg. Loss over training batch %d = %.4f' % i, batch_loss/len(train_batches[i]))
            print('=' * 60)

def evaluate(ixgen, encoder, decoder, eval_batches, learning_rate=0.2, model_save_path='./model.ckpt', log_interval=1000):
    ckpt = torch.load(model_save_path)
    encoder.load_state_dict(ckpt['encoder'])
    decoder.load_state_dict(ckpt['decoder'])
    
    criterion = nn.NLLLoss()
    encoder_optim = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optim = optim.SGD(encoder.parameters(), lr=learning_rate)
    
    encoder.load_state_dict(ckpt['encoder'])
    decoder.load_state_dict(ckpt['decoder'])
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        print('Starting evaluation...')
        for i in range(len(eval_batches)):
            batch_loss = 0
            start = time.time()
            print('=' * 60) 
            print('Evaluating batch: %d' % i)
            print('-' * 60)
            print('Elapsed \t Left \t Iter. \t Avg. Evaluation Loss')
            print('-' * 60)
            count = 0
            print_loss = 0
            for pair in eval_batches[i]:
                input_tensor, output_tensor = embed.tensorsFromPair(ixgen, pair)
                loss = train(input_tensor, output_tensor, encoder, decoder, encoder_optim, decoder_optim, criterion)
                count += 1
                batch_loss += loss
                print_loss += loss
                if count % log_interval == 0:
                    print('%s \t %d \t %d%% \t\t %.4f' % (timeSince(start, count / len(train_batches)), int(count), float(count/len(train_batches) * 100), float(print_loss/log_interval)))
            print('-' * 60)
            print('Avg. Loss over training batch %d = %.4f' % i, batch_loss/len(eval_batch[i]))
            print('=' * 60)

encoder = model.EncoderRNN(ixgen.n_words, args.nhid, args.nlayers, wtmatrix, dropout=args.dropout).to(device)
if not args.attention:
    decoder = model.DecoderRNN(args.nhid, ixgen.n_words, args.nlayers, dropout=args.dropout).to(device)
else:
    decoder = model.AttnDecoderRNN(args.nhid, ixgen.n_words, args.nlayers, dropout_p=args.dropout).to(device)

trainIters(ixgen, encoder, decoder, train_batches, dev_batches, args.log_interval, args.lr, args.model_save_path)
evaluate(ixgen, encoder, decoder, test_batches, args.lr, args.model_save_path, args.log_interval)
