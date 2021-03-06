import preprocess
import model
import embed

import argparse
import time
import pickle
import random
import math
import torch
import torch.nn as nn
import numpy as np
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 50
SOS_token = 0
EOS_token = 1

teacher_forcing_ratio = 0.2
def train(input_tensor, inputpos_tensor, target_tensor, encoder, decoder, encoder_optim, decoder_optim, criterion, max_length=MAX_LENGTH, train=1, attention=1):
    encoder_hidden = encoder.initHidden()
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.nhid, device=device)

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], inputpos_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[preprocess.SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        for di in range(target_length):
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        for di in range(target_length):
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == preprocess.EOS_token:
                break
    if train:
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

def trainIters(ixgen, encoder, decoder, train_data, dev_data, batch_size, eval_batch_size, attention=1, log_interval=1, learning_rate=0.01, model_save_path='./model.ckpt', resume=False):
    print('Starting training...')
    criterion = nn.NLLLoss()
    encoder_optim = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optim = optim.SGD(encoder.parameters(), lr=learning_rate)
    best_val_loss = None
    if resume:
        ckpt = torch.load(model_save_path)
        i = ckpt['iter']
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
    for n in range(int(len(train_data)/batch_size)):
        batch_loss = 0
        start = time.time()
        print('=' * 65) 
        print('Training batch: %d' % n)
        print('-' * 65)
        print('Elapsed (Left) \t\t Iter. \t\t Avg. Training Loss')
        print('-' * 65)
        print_loss_total = 0
        training_pairs = [random.choice(train_data) for i in range(batch_size)] 
        for iter in range(1, batch_size + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor, target_tensor = embed.tensorsFromPair(ixgen, training_pair)
            input_postensor = embed.posTensorFromSentence(ixgen, training_pair[0])
            loss = train(input_tensor, input_postensor, target_tensor, encoder, decoder, encoder_optim, decoder_optim, criterion, train=1, attention=attention)
            print_loss_total += loss
            batch_loss += loss
            if iter % log_interval == 0:
                print_loss_avg = print_loss_total / log_interval
                print_loss_total = 0
                print('%s \t %d \t %d%% \t\t %.4f' % (timeSince(start, iter / batch_size), iter, iter / batch_size * 100, print_loss_avg))
        start = time.time()
        print('=' * 65) 
        print('Validating batch: %d' % n)
        print('-' * 65)
        print('Elapsed (Left) \t\t Iter. \t\t Avg. Training Loss')
        print('-' * 65)
        print_loss_total = 0
        val_loss = 0
        dev_pairs = [random.choice(dev_data) for i in range(eval_batch_size)] 
        for i in range(1, eval_batch_size + 1):
            dev_pair = dev_pairs[i - 1]
            input_tensor, target_tensor = embed.tensorsFromPair(ixgen, dev_pair)
            input_postensor = embed.posTensorFromSentence(ixgen, dev_pair[0])
            with torch.no_grad():
                loss = train(input_tensor, input_postensor, target_tensor, encoder, decoder, encoder_optim, decoder_optim, criterion, train=0, attention=attention)
            print_loss_total += loss
            val_loss += loss
            if i % int(0.1*log_interval) == 0:
                print_loss_avg = print_loss_total / int(0.1*log_interval)
                print_loss_total = 0
                print('%s \t %d \t %d%% \t\t %.4f' % (timeSince(start, i / eval_batch_size), i, i / eval_batch_size * 100, print_loss_avg))
            
            val_loss = val_loss / eval_batch_size
            if not best_val_loss or val_loss < best_val_loss:
                with open(model_save_path, 'wb') as f:
                    torch.save({
                        'iter': n,
                        'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict(),
                        }, f)
                    best_val_loss = val_loss
        print('-' * 65)
        print('Avg. Loss over training batch %d = %.4f' % (n, batch_loss/batch_size * 100))
        print('-' * 65)

def evaluate(ixgen, encoder, decoder, eval_data, eval_batch_size, attention=1, learning_rate=0.01, model_save_path='./model.ckpt', log_interval=1000):
    ckpt = torch.load(model_save_path)
    encoder.load_state_dict(ckpt['encoder'])
    decoder.load_state_dict(ckpt['decoder'])
    
    criterion = nn.NLLLoss()
    encoder_optim = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optim = optim.SGD(encoder.parameters(), lr=learning_rate)
    
    encoder.eval()
    decoder.eval()
    for i in range(int(len(test_data)/eval_batch_size)):
        start = time.time()
        print('=' * 65) 
        print('Evaluating batch: %d' % i)
        print('-' * 65)
        print('Elapsed (Left) \t\t Iter. \t\t Avg. Evaluation Loss')
        print('-' * 65)
        print_loss_total = 0
        batch_loss = 0
        test_pairs = [random.choice(test_data) for i in range(eval_batch_size)] 
        for iter in range(1, eval_batch_size + 1):
            test_pair = test_pairs[iter - 1]
            input_tensor, target_tensor = embed.tensorsFromPair(ixgen, test_pair)
            input_postensor = embed.posTensorFromSentence(ixgen, test_pair[0])
            with torch.no_grad():
                loss = train(input_tensor, input_postensor, target_tensor, encoder, decoder, encoder_optim, decoder_optim, criterion, train=0, attention=attention)
            print_loss_total += loss
            batch_loss += loss
            if iter % 0.1*log_interval == 0:
                print_loss_avg = print_loss_total / 0.1*log_interval
                print_loss_total = 0
                print('%s \t %d \t %d%% \t\t %.4f' % (timeSince(start, iter / eval_batch_size), iter, iter / eval_batch_size * 100, print_loss_avg))
            
            print('-' * 65)
            print('Avg. Loss over training batch %d = %.4f' % (i, batch_loss/eval_batch_size * 100))
            print('-' * 65)


def manual(ixgen, encoder, decoder, sentence, model_save_path, max_length=MAX_LENGTH):
    ckpt = torch.load(model_save_path)
    encoder.load_state_dict(ckpt['encoder'])
    decoder.load_state_dict(ckpt['decoder'])
    with torch.no_grad():
        input_tensor = embed.tensorFromSentence(ixgen, sentence)
        input_postensor = embed.posTensorFromSentence(ixgen, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        decoded_words = list()
        encoder_outputs = torch.zeros(MAX_LENGTH, encoder.nhid, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], input_postensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        decoder_input = torch.tensor([[preprocess.SOS_token]], device=device)
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
    print('> Enter sentences at the prompt. To quit, type \'QUIT\' \n< The output will be provided as soon as the model computes')
    while True:
        sentence = input('> ')
        if sentence == 'QUIT':
            break
        elif sentence == 'RANDOM':
            evaluateRandomly(ixgen, encoder, decoder, lines, model_save_path, n=10)
        elif len(sentence.split()) > MAX_LENGTH:
            print('<< Maximum sentence length is %d words' % MAX_LENGTH)
        output = manual(ixgen, encoder, decoder, sentence, model_save_path)
        print('< ' + str(output))
        print('')

def evaluateRandomly(ixgen, encoder, decoder, pairs, model_save_path, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        if len(pair[1]) > MAX_LENGTH or len(pair[2]) > MAX_LENGTH:
            n -= 1
            continue
        print('>', pair[1])
        print('=', pair[2])
        output_sentence = manual(ixgen, encoder, decoder, pair[1], model_save_path)
        print('<', output_sentence)
        print('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicting edited version from original version')
    parser.add_argument('--data_path', type=str, default='../data/wikiHow_revisions_corpus.txt', help='location of dataset')
    parser.add_argument('--test_path', type=str, default='../data/test_files.txt', help='path to test split file')
    parser.add_argument('--dev_path', type=str, default='../data/dev_files.txt', help='path to dev split file')
    parser.add_argument('--attention', action='store_true', help='use attention decoder? Default: True')
    parser.add_argument('--nlayers', type=int, default=5, help='number of hidden layers in the encoder and decoder')
    parser.add_argument('--nhid', type=int, default=256, help='hidden dimension of encoder and decoder')
    parser.add_argument('--batch_size', type=int, default=1000, help='size of the training data batches. Test set bsz is 0.1*train_bsz')
    parser.add_argument('--dropout', type=float, default=0, help='dropout for the encode and decoder')
    parser.add_argument('--model_save_path', type=str, default='/tmp/model.ckpt', help='save location for the model for evaluation')
    parser.add_argument('--lr', type=float, default=0.01, help='model learning rate')
    parser.add_argument('--log_interval', type=int, default=100, help='reporting interval')
    parser.add_argument('--seed', type=int, default=1111, help='manual seed for reproducability')
    parser.add_argument('--resume', action='store_false', help='resume from a previous checkpoint?')
    parser.add_argument('--limit', type=int, default=None, help='number of random cases to consider')
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    
    ixgen, lines = preprocess.prepareData(args.data_path, args.limit)
    train_data, test_data, dev_data = preprocess.read_train_test_dev(lines, args.test_path, args.dev_path)
    
    eval_batch_size = int(0.1 * args.batch_size)
    # train_batches = preprocess.get_batches(args.batch_size, train_data)
    # test_batches = preprocess.get_batches(eval_batch_size, test_data)
    # dev_batches = preprocess.get_batches(eval_batch_size, dev_data) 
    
    # train_batches = train_batches[:min(len(train_batches), len(dev_batches))]
    # dev_batches = dev_batches[:min(len(train_batches), len(dev_batches))]
    
    # print('Number of batches in train, test and validation: %d' % (len(train_batches)))
    
    encoder = model.EncoderRNN(ixgen.n_words, ixgen.n_postags, args.nhid, args.nlayers, dropout=args.dropout).to(device)
    if not args.attention:
        decoder = model.DecoderRNN(args.nhid, ixgen.n_words, args.nlayers, dropout=args.dropout).to(device)
    else:
        decoder = model.AttnDecoderRNN(args.nhid, ixgen.n_words, args.nlayers, dropout_p=args.dropout).to(device)
    trainIters(ixgen, encoder, decoder, train_data, dev_data, args.batch_size, eval_batch_size, int(args.attention), args.log_interval, args.lr, args.model_save_path)
    evaluate(ixgen, encoder, decoder, test_data, eval_batch_size, int(args.attention), args.lr, args.model_save_path, args.log_interval)

    cli(ixgen, encoder, decoder, lines, args.model_save_path) 
