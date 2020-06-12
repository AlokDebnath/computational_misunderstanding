# -*- coding: utf-8 -*-
import os
import io
import numpy as np
from tqdm import tqdm
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import unicodedata
import embed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IxGen:
    def __init__(self, fpath):
        self.word2index, self.glove = embed.getEmbeddings(fpath)
        self.word2count = {word : 1 for word in self.word2index.keys()}
        self.index2word = {ix: word for word, ix in self.word2index.items()}
        self.n_words = len(self.word2index.keys())

    def addSentence(self, sentence):
        for word in sentence.split():
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

SOS_token = 0
EOS_token = 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    contractions = { 
	"ain't": "am not",
	"aren't": "are not",
	"can't": "cannot",
	"can't've": "cannot have",
	"'cause": "because",
	"could've": "could have",
	"couldn't": "could not",
	"couldn't've": "could not have",
	"didn't": "did not",
	"doesn't": "does not",
	"don't": "do not",
	"hadn't": "had not",
	"hadn't've": "had not have",
	"hasn't": "has not",
	"haven't": "have not",
	"he'd": "he would",
	"he'd've": "he would have",
	"he'll": "he will",
	"he's": "he is",
	"how'd": "how did",
	"how'll": "how will",
	"how's": "how is",
	"i'd": "i would",
	"i'll": "i will",
	"i'm": "i am",
	"i've": "i have",
	"isn't": "is not",
	"it'd": "it would",
	"it'll": "it will",
	"it's": "it is",
	"let's": "let us",
	"ma'am": "madam",
	"mayn't": "may not",
	"might've": "might have",
	"mightn't": "might not",
	"must've": "must have",
	"mustn't": "must not",
	"needn't": "need not",
	"oughtn't": "ought not",
	"shan't": "shall not",
	"sha'n't": "shall not",
	"she'd": "she would",
	"she'll": "she will",
	"she's": "she is",
	"should've": "should have",
	"shouldn't": "should not",
	"that'd": "that would",
	"that's": "that is",
	"there'd": "there had",
	"there's": "there is",
	"they'd": "they would",
	"they'll": "they will",
	"they're": "they are",
	"they've": "they have",
	"wasn't": "was not",
	"we'd": "we would",
	"we'll": "we will",
	"we're": "we are",
	"we've": "we have",
	"weren't": "were not",
	"what'll": "what will",
	"what're": "what are",
	"what's": "what is",
	"what've": "what have",
	"where'd": "where did",
	"where's": "where is",
	"who'll": "who will",
	"who's": "who is",
	"won't": "will not",
	"wouldn't": "would not",
	"you'd": "you would",
	"you'll": "you will",
	"you're": "you are"
    }
    s = unicodeToAscii(s.lower().strip())
    s = ' '.join([contractions[w] if w in contractions else w for w in s.split()])
    s = re.sub(r"([.!?])", r" ", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readIxGens(lines, embpath):
    print("Reading lines...")
    newlines = []
    for line in lines:
        topic, revision, p0, p1 = line.strip().split('\t')
        newlines.append([topic, p0, p1])
    lines = newlines
    pairs = []
    for l in tqdm(lines):
        pair = []
        pair.append(l[0])
        pair.append(normalizeString(l[1]))
        pair.append(normalizeString(l[2]))
        pairs.append(pair)
    ixgen = IxGen(embpath)
    return ixgen, pairs

MAX_LENGTH = 150


def filterPair(p):
    return len(p[0].split()) < MAX_LENGTH and len(p[1].split()) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(fpath, embpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
    ixgen, pairs = readIxGens(lines, embpath)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        ixgen.addSentence(pair[1])
        ixgen.addSentence(pair[2])
    print("Counted words:")
    print(ixgen.n_words)
    return ixgen, pairs

def wtMatrix(ixgen):
    matrix_len = ixgen.n_words
    wtmatrix = list()
    for word in ixgen.word2index.values():
        if word in ixgen.glove.keys():
            wtmatrix.append(ixgen.glove[word])
        else:
            wtmatrix.append(np.random.normal(scale=0.5, size=(300,)))
    wtmatrix = np.array(wtmatrix, dtype='float32').reshape((matrix_len, 300))
    return wtmatrix
            
def read_train_test_dev(pairs, dev_files, test_files):
    dev_X = []
    test_X = []
    train_X = []
    dev_files = set(open(dev_files).read().split())
    test_files = set(open(test_files).read().split())
    print('Splitting data into train, test and dev...')
    for pair in tqdm(pairs):
        title = pair[0]
        src = pair[1]
        tgt = pair[2]
        if len(src) > 150 or len(tgt) > 150:
            continue
        if title in dev_files:
            dev_X.append([src, tgt])
        elif title in test_files:
            test_X.append([src, tgt])
        else:
            train_X.append([src, tgt])
    return train_X, test_X, dev_X

def get_batches(batch_size, data):
    num_batches = int(len(data)/batch_size)
    batches = dict()
    i = 0
    fin_size = len(data) % batch_size
    for i in range(num_batches):
        batches[i] = data[i*batch_size:(i+1)*batch_size]
    batches[i+1] = data[(i+1)*batch_size:]
    return batches

if __name__ == '__main__':
    ixgen, data = prepareData('../data/wikiHow_revisions_corpus.txt', './glove.bin')
    print(data[0])
    train_X, test_X, dev_X = read_train_test_dev(data, '../data/test_files.txt', '../data/dev_files.txt')
    # print(train_X[0], test_X[0], dev_X[0])
    wtmatrix = wtMatrix(ixgen)
    print(wtmatrix[0])
    print(get_batches(10, train_X))
