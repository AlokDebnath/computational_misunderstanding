# -*- coding: utf-8 -*-
import os
import io
from tqdm import tqdm
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import unicodedata

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IxGen:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" ", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readIxGens(lines):
    print("Reading lines...")
    pairs = [[normalizeString(s) for s in l] for l in tqdm(lines)]
    ixgen = IxGen()
    return ixgen, pairs

MAX_LENGTH = 50

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lines):
    ixgen, pairs = readIxGens(lines)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        ixgen.addSentence(pair[0])
    print("Counted words:")
    print(ixgen.n_words)
    return ixgen, pairs

def indexesFromSentence(sentence):
    ixs = list()
    for word in sentence.split():
        if word in ixgen.word2index.keys():
            ixs.append(ixgen.word2index[word])
        else:
            ixgen.addWord(word)
            ixs.append(ixgen.word2index[word])
    return ixs


def tensorFromSentence(sentence):
    indexes = indexesFromSentence(sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(pair[0])
    target_tensor = tensorFromSentence(pair[1])
    return (input_tensor, target_tensor)

def read_train_test_dev(batch_size, dev_files, test_files, data_file):
    dev_X = []
    test_X = []
    train_X = []
    dev_files = set(open(dev_files).read().split())
    test_files = set(open(test_files).read().split())
    dev_id, test_id = 0, 0
    with io.open(data_file, encoding='utf-8') as fp:
        for line in fp:
            title, revision_group, src, tgt = line.strip().split('\t')
            if len(src) > 150 or len(tgt) > 150:
                continue
            if title in dev_files:
                dev_X.append([src, tgt])
            elif title in test_files:
                test_X.append([src, tgt])
            else:
                train_X.append([src, tgt])
    return train_X, test_X, dev_X

a, b, c = read_train_test_dev(100, os.path.join('../data', 'dev_files.txt'), os.path.join('../data/', 'test_files.txt'), 
                                '../data/wikiHow_revisions_corpus.txt')
prepareData(a)
