# -*- coding: utf-8 -*-
import random
import io
from tqdm import tqdm
import re
import torch
import unicodedata
import embed
import feature

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SOS_token = 0
EOS_token = 1
PAD_token = 2

class IxGen:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.pos2index = {0: "TAG"}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 3
        self.n_postags = 0

    def addSentence(self, sentence):
        pos = feature.posParse(sentence)
        for ix in range(len(sentence.strip().split(' '))):
            self.addWord(sentence.strip().split(' ')[ix])
            self.addPos(pos[ix])

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def addPos(self, pos):
        if pos not in self.pos2index:
            self.pos2index[pos] = self.n_postags
            self.n_postags += 1

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
    # s = ' '.join([contractions[w] if w in contractions else w for w in s.split()])
    s = re.sub(r"([.!?])", r" ", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readIxGens(lines, limit):
    print("Reading lines...")
    newlines = []
    for line in lines:
        topic, revision, p0, p1 = line.strip().split('\t')
        newlines.append([topic, p0, p1])
    lines = newlines
    pairs = []
    limit = len(lines) if (limit == None) else limit
    lines = [random.choice(lines) for i in range(limit)]
    for l in tqdm(lines):    
        pair = []
        pair.append(l[0])
        pair.append(normalizeString(l[1]))
        pair.append(normalizeString(l[2]))
        pairs.append(pair)
    ixgen = IxGen()
    return ixgen, pairs

MAX_LENGTH = 50


def filterPair(p):
    return len(p[1].split()) < MAX_LENGTH and len(p[2].split()) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(fpath, limit):
    lines = open(fpath, encoding='utf-8').read().strip().split('\n')
    ixgen, pairs = readIxGens(lines, limit)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in tqdm(pairs):
        ixgen.addSentence(pair[1])
    print("Counted words:")
    print(ixgen.n_words)
    return ixgen, pairs

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
        if title in dev_files:
            dev_X.append([src, tgt])
        elif title in test_files:
            test_X.append([src, tgt])
        else:
            train_X.append([src, tgt])
    return train_X, test_X, dev_X

def get_batches(batch_size, data):
    num_batches = int(len(data)/batch_size)
    batches = list()
    i = 0
    for i in range(num_batches):
        batches.append(data[i*batch_size:(i+1)*batch_size])
    batches.append(data[(i+1)*batch_size:])
    return batches

if __name__ == '__main__':
    ixgen, data = prepareData('../data/wikiHow_revisions_corpus.txt', 10000)
    print(data[0])
    print(ixgen.n_postags)
    train_X, test_X, dev_X = read_train_test_dev(data, '../data/test_files.txt', '../data/dev_files.txt')
    train_b = get_batches(1000, train_X)
    print(train_b[0][0])
