import os
from tqdm import tqdm
import numpy as np

def load_data(fpath):
    inp = list()
    out = list()
    with open(fpath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            inp, out = line.split('\t')
            inp = inp.strip()
            out = out.strip()
    return inp, out

def load_embedding(fpath):
    emb = dict()
    with open(fpath, 'rb') as f:
        text = f.read().decode(encoding='utf-8')
        # lines = text.split('\n')
        [vocab_size, NDIMS] = text.split('\n')[0].split()
        # print(vocab_size, NDIMS)
        for line in tqdm(text.split('\n')[1:-1]):
            word = line.split()[0]
            embedding = np.array(line.split()[1:], dtype='float')
            emb[word.lower()] = embedding
        emb['<SOS>'] = np.random.randn(300)
        emb['<EOS>'] = np.random.randn(300)
    return emb

def get_sent_rep(sent, emb_dict):
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
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
        for word in sentence:
            if word in contractions:
                word = contractions[word]
        

if __name__ == '__main__':    
    load_embedding('./wiki-news-300d-1M.bin')
    
