import numpy as np
import os
import operator
from tqdm import tqdm
import torch
from gensim.models.keyedvectors import KeyedVectors


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def getEmbeddings(fpath):
#     wv_from_bin = KeyedVectors.load_word2vec_format(fpath, binary=True)
#     word2ix = dict()
#     glove = dict()
#     ix = 0
#     print('Getting embeddings...')
#     for word, vector in tqdm(zip(wv_from_bin.vocab, wv_from_bin.vectors)):
#         coefs = np.asarray(vector, dtype='float32')
#         word2ix[word] = ix
#         ix += 1
#         glove[word] = coefs
#     return word2ix, glove
    

def indexesFromSentence(ixgen, sentence):
    ixs = list()
    for word in sentence.split():
        if word in ixgen.word2index.keys():
            ixs.append(ixgen.word2index[word])
        else:
            ixgen.addWord(word)
            ixs.append(ixgen.word2index[word])
    return ixs


def tensorFromSentence(ixgen, sentence):
    indexes = indexesFromSentence(ixgen, sentence)
    indexes.append(1)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(ixgen, pair):
    input_tensor = tensorFromSentence(ixgen, pair[0])
    target_tensor = tensorFromSentence(ixgen, pair[1])
    return input_tensor, target_tensor

if __name__ == '__main__':
    word2ix, glove = getEmbeddings('./glove.bin')
    print(word2ix['the'], glove['the'])
