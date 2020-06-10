import numpy as np
import os
import operator

def getEmbeddings():
    word2ix = dict()
    vecs = np.array()
    words = list()
    ix = 0
    with open('./glove.6B.300d.txt',  encoding='utf-8') as f:
        for line in f:
            word = line.decode().split()[0]
            vecs.append(np.array(line.decode().split()[1:], dtype='float32'))
            word2ix[word] = ix
            words.append(word)
            ix += 1
    vecs = vecs.reshape((len(words), 300))
    glove = {w: vecs[word2ix[w]] for w in words}
    sos_index = word2idx['sos']
    eos_index = word2idx['eos']
    sos_swap_word = words[0]
    eos_swap_word = words[1]
     
    words[0], words[sos_index] = words[sos_index], words[0]
    words[1], words[eos_index] = words[eos_index], words[1]
    word2ix[sos_swap_word], word2ix['sos'] = word2ix['sos'], word2ix[sos_swap_word]
    word2ix[eos_swap_word], word2ix['eos'] = word2ix['eos'], word2ix[eos_swap_word]
    word2ix = { k : v for k , v in sorted(word2ix.items(), key=operator.itemgetter(1))}
    

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

