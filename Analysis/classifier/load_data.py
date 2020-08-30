# _*_ coding: utf-8 _*_

import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, FastText
import pandas as pd

from torchtext import data

class DataFrameDataset(data.Dataset):
    def __init__(self, df, text_field, label_field, is_test=False, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        for i, row in df.iterrows():
            label = row.label if not is_test else None
            text = row.text
            examples.append(data.Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, text_field, label_field, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)

        if train_df is not None:
            train_data = cls(train_df.copy(), text_field, label_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), text_field, label_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), text_field, label_field, True, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)

def create_dataset_from_csv(datadir, dev_file_path, test_file_path):
    df = pd.concat([pd.read_csv(os.path.join(datadir, f)) for f in os.listdir(datadir) if '.csv' in f], ignore_index=True, copy=False)
    print(df)
    train_src = list()
    test_src = list()
    dev_src = list()
    train_tgt = list()
    test_tgt = list()
    dev_tgt = list()
    dev_files = set(open(dev_file_path).read().split())
    test_files = set(open(test_file_path).read().split())
    for ix in range(len(df['File Name'])):
        fname = str(df['File Name'][ix])
        src = str(df['Source'][ix])
        tgt = str(df['Target'][ix])
        if fname in dev_files:
            dev_src.append((src, 0))
            dev_tgt.append((tgt, 1))
        elif fname in test_files:
            test_src.append((src, 0))
            test_tgt.append((tgt, 1))
        else:
            train_src.append((src, 0))
            train_tgt.append((tgt, 1))
    
    val_df = pd.DataFrame(dev_src + dev_tgt, columns=['text', 'label'])
    test_df = pd.DataFrame(test_src + test_tgt, columns=['text', 'label'])
    train_df = pd.DataFrame(train_src + train_tgt, columns=['text', 'label'])
    return train_df, test_df, val_df


def load_dataset(train_df, test_df, val_df):
    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
    LABEL = data.LabelField(tensor_type=torch.FloatTensor)
    train_data, valid_data, test_data = DataFrameDataset.splits(text_field=TEXT, label_field=LABEL, train_df=train_df, val_df=val_df, test_df=test_df)
    TEXT.build_vocab(train_data, vectors=FastText(language='en'))
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print("Label Length: " + str(len(LABEL.vocab)))
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32, sort_key=lambda x: len(x.text), repeat=False, shuffle=False)
    vocab_size = len(TEXT.vocab)
    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter 
