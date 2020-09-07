import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
from transformers import BertTokenizer, BertModel

class DataFrameDataset(data.Dataset):
    def __init__(self, df, text_field, textA_field, textB_field, labelA_field, labelB_field, is_test=False, **kwargs):
        fields = [('text', text_field), ('textA', textA_field), ('textB', textB_field), ('labelA', labelA_field), ('labelB', labelB_field)]
        examples = []
        for i, row in df.iterrows():
            labelA = row.labelA if not is_test else None
            labelB = row.labelB if not is_test else None
            text = row.text
            textA = row.textA
            textB = row.textB
            examples.append(data.Example.fromlist([text, textA, textB, labelA, labelB], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, text_field, textA_field, textB_field, labelA_field, labelB_field, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)

        if train_df is not None:
            train_data = cls(train_df.copy(), text_field, textA_field, textB_field, labelA_field, labelB_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), text_field, textA_field, textB_field, labelA_field, labelB_field, True, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), text_field, textA_field, textB_field, labelA_field, labelB_field, True, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)

def create_dataset_from_csv(datadir, test_file_path, dev_file_path):
    df = pd.concat([pd.read_csv(os.path.join(datadir, f)) for f in os.listdir(datadir) if '.csv' in f], ignore_index=True, copy=False)
    
    X_train = list()
    X_test = list()
    X_val = list()
    y_train = list()
    y_test = list()
    y_val = list()
    
    val_files = set(open(dev_file_path).read().split())
    test_files = set(open(test_file_path).read().split())
    ntrain = 0
    ntest = 0
    nval = 0

    for ix in range(len(df['Source'])):
        if df['File Name'][ix] in val_files:
            if nval % 2 != 0:
                X_val.append((df['Source'][ix], df['Target'][ix]))
                y_val.append((0, 1))
            else:
                X_val.append((df['Target'][ix], df['Source'][ix]))
                y_val.append((1, 0))
            nval += 1
        elif df['File Name'][ix] in test_files:
            if ntest % 2 != 0:
                X_test.append((df['Source'][ix], df['Target'][ix]))
                y_test.append((0, 1))
            else:
                X_test.append((df['Target'][ix], df['Source'][ix]))
                y_test.append((1, 0))
            ntest += 1
        else:
            if ntrain % 2 != 0:
                X_train.append((df['Source'][ix], df['Target'][ix]))
                y_train.append((0, 1))
            else:
                X_train.append((df['Target'][ix], df['Source'][ix]))
                y_train.append((1, 0))
            ntrain += 1
    
    
    train_= [(' '.join((X_train[i][0], X_train[i][1])), X_train[i][0], X_train[i][1], y_train[i][0], y_train[i][1]) for i in range(len(X_train))]
    test_= [(' '.join((X_train[i][0], X_train[i][1])), X_test[i][0], X_test[i][1], y_test[i][0], y_test[i][1]) for i in range(len(X_test))]
    val_= [(' '.join((X_train[i][0], X_train[i][1])), X_val[i][0], X_val[i][1], y_val[i][0], y_val[i][1]) for i in range(len(X_val))]
    

    val_df = pd.DataFrame(val_, columns=['text', 'textA', 'textB', 'labelA', 'labelB'])
    test_df = pd.DataFrame(test_, columns=['text', 'textA', 'textB', 'labelA', 'labelB'])
    train_df = pd.DataFrame(train_, columns=['text', 'textA', 'textB', 'labelA', 'labelB'])
    return train_df, test_df, val_df

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token
init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens

def load_dataset(train_df, test_df, val_df):
    TEXT_A = data.Field(batch_first = True,
            use_vocab = False,
            sequential = True,
            tokenize = tokenize_and_cut,
            preprocessing = tokenizer.convert_tokens_to_ids,
            init_token = init_token_idx,
            eos_token = eos_token_idx,
            pad_token = pad_token_idx,
            unk_token = unk_token_idx,
            include_lengths=True,
            fix_length = 50)
    TEXT_B = data.Field(batch_first = True,
            use_vocab = False,
            sequential = True,
            tokenize = tokenize_and_cut,
            preprocessing = tokenizer.convert_tokens_to_ids,
            init_token = init_token_idx,
            eos_token = eos_token_idx,
            pad_token = pad_token_idx,
            unk_token = unk_token_idx,
            include_lengths=True,
            fix_length = 50)
    TEXT = data.Field(batch_first = True,
            use_vocab = False,
            sequential = True,
            tokenize = tokenize_and_cut,
            preprocessing = tokenizer.convert_tokens_to_ids,
            init_token = init_token_idx,
            eos_token = eos_token_idx,
            pad_token = pad_token_idx,
            unk_token = unk_token_idx,
            include_lengths=True,
            fix_length = 100)
    LABEL_A = data.LabelField(tensor_type=torch.FloatTensor)
    LABEL_B = data.LabelField(tensor_type=torch.FloatTensor)
    train_data, valid_data, test_data = DataFrameDataset.splits(text_field=TEXT, textA_field=TEXT_A, textB_field=TEXT_B, labelA_field=LABEL_A, labelB_field=LABEL_B, train_df=train_df, val_df=val_df, test_df=test_df)

    TEXT.vocab = tokenizer.vocab
    TEXT_A.vocab = TEXT.vocab
    TEXT_B.vocab = TEXT.vocab
    LABEL_A.build_vocab(train_data)
    LABEL_B.build_vocab(train_data)

    vocab_size = len(TEXT.vocab)

    print("Length of Text Vocabulary: " + str(len(tokenizer.vocab)))
    print("Label Length: " + str(len(LABEL_A.vocab)))
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32, repeat=False, shuffle=True)
    return TEXT_A, TEXT_B, vocab_size, train_iter, valid_iter, test_iter 



if __name__ == '__main__':
    datadir = '/mount/projekte/emmy-noether-roth/mist/misunderstanding/csv2'
    # datadir = '/tmp/misunderstanding'
    test_path = '/tmp/misunderstanding/computational_misunderstanding/Experiments/data/test_files.txt'
    val_path = '/tmp/misunderstanding/computational_misunderstanding/Experiments/data/dev_files.txt'
    train_df, test_df, val_df = create_dataset_from_csv(datadir, test_path, val_path)
    TEXT_A, TEXT_B, vocab_size, train_iter, valid_iter, test_iter = load_dataset(train_df, test_df, val_df)
    for idx, batch in enumerate(train_iter):
        textA = batch.textA[0]
        textB = batch.textB[0]
        print("textA[0]: " + str(textA))
        print("textB[0]: " + str(textB))
        break
