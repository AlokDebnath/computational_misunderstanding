import pandas as pd
from tqdm import tqdm
import os
import re


def get_lines(fpath):
    print('Reading data...')
    df_1 = pd.read_pickle(fpath)
    src_lines = list(df_1['Source_Line'])
    tgt_lines = list(df_1['Target_Line'])
    return dict(zip(src_lines, tgt_lines))

def clean(text):
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    return text
    

def fwritedata(fname, data):
    col1 = list(data.keys())
    col2n = list(data.values())
    print('Cleaning and Writing dataset...')
    with open(fname, 'w+') as f:
        for ix in tqdm(range(len(col1))):
            f.write(col1[ix] + '\t' + col2n[ix] + '\n')
    f.close()


if __name__ == '__main__':
    path_rev1 = './Revision_Depth1.pkl'
    path_revm = './Revision_Depth_Multiple.pkl'
    data = get_lines(path_rev1)
    data.update(get_lines(path_revm))
    fwritedata('./dataset.tsv', data)
    print('Dataset Prepared!')
