import os
import pandas as pd
from tqdm import tqdm

def construct_csv(fname):
    example = list()
    with open(fname, 'r') as f:
        lines = f.readlines()
        topic = list()
        old = list()
        new = list()
        ctx = list()
        for line in tqdm(lines):
            if line[0] == '=':
                topic.append(line.strip('=').strip())
            elif line[:3] ==  'OLD':
                old.append(line.strip('OLD: ').strip())
            elif line[:3] ==  'NEW':
                new.append(line.strip('NEW: ').strip())
            elif '----' in line:
                example.append([topic, old, new, ctx])
                topic = list()
                old = list()
                new = list()
                ctx = list()
            elif line != '':
                ctx.append(line.strip('\t').strip())
            else:
                print(line)
    return example

if __name__ == '__main__':
    fname = './wikiHow_revisions_filtered_dev1perc_wContext.txt'
    ex = construct_csv(fname)
