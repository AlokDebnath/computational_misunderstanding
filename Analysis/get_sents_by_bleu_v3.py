import re
import os
import sys
import bz2
import string
import difflib

import editdistance as ed
from polyglot_tokenizer import Tokenizer

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
cc = SmoothingFunction()

tok = Tokenizer(lang='en', split_sen=True)

stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}

table = str.maketrans({key: None for key in string.punctuation+string.digits})

dirr = 'wikiHow_articles'

def normalize_punkt(text):
    """replace unicode punctuation by ascii"""
    text = re.sub('[\u2010\u2043]', '-', text)  # hyphen
    text = re.sub('[\u2018\u2019]', "'", text)  # single quotes
    text = re.sub('[\u201c\u201d]', '"', text)  # double quotes
    return line

def detok(tok_lines, line):
    line = normalize_punkt(line)
    if len(tok_lines) == 1:
        return [line]
    nline = ''
    for tline in tok_lines:
        sln = len(' '.join(tline))-len(tline)+1
        while sln:
            nline += line[0]
            if line[0] != ' ':
                sln -= 1 
            line = line[1:]
        nline += '\n'
    nline = [x.strip() for x in nline.strip().split('\n')]
    return nline

#articles = set(open(sys.argv[1]).read().split('\n'))
#print('Article_Name\tBLEU_Score\tSection_Src\tLine_No_Src\tSource_Line\tSection_Tgt\tLine_No_Tgt\tTarget_Line')
all_files = os.listdir(dirr)
idx = int(sys.argv[1])
for file_ in all_files[idx*2570:idx*2570+2570]:
  sys.stderr.write(file_+'\n')
  sys.stderr.flush()
  #if file_ in articles:
  #  continue
  with bz2.open('%s/%s' %(dirr,file_)) as fp:
    ts_names = []
    tstamp = []
    tstamp_list = []
    recompute = False
    title = fp.readline().decode('utf-8') 
    for i,line in enumerate(fp, 2):
        line = line.strip().decode('utf-8')
        if not line:
            continue
        if line.startswith('## Timestamp'):
            key = 'Intro'
            ts_names.append(line)
            if not tstamp:
                continue
            tstamp_list.append(tstamp)
            tstamp = []
            continue
        if line.strip().startswith('#'):
            key = line
            continue
        line = re.sub(r'<.*?>', r'', line)
        line = re.sub(r'\[\[Image:.*?\]\]', r'', line)
        line = line.lstrip('-*\\0123456789. ')
        tok_lines = tok.tokenize(line)
        #detok_lines = detok(tok_lines, line)
        #for tln, dln in zip(tok_lines, tok_lines):
        for tln in tok_lines:
            tln_str = ' '.join(tln)
            if 'Related' in key and '[ [' in tln_str:
                continue
            if '[ [' in tln_str:
                tln_str = re.sub(r'\[ \[[^\[]*?\|(.*?)\] \]', r'\1', tln_str)
            if '[ [' in tln_str:
                tln_str = re.sub(r'\[ \[[^\[]*?\|(.*?)\] \]', r'\1', tln_str)
            if '[ [' in tln_str:
                tln_str = re.sub(r'\[ \[[^\[]*?\|(.*?)\] \]', r'\1', tln_str)
            tln_str = tln_str.replace('[ [', ' ').replace('] ]', ' ')
            tln = tln_str.split()
            #if len(tln) > 50:  #NOTE EMNLP addition
            #    continue
            #tstamp.append((tuple(tln), dln, key, str(i)))
            tstamp.append((tuple(tln), tln_str, key, str(i)))
    if tstamp:
        tstamp_list.append(tstamp)
        tstamp = []
    for ts1, ts2  in zip(tstamp_list[:-1], tstamp_list[1:]):
        seta = set([t for t,_,_,_ in ts1])
        for x,dx,key1,line_no1 in ts1:
            if x == None:
                continue
            lnx = len(x)
            if lnx < 2:
                continue
            max_bleu = -1
            yy = ''
            dyy = ''
            line_no2_yy = '-1'
            key2_yy = '-1'
            sx = ' '.join(x).translate(table).split()
            sx = set(sx) - stopwords
            for y,dy,key2,line_no2 in ts2:
                if y in seta:
                    continue
                if y == None:
                    continue
                lny = len(y)
                #if min(lnx, lny) / max(lnx, lny) < 0.6:
                sy = ' '.join(y).translate(table).split()
                sy = set(sy) -  stopwords
                common = sx & sy
                if not common:
                    continue
                if len(common) / len(sx | sy) < 0.4:
                    continue
                if lny > 1:
                    ref = list(map(str.lower, x))
                    hyp = list(map(str.lower, y))
                    if len(ref) < 5:
                        ref = (ref + ['x'] * 5)[:5]
                    if len(hyp) < 5:
                        hyp = (hyp + ['x'] * 5)[:5]
                    bleu = sentence_bleu([ref], hyp, smoothing_function=cc.method4)
                    if bleu > max_bleu:
                        max_bleu = bleu
                        yy = y
                        dyy = dy
                        key2_yy = key2
                        line_no2_yy = line_no2
            if max_bleu > 0.3:
                print(file_+'\t'+str(round(max_bleu, 3))+'\t'+key1+'\t'+line_no1+'\t'+dx+'\t'+key2_yy+'\t'+line_no2_yy+'\t'+dyy)
