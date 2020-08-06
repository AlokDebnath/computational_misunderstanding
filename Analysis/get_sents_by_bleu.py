import re
import os
import sys
import string
import difflib

import editdistance as ed
from polyglot_tokenizer import Tokenizer

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
cc = SmoothingFunction()

tok = Tokenizer(lang='en', split_sen=True)

stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}

table = str.maketrans({key: None for key in string.punctuation+string.digits})

for file in os.listdir('wikiHow_articles'):
  with open('wikiHow_articles/%s' %file) as fp:
    ts_names = []
    tstamp = []
    tstamp_list = []
    for line in fp:
        line = line.strip()
        if not line:
            continue
        if line.startswith('## Timestamp'):
            ts_names.append(line)
            if len(tstamp) > 3:
                tstamp_list.append(tstamp)
                tstamp = []
            #if len(tstamp_list) == 5:
            #    break
        if line.strip().startswith('#'):
            continue
        line = re.sub(r'<.*?>', r'', line)
        line = re.sub(r'\[\[Image:.*?\]\]', r'', line)
        line = line.lstrip('-*\\0123456789. ')
        #tstamp.append(line)#.split())
        for ln in tok.tokenize(line):
            tstamp.append(tuple(ln))
            #tstamp.append(' '.join(ln))
    h = False
    sys.stderr.write(file+'\n')
    sys.stderr.flush()
    '''
    differ = difflib.Differ()
    for version, (ts1, ts2)  in enumerate(zip(tstamp_list[:-1], tstamp_list[1:]), 1):
        #for line in difflib.unified_diff(ts1, ts2):
        result = []
        for line in differ.compare(ts1, ts2):
            if line.startswith('  ') or line.startswith('? '):
                continue
            result.append(line.rstrip())
        for s1, s2 in zip(result[:-1], result[1:]):
            if s1[0] == s2[0]:
                continue
            if len(s1.split()) < 3 or len(s2.split()) < 3:
                continue
            bleu = sentence_bleu([s1.lower().split()[1:]], s2.lower().split()[1:], smoothing_function=cc.method4)
            if bleu > 0.3:
                bleu = str(round(bleu, 3))
                if s1[0] == '-':
                    print('\t'.join([file, str(version), bleu, s1[2:], s2[2:]]))
                else:
                    print('\t'.join([file, str(version), bleu, s2[2:], s1[2:]]))
        continue
    #'''
    #'''
    for version, (ts1, ts2)  in enumerate(zip(tstamp_list[:-1], tstamp_list[1:]), 1):
        seta = set(ts1)
        for x in ts1:
            if x == None:
                continue
            lnx = len(x)
            if lnx < 2:
                continue
            max_bleu = -1
            yy = ''
            sx = ' '.join(x).translate(table).split()
            sx = set(sx) - stopwords
            for y in ts2:
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
                if len(sx & sy) / len(sx | sy) < 0.4:
                    continue
                if lny > 1:
                    bleu = sentence_bleu([list(map(str.lower, x))], list(map(str.lower, y)), smoothing_function=cc.method4)
                    if bleu > max_bleu:
                        max_bleu = bleu
                        yy = y
            if max_bleu > 0.3:
                print(file+'\t'+str(version)+'\t'+str(round(max_bleu, 3))+'\t'+' '.join(x)+'\t'+' '.join(yy))
    #'''
    #print()
