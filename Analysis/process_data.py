import pandas as pd
import stanza
from tqdm import tqdm 
import editdistance

"""
    The idea of this preprocessing is to extract imperative sentences as a source of instructions
    and therefore, the first idea was simply to isolate those sentences which have the first word
    as verbs.
    But, the property of imperative sentences is not that it is the first word is a verb, rather
    it is about the lack of a subject (or a subject like "you") for the main verb
"""


def constructDf(fname):
    data = {
            'File Name': list(),
            'Revision': list(),
            'Source': list(),
            'Target': list()
            }

    with open(fname, 'r') as f:
        for line in f.readlines():
            try:
                [name, rev, src, tgt, _] = line.split('\t')
            except:
                [name, rev, src, tgt] = line.split('\t')
            if abs(len(src.split()) - len(tgt.split())) < 3 and len(src.split()) < 40 and len(tgt.split()) < 40:
                data['File Name'].append(name)
                data['Revision'].append(rev)
                data['Source'].append(src)
                data['Target'].append(tgt)

    df = pd.DataFrame.from_dict(data, orient='columns')
    print('Size of the dataframe: \t' + str(len(df['File Name'])))
    return df

def addposAndDep(df, lim):
    nlp = stanza.Pipeline('en', processors='tokenize, lemma, pos, depparse')
    srctok = list()
    srcpos = list()
    srcdep = list()
    srchead = list()
    tgttok = list()
    tgtpos = list()
    tgtdep = list()
    tgthead = list()
    for ix in tqdm(range(len(df['Revision'][:lim]))):
        src = nlp(df['Source'][ix])
        tgt = nlp(df['Target'][ix])
        srctok.append([word.text for word in src.sentences[0].words])
        tgttok.append([word.text for word in tgt.sentences[0].words])
        srcpos.append([word.xpos for word in src.sentences[0].words])
        tgtpos.append([word.xpos for word in tgt.sentences[0].words])
        srcdep.append([word.deprel for word in src.sentences[0].words])
        tgtdep.append([word.deprel for word in tgt.sentences[0].words])
        srchead.append([word.head for word in src.sentences[0].words])
        tgthead.append([word.head for word in tgt.sentences[0].words])
    data = {
            'File Name': df['File Name'][:lim],
            'Revision': df['Revision'][:lim],
            'Source': df['Source'][:lim],
            'Target': df['Target'][:lim]
            }
    data['SourceTok'] = srctok
    data['TargetTok'] = tgttok
    data['SourcePOS'] = srcpos
    data['TargetPOS'] = tgtpos
    data['SourceDep'] = srcdep
    data['TargetDep'] = tgtdep
    data['SourceHead'] = srchead
    data['TargetHead'] = tgthead
    df = pd.DataFrame.from_dict(data, orient="columns")
    return df

def filterImperative(df):
    siti = 0
    si = 0
    ti = 0
    siti_l = []
    si_l = []
    ti_l = []
    subj_list = ['you', 'one', 'it']
    for ix in tqdm(range(len(df['SourceDep']))):
        try:
            src_root_loc = df['SourceDep'][ix].index('root')
            tgt_root_loc = df['TargetDep'][ix].index('root')
            src_root = df['SourceTok'][ix][src_root_loc].lower()
            tgt_root = df['TargetTok'][ix][tgt_root_loc].lower()

        except:
            continue

        # Condition 0: Sentence root is the word "let"
        if (src_root == 'let') and (tgt_root == 'let'):
            siti += 1
            siti_l.append([df['Source'][ix], df['SourceTok'][ix], df['SourcePOS'][ix], df['SourceDep'][ix], df['SourceHead'][ix],
                           df['Target'][ix], df['TargetTok'][ix], df['TargetPOS'][ix], df['TargetDep'][ix], df['TargetHead'][ix]])

        # Condition 1: Sentence has a root but no `nsubj`
        elif 'nsubj' not in df['SourceDep'][ix] and 'nsubj' not in df['TargetDep'][ix]:
            siti += 1
            siti_l.append([df['Source'][ix], df['SourceTok'][ix], df['SourcePOS'][ix], df['SourceDep'][ix], df['SourceHead'][ix],
                           df['Target'][ix], df['TargetTok'][ix], df['TargetPOS'][ix], df['TargetDep'][ix], df['TargetHead'][ix]])
        else:
            # Condition 2: the `nsubj` in the sentence is not linked to the root verb.
            try:
                src_nsubj_loc = df['SourceDep'][ix].index('nsubj')
                src_nsubj_head = df['SourceTok'][ix][df['SourceHead'][ix][src_nsubj_loc]].lower()
                tgt_nsubj_loc = df['TargetDep'][ix].index('nsubj')
                tgt_nsubj_head = df['TargetTok'][ix][df['TargetHead'][ix][tgt_nsubj_loc]].lower()
            except:
                continue
            if src_nsubj_head not in df['SourceTok'][ix][src_root_loc].lower() and tgt_nsubj_head != df['TargetTok'][ix][tgt_root_loc].lower():
                siti += 1
                siti_l.append([df['Source'][ix], df['SourceTok'][ix], df['SourcePOS'][ix], df['SourceDep'][ix], df['SourceHead'][ix],
                               df['Target'][ix], df['TargetTok'][ix], df['TargetPOS'][ix], df['TargetDep'][ix], df['TargetHead'][ix]])
            # Condition 3: the `nsubj` in the sentence is "one", "you" or "it"
            elif src_nsubj_head in subj_list and tgt_nsubj_head in subj_list:
                siti += 1
                siti_l.append([df['Source'][ix], df['SourceTok'][ix], df['SourcePOS'][ix], df['SourceDep'][ix], df['SourceHead'][ix],
                               df['Target'][ix], df['TargetTok'][ix], df['TargetPOS'][ix], df['TargetDep'][ix], df['TargetHead'][ix]])


    print("Source I Target I: \t" + str(siti))
    siti_df = pd.DataFrame(siti_l, columns=['Source', 'SourceTok', 'SourcePOS', 'SourceDep', 'SourceHead', 
                                            'Target', 'TargetTok', 'TargetPOS', 'TargetDep', 'TargetHead'])
    print(siti_df)
    siti_df.to_csv(path_or_buf='siti.csv', index=True)
    return siti_df

def filterPos(df):
    svtv = 0
    sv = 0
    tv = 0
    svtv_l = []
    sv_l = []
    tv_l = []
    for ix in tqdm(range(len(df['SourcePOS']))):
        if 'V' in df['SourcePOS'][ix][0]:
            if 'V' in df['TargetPOS'][ix][0]:
                svtv_l.append([df['Source'][ix], df['SourceTok'][ix], df['SourcePOS'][ix], df['SourceDep'][ix], 
                               df['Target'][ix], df['TargetTok'][ix], df['TargetPOS'][ix], df['TargetDep'][ix]])
                svtv += 1
            else:
                sv_l.append([df['Source'][ix], df['SourceTok'][ix], df['SourcePOS'][ix], df['SourceDep'][ix], 
                             df['Target'][ix], df['TargetTok'][ix], df['TargetPOS'][ix], df['TargetDep'][ix]])
                sv += 1
        elif 'V' in df['TargetPOS'][ix][0]:
                tv_l.append([df['Source'][ix], df['SourceTok'][ix], df['SourcePOS'][ix], df['SourceDep'][ix], 
                             df['Target'][ix], df['TargetTok'][ix], df['TargetPOS'][ix], df['TargetDep'][ix]])
                tv += 1
            
    print("Source V Target V: \t" + str(svtv))
    print("Source V: \t" + str(sv))
    print("Target V: \t" + str(tv))
    svtv_df = pd.DataFrame(svtv_l, columns=['Source', 'SourceTok', 'SourcePOS', 'SourceDep', 'Target', 'TargetTok', 'TargetPOS', 'TargetDep'])
    sv_df = pd.DataFrame(sv_l, columns=['Source', 'SourceTok', 'SourcePOS', 'SourceDep', 'Target', 'TargetTok', 'TargetPOS', 'TargetDep'])
    tv_df = pd.DataFrame(tv_l, columns=['Source', 'SourceTok', 'SourcePOS', 'SourceDep', 'Target', 'TargetTok', 'TargetPOS', 'TargetDep'])
    print(svtv_df)
    return svtv_df, sv_df, tv_df


def chRoot(fname, df):
    """
        Identify change in the position of the root, the root word itself, or both. 
        Useful when looking at sentences where both the source and target sentence
    """
    cRootPosn = list()
    cRootWord = list()
    cRootWP = list()
    rephrase = list()
    for ix in tqdm(range(len(df['SourceDep']))):
        src_root_posn = df['SourceDep'][ix].index('root')
        tgt_root_posn = df['TargetDep'][ix].index('root')
        try:
            src_root_word = df['SourceTok'][ix][src_root_posn].lower()
            tgt_root_word = df['TargetTok'][ix][tgt_root_posn].lower()
        except:
            print(src_root_posn, tgt_root_posn)
            src_root_word = ''
            tgt_root_word = ''
        
        # Rephrase only
        if editdistance.eval(df['Source'][ix], df['Target'][ix]) > 5 and editdistance.eval(df['Source'][ix], df['Target'][ix]) < 25:
            rephrase.append([df['Source'][ix], df['Target'][ix]])
        if src_root_word != tgt_root_word and editdistance.eval(df['Source'][ix], df['Target'][ix]) > 5 and editdistance.eval(df['Source'][ix], df['Target'][ix]) < 25:
            cRootWord.append([df['Source'][ix], df['Target'][ix]])
        if src_root_posn == tgt_root_posn and src_root_word != tgt_root_word and editdistance.eval(df['Source'][ix], df['Target'][ix]) > 3 and editdistance.eval(df['Source'][ix], df['Target'][ix]) < 10:
            cRootWP.append([df['Source'][ix], df['Target'][ix]])
    
    rephrase_df = pd.DataFrame(rephrase, columns=['Source',  'Target'])
    cRootWord_df = pd.DataFrame(cRootWord, columns=['Source',  'Target'])
    cRootWP_df = pd.DataFrame(cRootWP, columns=['Source', 'Target']) 
    
    rephrase_df.to_csv(path_or_buf=fname + 'rephrase.csv', index=True)
    cRootWord_df.to_csv(path_or_buf=fname + 'word.csv', index=True)
    cRootWP_df.to_csv(path_or_buf=fname + 'wp.csv', index=True)
    return rephrase_df, cRootWord_df, cRootWP_df

def rephrase(fname, df):
    """
        Rephrases based on just source and target based on the difference in the text and POS
    """
    pass

if __name__ == '__main__':
    fname = '/tmp/misunderstanding/typo_filtered_revisions.txt'
    # fname = './both0s.txt'
    df = constructDf(fname)
    lim = 100000
    df = addposAndDep(df, lim)
    siti_df = filterImperative(df)
    # svtv_df, sv_df, tv_df = filterPos(df)
    # chRoot('chRoot_svtv', svtv_df)
    # chRoot('chRoot_sv', sv_df)
    # chRoot('chRoot_tv', tv_df)
    # rephrase('rephrase_svtv', svtv_df)
    # rephrase('rephrase_sv', sv_df)
    # rephrase('rephrase_tv', tv_df)
    chRoot('chRoot_siti', siti_df)
