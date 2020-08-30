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
            if len(src.split()) > 5 and len(tgt.split()) > 5 and len(src.split()) < 50 and len(tgt.split()) < 50:
                data['File Name'].append(name)
                data['Revision'].append(rev)
                data['Source'].append(src)
                data['Target'].append(tgt)

    df = pd.DataFrame.from_dict(data, orient='columns')
    print('Size of the dataframe: \t' + str(len(df['File Name'])))
    return df

def addposAndDep(df, start, lim):
    nlp = stanza.Pipeline('en', processors='tokenize, lemma, pos, depparse')
    indlist = list()
    srctok = list()
    srcpos = list()
    srcdep = list()
    srchead = list()
    tgttok = list()
    tgtpos = list()
    tgtdep = list()
    tgthead = list()
    ix = start
    for ix in tqdm(range(len(df['Revision'][start:lim]))):
        indlist.append(start + ix)
        src = nlp(df['Source'][start + ix])
        tgt = nlp(df['Target'][start + ix])
        srctok.append([word.text for word in src.sentences[0].words])
        tgttok.append([word.text for word in tgt.sentences[0].words])
        srcpos.append([word.xpos for word in src.sentences[0].words])
        tgtpos.append([word.xpos for word in tgt.sentences[0].words])
        srcdep.append([word.deprel for word in src.sentences[0].words])
        tgtdep.append([word.deprel for word in tgt.sentences[0].words])
        srchead.append([word.head for word in src.sentences[0].words])
        tgthead.append([word.head for word in tgt.sentences[0].words])

    data = {
            'File Name': df['File Name'][start:lim],
            'Revision': df['Revision'][start:lim],
            'Source': df['Source'][start:lim],
            'Target': df['Target'][start:lim]
            }
    data['SourceTok'] = srctok
    data['TargetTok'] = tgttok
    data['SourcePOS'] = srcpos
    data['TargetPOS'] = tgtpos
    data['SourceDep'] = srcdep
    data['TargetDep'] = tgtdep
    data['SourceHead'] = srchead
    data['TargetHead'] = tgthead
    data['index'] = indlist
    o_df = pd.DataFrame.from_dict(data, orient="columns")
    print(o_df)
    return o_df

def filterImperative(df):
    siti = 0
    siti_l = []
    subj_list = ['you', 'one', 'it']
    for ix in df['index']:
        try:
            src_root_loc = df['SourceDep'][ix].index('root')
            tgt_root_loc = df['TargetDep'][ix].index('root')
            src_root = df['SourceTok'][ix][src_root_loc].lower()
            tgt_root = df['TargetTok'][ix][tgt_root_loc].lower()

        except Exception:
            continue

        # Condition 0: Sentence root is the word "let"
        if 'let' in df['Source'][ix]  and 'let' in df['Target'][ix]:
            siti += 1
            siti_l.append([df['File Name'][ix],
                           df['Source'][ix], df['SourceTok'][ix], df['SourcePOS'][ix], df['SourceDep'][ix], df['SourceHead'][ix],
                           df['Target'][ix], df['TargetTok'][ix], df['TargetPOS'][ix], df['TargetDep'][ix], df['TargetHead'][ix]])

        # Condition 1: Sentence has a root but no `nsubj`
        elif 'nsubj' not in df['SourceDep'][ix] and 'nsubj' not in df['TargetDep'][ix]:
            siti += 1
            siti_l.append([df['File Name'][ix],
                           df['Source'][ix], df['SourceTok'][ix], df['SourcePOS'][ix], df['SourceDep'][ix], df['SourceHead'][ix],
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
                siti_l.append([df['File Name'][ix],
                               df['Source'][ix], df['SourceTok'][ix], df['SourcePOS'][ix], df['SourceDep'][ix], df['SourceHead'][ix],
                               df['Target'][ix], df['TargetTok'][ix], df['TargetPOS'][ix], df['TargetDep'][ix], df['TargetHead'][ix]])
            # Condition 3: the `nsubj` in the sentence is "one", "you" or "it"
            elif src_nsubj_head in subj_list and tgt_nsubj_head in subj_list:
                siti += 1
                siti_l.append([df['File Name'][ix],
                               df['Source'][ix], df['SourceTok'][ix], df['SourcePOS'][ix], df['SourceDep'][ix], df['SourceHead'][ix],
                               df['Target'][ix], df['TargetTok'][ix], df['TargetPOS'][ix], df['TargetDep'][ix], df['TargetHead'][ix]])


    print("Source I Target I: \t" + str(siti))
    siti_df = pd.DataFrame(siti_l, columns=['File Name',
                                            'Source', 'SourceTok', 'SourcePOS', 'SourceDep', 'SourceHead', 
                                            'Target', 'TargetTok', 'TargetPOS', 'TargetDep', 'TargetHead'])
    print(siti_df)
    return siti_df

def chRoot(fname, df):
    """
        Identify change in the position of the root, the root word itself, or both. 
        Useful when looking at sentences where both the source and target sentence
    """
    cRootWP = list()
    for ix in tqdm(range(len(df['Source']))):
        src_root_posn = df['SourceDep'][ix].index('root')
        tgt_root_posn = df['TargetDep'][ix].index('root')
        src_root_word = df['SourceTok'][ix][src_root_posn].lower()
        tgt_root_word = df['TargetTok'][ix][tgt_root_posn].lower()
        src_root_pos = df['SourcePOS'][ix][src_root_posn]
        tgt_root_pos = df['TargetPOS'][ix][tgt_root_posn]
        
        # Rephrase only
        if src_root_word not in tgt_root_word and tgt_root_word not in src_root_word and editdistance.eval(df['Source'][ix], df['Target'][ix]) > 4 and editdistance.eval(df['Source'][ix], df['Target'][ix]) < 10 and 'V' in src_root_pos and 'V' in tgt_root_pos:
            # print(src_root_word, tgt_root_word)
            cRootWP.append([df['File Name'][ix], df['Source'][ix], df['Target'][ix], src_root_word, tgt_root_word])
    
    cRootWP_df = pd.DataFrame(cRootWP, columns=['File Name', 'Source', 'Target', 'Src Root', 'Tgt Root']) 
    cRootWP_df.to_csv(path_or_buf=fname + 'wp.csv', index=True)
    return cRootWP_df


if __name__ == '__main__':
    fname = '/mount/projekte/emmy-noether-roth/mist/misunderstanding/typo_filtered_revisions.txt'
    # fname = './both0s.txt'
    df = constructDf(fname)
    for ix in tqdm(range(int(len(df['Source'])/1000) - 1)):
        start = ix * 1000
        lim = (ix + 1) * 1000
        o_df = addposAndDep(df, start, lim)
        siti_df = filterImperative(o_df)
        chRoot('/tmp/misunderstanding/chRoot_siti_' + str(ix), siti_df)

    start = ix * int(df.size/1000)
    lim = df.size
    # ix = 1
    # start = ix * 1000
    # lim = (ix + 1) * 1000
    o_df = addposAndDep(df, start, lim)
    siti_df = filterImperative(o_df)
    chRoot('/tmp/misunderstanding/chRoot_siti_' + str(ix), siti_df)
