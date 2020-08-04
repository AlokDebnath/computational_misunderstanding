import pandas as pd
import stanza
from tqdm import tqdm 
import editdistance

def constructDf(fname):
    data = {
            'File Name': list(),
            'Revision': list(),
            'Source': list(),
            'Target': list()
            }

    with open(fname, 'r') as f:
        for line in f.readlines():
            [name, rev, src, tgt, _] = line.split('\t')
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
    srcpos = list()
    srcdep = list()
    tgtpos = list()
    tgtdep = list()
    for ix in tqdm(range(len(df['Revision'][:lim]))):
        src = nlp(df['Source'][ix])
        tgt = nlp(df['Target'][ix])
        srcpos.append([word.xpos for word in src.sentences[0].words])
        tgtpos.append([word.xpos for word in tgt.sentences[0].words])
        srcdep.append([word.deprel for word in src.sentences[0].words])
        tgtdep.append([word.deprel for word in tgt.sentences[0].words])
    data = {
            'File Name': df['File Name'][:lim],
            'Revision': df['Revision'][:lim],
            'Source': df['Source'][:lim],
            'Target': df['Target'][:lim]
            }
    data['SourcePOS'] = srcpos
    data['TargetPOS'] = tgtpos
    data['SourceDep'] = srcdep
    data['TargetDep'] = tgtdep
    df = pd.DataFrame.from_dict(data, orient="columns")
    return df

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
                svtv_l.append([df['Source'][ix], df['SourcePOS'][ix], df['SourceDep'][ix], df['Target'][ix], df['TargetPOS'][ix], df['TargetDep'][ix]])
                svtv += 1
            else:
                sv_l.append([df['Source'][ix], df['SourcePOS'][ix], df['SourceDep'][ix], df['Target'][ix], df['TargetPOS'][ix], df['TargetDep'][ix]])
                sv += 1
        elif 'V' in df['TargetPOS'][ix][0]:
                tv_l.append([df['Source'][ix], df['SourcePOS'][ix], df['SourceDep'][ix], df['Target'][ix], df['TargetPOS'][ix], df['TargetDep'][ix]])
                tv += 1
            
    print("Source V Target V: \t" + str(svtv))
    print("Source V: \t" + str(sv))
    print("Target V: \t" + str(tv))
    svtv_df = pd.DataFrame(svtv_l, columns=['Source', 'SourcePOS', 'SourceDep', 'Target', 'TargetPOS', 'TargetDep'])
    sv_df = pd.DataFrame(sv_l, columns=['Source', 'SourcePOS', 'SourceDep', 'Target', 'TargetPOS', 'TargetDep'])
    tv_df = pd.DataFrame(tv_l, columns=['Source', 'SourcePOS', 'SourceDep', 'Target', 'TargetPOS', 'TargetDep'])
    print(svtv_df)
    return svtv_df, sv_df, tv_df


def chRoot(fname, df):
    """
        Identify change in the position of the root, the root word itself, or both. Useful when looking at sentences where both the source and target sentence
    """
    cRootPosn = list()
    cRootWord = list()
    cRootWP = list()
    for ix in tqdm(range(len(df['SourceDep']))):
        src_root_posn = df['SourceDep'][ix].index('root')
        tgt_root_posn = df['TargetDep'][ix].index('root')
        src_root_word = df['Source'][ix].split()[src_root_posn]
        tgt_root_word = df['Target'][ix].split()[tgt_root_posn]
        if src_root_posn != tgt_root_posn:
            cRootPosn.append([df['Source'][ix], df['Target'][ix]])
        if src_root_word != tgt_root_word:
            cRootWord.append([df['Source'][ix], df['Target'][ix]])
        if src_root_posn == tgt_root_posn and src_root_word != tgt_root_word:
            cRootWP.append([df['Source'][ix], df['Target'][ix]])
    
    cRootPosn_df = pd.DataFrame(cRootPosn, columns=['Source',  'Target'])
    cRootWord_df = pd.DataFrame(cRootWord, columns=['Source',  'Target'])
    cRootWP_df = pd.DataFrame(cRootWP, columns=['Source', 'Target']) 
    cRootPosn_df.to_csv(path_or_buf=fname + 'posn.csv', index=True)
    cRootWord_df.to_csv(path_or_buf=fname + 'word.csv', index=True)
    cRootWP_df.to_csv(path_or_buf=fname + 'wp.csv', index=True)
    return cRootPosn_df, cRootWord_df, cRootWP_df

def rephrase(fname, df):
    """
        Rephrases based on just source and target based on the difference in the text and POS
    """
    rephraseStr = list()
    rephrasePOS = list()
    rephraseDep = list()

    for ix in tqdm(range(len(df['Source']))):
        if editdistance.eval(df['Source'][ix], df['Target'][ix]) < 10 and editdistance.eval(df['Source'][ix], df['Target'][ix]) > 4:
            rephraseStr.append([df['Source'][ix], df['Target'][ix]])
        if editdistance.eval(' '.join(df['SourcePOS'][ix]), ' '.join(df['TargetPOS'][ix])) < 10 and editdistance.eval(' '.join(df['SourcePOS'][ix]), ' '.join(df['TargetPOS'][ix])) > 4:
            rephrasePOS.append([df['Source'][ix], df['Target'][ix]])
        if editdistance.eval(' '.join(df['SourceDep'][ix]), ' '.join(df['TargetDep'][ix])) < 10 and editdistance.eval(' '.join(df['SourceDep'][ix]), ' '.join(df['TargetDep'][ix])) > 4:
            rephraseDep.append([df['Source'][ix], df['Target'][ix]])

    rephrase_str_df = pd.DataFrame(rephraseStr,  columns=['Source', 'Target'])
    rephrase_str_df.to_csv(path_or_buf=fname + '_str.csv', index=True)
    rephrase_pos_df = pd.DataFrame(rephrasePOS,  columns=['Source', 'Target'])
    rephrase_pos_df.to_csv(path_or_buf=fname + '_POS.csv', index=True)
    rephrase_dep_df = pd.DataFrame(rephraseDep,  columns=['Source', 'Target'])
    rephrase_dep_df.to_csv(path_or_buf=fname + '_Dep.csv', index=True)
    rephrase_df = pd.merge(rephrase_str_df, rephrase_dep_df, how="inner", on='Source')
    rephrase_df.to_csv(path_or_buf=fname + '.csv', index=True)
    return rephrase_df

if __name__ == '__main__':
    fname = '/tmp/misunderstanding/typo_filtered_revisions.txt'
    df = constructDf(fname)
    lim = 100000
    df = addposAndDep(df, lim)
    svtv_df, sv_df, tv_df = filterPos(df)
    cRootPosn_df, cRootWord_df, cRootWP_df = chRoot('chRoot_svtv', svtv_df)
    rephrase_df = rephrase('rephrase_svtv', svtv_df)
    rephrase_df = rephrase('rephrase_sv', sv_df)
