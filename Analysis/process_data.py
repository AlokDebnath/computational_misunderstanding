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
        srcdep.append([(word.deprel, word.head) for word in src.sentences[0].words])
        tgtdep.append([(word.deprel, word.head) for word in tgt.sentences[0].words])
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
    f = open('SrcVTgtVi.txt', 'w+')
    g = open('SrcVi.txt', 'w+')
    h = open('TgtVi.txt', 'w+')
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
                f.write(str(df['Source'][ix]) + '\t'
                        + str(df['SourcePOS'][ix]) + '\t'   
                        + str(df['SourceDep'][ix]) + '\n'
                        + str(df['Target'][ix]) + '\t'
                        + str(df['TargetPOS'][ix]) + '\t'
                        + str(df['TargetDep'][ix]) + '\n\n')
                svtv += 1
            else:
                sv_l.append([df['Source'][ix], df['SourcePOS'][ix], df['SourceDep'][ix], df['Target'][ix], df['TargetPOS'][ix], df['TargetDep'][ix]])
                g.write(str(df['Source'][ix]) + '\t'
                        + str(df['SourcePOS'][ix]) + '\t'   
                        + str(df['SourceDep'][ix]) + '\n'
                        + str(df['Target'][ix]) + '\t'
                        + str(df['TargetPOS'][ix]) + '\t'
                        + str(df['TargetDep'][ix]) + '\n\n')
                sv += 1
        elif 'V' in df['TargetPOS'][ix][0]:
                tv_l.append([df['Source'][ix], df['SourcePOS'][ix], df['SourceDep'][ix], df['Target'][ix], df['TargetPOS'][ix], df['TargetDep'][ix]])
                h.write(str(df['Source'][ix]) + '\t'
                        + str(df['SourcePOS'][ix]) + '\t'   
                        + str(df['SourceDep'][ix]) + '\n'
                        + str(df['Target'][ix]) + '\t'
                        + str(df['TargetPOS'][ix]) + '\t'
                        + str(df['TargetDep'][ix]) + '\n\n')
                tv += 1
            
    f.close()
    g.close()
    h.close()
    print("Source V Target V: \t" + str(svtv))
    print("Source V: \t" + str(sv))
    print("Target V: \t" + str(tv))
    svtv_df = pd.DataFrame(svtv_l, columns=['Source', 'SourcePOS', 'SourceDep', 'Target', 'TargetPOS', 'TargetDep'])
    sv_df = pd.DataFrame(sv_l, columns=['Source', 'SourcePOS', 'SourceDep', 'Target', 'TargetPOS', 'TargetDep'])
    tv_df = pd.DataFrame(tv_l, columns=['Source', 'SourcePOS', 'SourceDep', 'Target', 'TargetPOS', 'TargetDep'])
    return svtv_df, sv_df, tv_df

def edit_distance(str1, str2, l1, l2):
    if l1 == 0:
        return l2
    if l2 == 0:
        return l1
    if str1[l1 - 1] == str2[l2 - 1]:
        return edit_distance(str1, str2, l1-1, l2-1)

    return 1 + min(edit_distance(str1, str2, l1, l2 - 1),
                   edit_distance(str1, str2, l1 - 1, l2),
                   edit_distance(str1, str2, l1 - 1, l2 - 1)
                   )

def chRoot(df):
    cRoot = list()
    for ix in tqdm(range(len(df['SourceDep']))):
        if 'root' in df['SourceDep'][ix][0] and 'root' not in df['TargetDep'][ix][0]:
            cRoot.append([df['Source'][ix], df['SourcePOS'][ix], df['SourceDep'][ix], df['Target'][ix], df['TargetPOS'][ix], df['TargetDep'][ix]])
    cRoot_df = pd.DataFrame(cRoot,  columns=['Source', 'SourcePOS', 'SourceDep', 'Target', 'TargetPOS', 'TargetDep'])
    cRoot_df.to_csv(path_or_buf='./chroot.csv', index=True)
    return cRoot_df

def rephrase(df):
    rephrase = list()
    for ix in tqdm(range(len(df['Source']))):
        if editdistance.eval(df['Source'][ix], df['Target'][ix]) < 5 and editdistance.eval(df['Source'][ix], df['Target'][ix]) > 2:
            rephrase.append([df['Source'][ix], df['SourcePOS'][ix], df['SourceDep'][ix], df['Target'][ix], df['TargetPOS'][ix], df['TargetDep'][ix]])
    rephrase_df = pd.DataFrame(rephrase,  columns=['Source', 'SourcePOS', 'SourceDep', 'Target', 'TargetPOS', 'TargetDep'])
    rephrase_df.to_csv(path_or_buf='./rephrase.csv', index=True)
    return rephrase_df

if __name__ == '__main__':
    fname = '/tmp/misunderstanding/typo_filtered_revisions.txt'
    df = constructDf(fname)
    lim = 1000
    df = addposAndDep(df, lim)
    svtv_df, sv_df, tv_df = filterPos(df)
    cRoot_df = chRoot(sv_df)
    print(cRoot_df)
    rephrase_df = rephrase(svtv_df)
    print(rephrase_df)
