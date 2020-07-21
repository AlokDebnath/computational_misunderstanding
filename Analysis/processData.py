import pandas as pd
import stanza
from tqdm import tqdm 

def constructDf(fname):
    data = {
            'File Name': list(),
            'Revision': list(),
            'Source': list(),
            'Target': list()
            }

    with open(fname, 'r') as f:
        for line in f.readlines():
            [name, rev, src, tgt] = line.split('\t')
            data['File Name'].append(name)
            data['Revision'].append(rev)
            data['Source'].append(src)
            data['Target'].append(tgt)

    df = pd.DataFrame.from_dict(data, orient='columns')
    return df

def addposAndDep(df):
    nlp = stanza.Pipeline('en', processors='tokenize, lemma, pos, depparse')
    srcpos = list()
    srcdep = list()
    tgtpos = list()
    tgtdep = list()
    for ix in tqdm(range(len(df['Revision'][:10000]))):
        src = nlp(df['Source'][ix])
        tgt = nlp(df['Target'][ix])
        srcpos.append([word.xpos for word in src.sentences[0].words])
        tgtpos.append([word.xpos for word in tgt.sentences[0].words])
        srcdep.append([word.deprel for word in src.sentences[0].words])
        tgtdep.append([word.deprel for word in tgt.sentences[0].words])
    data = {
            'File Name': df['File Name'][:10000],
            'Revision': df['Revision'][:10000],
            'Source': df['Source'][:10000],
            'Target': df['Target'][:10000]
            }
    data['SourcePOS'] = srcpos
    data['TargetPOS'] = tgtpos
    data['SourceDep'] = srcdep
    data['TargetDep'] = tgtdep
    df = pd.DataFrame.from_dict(data, orient="columns")
    return df

def filterPos(df):
    f = open('./SrcVTgtV.txt', 'w+')
    g = open('./SrcV.txt', 'w+')
    h = open('./TgtV.txt', 'w+')
    for ix in tqdm(range(len(df['SourcePOS']))):
        if 'V' in df['SourcePOS'][ix][0]:
            if 'V' in df['TargetPOS'][ix][0]:
                f.write(str(df['Source'][ix]) + '\t'
                        + str(df['SourcePOS'][ix]) + '\t'   
                        + str(df['SourceDep'][ix]) + '\t'
                        + str(df['Target'][ix]) + '\t'
                        + str(df['TargetPOS'][ix]) + '\t'
                        + str(df['TargetDep'][ix]) + '\n')
            else:
                g.write(str(df['Source'][ix]) + '\t'
                        + str(df['SourcePOS'][ix]) + '\t'   
                        + str(df['SourceDep'][ix]) + '\t'
                        + str(df['Target'][ix]) + '\t'
                        + str(df['TargetPOS'][ix]) + '\t'
                        + str(df['TargetDep'][ix]) + '\n')
        elif 'V' in df['TargetPOS'][ix][0]:
                h.write(str(df['Source'][ix]) + '\t'
                        + str(df['SourcePOS'][ix]) + '\t'   
                        + str(df['SourceDep'][ix]) + '\t'
                        + str(df['Target'][ix]) + '\t'
                        + str(df['TargetPOS'][ix]) + '\t'
                        + str(df['TargetDep'][ix]) + '\n')
            
    f.close()
    g.close()
    h.close()
    return

if __name__ == '__main__':
    fname = '/tmp/misunderstanding/wikiHow_revisions_corpus.txt'
    df = constructDf(fname)
    df = addposAndDep(df)
    for i in range(len(df['TargetPOS'])):
        print(df['TargetPOS'][i][0])
    filterPos(df)
