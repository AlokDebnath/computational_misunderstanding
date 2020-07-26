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
            [name, rev, src, tgt, _] = line.split('\t')
            if abs(len(src.split()) - len(tgt.split())) < 3:
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
    f = open('./SrcVTgtV.txt', 'w+')
    g = open('./SrcV.txt', 'w+')
    h = open('./TgtV.txt', 'w+')
    for ix in tqdm(range(len(df['SourcePOS']))):
        if 'V' in df['SourcePOS'][ix][0]:
            if 'V' in df['TargetPOS'][ix][0]:
                f.write(str(df['Source'][ix]) + '\t'
                        + str(df['SourcePOS'][ix]) + '\t'   
                        + str(df['SourceDep'][ix]) + '\n'
                        + str(df['Target'][ix]) + '\t'
                        + str(df['TargetPOS'][ix]) + '\t'
                        + str(df['TargetDep'][ix]) + '\n\n')
            else:
                g.write(str(df['Source'][ix]) + '\t'
                        + str(df['SourcePOS'][ix]) + '\t'   
                        + str(df['SourceDep'][ix]) + '\n'
                        + str(df['Target'][ix]) + '\t'
                        + str(df['TargetPOS'][ix]) + '\t'
                        + str(df['TargetDep'][ix]) + '\n\n')
        elif 'V' in df['TargetPOS'][ix][0]:
                h.write(str(df['Source'][ix]) + '\t'
                        + str(df['SourcePOS'][ix]) + '\t'   
                        + str(df['SourceDep'][ix]) + '\n'
                        + str(df['Target'][ix]) + '\t'
                        + str(df['TargetPOS'][ix]) + '\t'
                        + str(df['TargetDep'][ix]) + '\n\n')
            
    f.close()
    g.close()
    h.close()
    return

if __name__ == '__main__':
    fname = '/tmp/misunderstanding/typo_filtered_revisions.txt'
    df = constructDf(fname)
    lim = 50000
    df = addposAndDep(df, lim)
    for i in range(len(df['TargetPOS'])):
        print(df['TargetPOS'][i][0])
    filterPos(df)
