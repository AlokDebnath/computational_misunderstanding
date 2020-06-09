from tqdm import tqdm 

def read_dataset(fpath):
    data = list()
    print('Reading data...')
    with open(fpath, 'r') as f:
        data = [[l.strip('\n').strip() for l in line.split('\t') if 'Begin_' not in l and 'Inside_' not in l] for line in tqdm(f.readlines())]
        print(data[0])
    f.close()
    return data

def write_dataset(writepath, data, done, readpath=None):
    if readpath is not None:
        f = open(readpath, 'r')
        lines = f.readlines()
        f.close()
    else:
        lines = [d[0].strip() for d in data if d not in done]
    print('Creating ' + writepath)
    with open(writepath, 'w+') as g:
        for d in tqdm(data):
            for line in lines:
                if d[0].strip() == line.strip('\n'):
                    g.write(d[1] + '\t' + d[2] + '\n')
                    done.append(d[0])
    return data, done

data = read_dataset('./wikiHow_revisions_corpus.txt')
data, done = write_dataset('wikiHow_revisions_corpus_test.txt', data, [], readpath = './test_files.txt')
data, done = write_dataset('wikiHow_revisions_corpus_dev.txt', data, done, readpath = './dev_files.txt')
data, done = write_dataset('wikiHow_revisions_corpus_train.txt', data, done)
assert len(data) == len(done)


