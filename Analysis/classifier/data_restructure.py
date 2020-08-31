import io
import os
import pandas as pd

def make_batches(batch_size, samples):
    data = {}
    bdata = []
    for sent in samples:
        bucket = len(sent[0])
        data.setdefault(bucket, [])
        data[bucket].append(sent)
    data = data.values()
    for bucket in data:
        n = max(1, int(len(bucket)/batch_size))
        bdata += [bucket[i::int(n)] for i in range(n)]
    return bdata

def join_data_files(datadir, data_path):
    df = pd.concat([pd.read_csv(os.path.join(datadir, f)) for f in os.listdir(datadir) if '.csv' in f])
    source = [(l, 0) for l in df['Source']]
    target = [(l, 1) for l in df['Target']]
    data = source + target
    df = pd.DataFrame(data, columns=['text', 'label'])
    df.to_csv(os.path.join(data_path), index=False)
    return data_path

def read_train_test_dev(data_path, test_file_path, dev_file_path, batch_size):
    dev_X = []
    test_X = []
    train_X = []
    dev_files = set(open(dev_file_path).read().split())
    test_files = set(open(test_file_path).read().split())
    dev_id, test_id = 0, 0
    with io.open(data_path, encoding='utf-8') as fp:
        for line in fp:
            title, src, tgt = line.strip().split(',')
            src = src.strip().split()
            tgt = tgt.strip().split()
            if len(src) > 150 or len(tgt) > 150:
                continue
            if title in dev_files:
                dev_id += 1
                dev_X.append((src, 0, dev_id))
                dev_id += 1
                dev_X.append((tgt, 1, dev_id))
            elif title in test_files:
                test_id += 1
                test_X.append((src, 0, test_id))
                test_id += 1
                test_X.append((tgt, 1, test_id))
            else:
                train_X.append((src, 0))
                train_X.append((tgt, 1))
    return make_batches(batch_size, train_X), make_batches(batch_size, dev_X), make_batches(batch_size, test_X)

if __name__ == '__main__':
    batch_size = 1000
    datadir = '/mount/projekte/emmy-noether-roth/mist/misunderstanding/csv'
    data_path = 'final_data.csv'
    dev_path = '../Experiments/data/dev_files.txt'
    test_path = '../Experiments/data/test_files.txt'
    data_path = join_data_files(datadir, data_path)
    train_batches, dev_batches, test_batches = read_train_test_dev(data_path, test_path, dev_path, batch_size)
    print(train_batches[0], test_batches[0], dev_batches[0])
