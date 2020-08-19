import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
import sys
from tqdm import tqdm

def get_data(dirpath):
    """
        Given a directory with .csv file names, append then
        into a single dataframe and run train_test split on
        that
    """
    files = [f for f in os.listdir(dirpath) if '.csv' in f]
    dflist = [pd.read_csv(os.path.join(dirpath, f)) for f in files]
    train_df, test_df = train_test_split(pd.concat(dflist), test_size=0.2)
    print('Size of train file: {:d}'.format(len(train_df)))
    print('Size of test file: {:d}'.format(len(test_df)))
    return train_df, test_df

def train_clf(train_df, test_df):
    """
        Given the training data and test data, convert it into
        count vectors, and then tfidf. use the same on the data
        in the testdf
    """
    train_X = list(train_df['Source'])
    l = len(train_X)
    train_y = [0] * l
    train_X.extend(list(train_df['Target']))
    train_y.extend([1] * l)
    
    test_X = list(test_df['Source'])
    l = len(test_X)
    test_y = [0] * l
    test_X.extend(test_df['Target'])
    test_y.extend([1] * l)

    sys.stderr.write('Extracting BOW ...\n')
    sys.stderr.flush()
    count_vect = CountVectorizer(max_features=None, lowercase=False, ngram_range=(1,2), stop_words=None)
    train_X_counts = count_vect.fit_transform(train_X)
    test_X_counts = count_vect.transform(test_X)
    print(train_X_counts.shape)
    print(test_X_counts.shape)
    normalize(train_X_counts, copy=False)
    normalize(test_X_counts, copy=False)
    sys.stderr.write('BOW feature representation done...\n')
    sys.stderr.write('Training MultinomialNB classifier ...\n')
    sys.stderr.flush()
    
    # clf = MultinomialNB()
    clf = MLPClassifier(verbose=1)
    clf.fit(train_X_counts, train_y) 
    sys.stderr.write('Training complete !\n')
    sys.stderr.flush()
    test_y_ = clf.predict_proba(test_X_counts)[:, 1]
    good = 0.0
    bad = 0.0
    for i,(s,t) in tqdm(enumerate(zip(test_y_[::2], test_y_[1::2]))):
        if s < t:
            good += 1.0
        else:
            bad += 1.0
    print(good/(good+bad))
    sys.stdout.flush()
    return count_vect, clf

if __name__ == '__main__':
    dirpath = '/tmp/misunderstanding/'
    train_data, test_data = get_data(dirpath)
    count_vect, clf = train_clf(train_data, test_data)
