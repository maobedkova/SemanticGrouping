# codecs: utf-8

__author__ = "mobedkova"

import pandas as pd
import numpy
import re
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.externals import joblib

def create_contexts():
    """Create dataframe from file"""
    ozhegov = pd.read_csv('OZHEGOV.txt', sep='|', header=0)
    df = pd.DataFrame(ozhegov, columns=['VOCAB', 'DEF'])
    df['CONTEXT'] = df['VOCAB'] + ' ' + df['DEF']
    with open('ozheg_for_mystem.txt', 'w', encoding='utf-8') as f:
        for line in numpy.array(df['CONTEXT']):
            if not isinstance(line, float):
                f.write(re.sub('[,\)\(;:\]\[]', '', ' '.join(line.split())) + '\n')

def lemmatize(a, b, c):
    """Lemmatization"""
    os.system('mystem ' + a + ' ' + b + ' ' + c)

def clear_data():
    """Clear data from punctuation and not Russian symbols"""
    with open('output_mystem.txt') as f:
        text = [re.sub('(-{.+?}| - |[0-9=><{}?a-zA-Z+/\"])', '', line) for line in f]
    with open('clear_contexts.txt', 'w') as w:
        w.write(re.sub(' +', ' ', ''.join(text)))

def remove_stopwords():
    """Remove stop words (all pos that are not adjective, adverb, noun, verb, comparative)"""
    ok = ['A', 'ADV', 'COM', 'S', 'V']
    text = ''
    with open('pos_contexts.txt', 'r', encoding='utf-8') as f:
        for line in f:
            context = re.findall('{([-\w]+)=([A-Z]+)(?:.+?)}', line)
            try:
                text += context[0][0] + ' '
                text += ' '.join([c[0] for c in context[1:] if c[1] in ok or c[0] == 'не']) + '\n'
            except:
                pass
    with open('contexts.txt', 'w', encoding='utf-8') as w:
        w.write(text.strip())

def preprocess():
    """Select only high frequency words"""
    defs = []
    f = open('contexts.txt', 'r', encoding='utf-8')
    for line in f:
        words = line.split()
        defs += words[1:]

    d = {}
    for el in defs:
        if el in d:
            d[el] += 1
        else:
            d[el] = 1

    new_defs = {}
    for el in d:
        if d[el] > 3:
            new_defs[el] = d[el]

    print (len(new_defs))

    f = open('contexts.txt', 'r', encoding='utf-8')
    w = open('vectores.csv', 'a', encoding='utf-8')

    for line in f:
        arr = []
        words = line.split()
        for el in list(new_defs):
            if el in words[1:]:
                arr.append('1')
            else:
                arr.append('0')
        w.write(';'.join(arr) + '\n')

def tokenize(text):
    """Common tokenization"""
    new_text = text.split()
    return new_text

def vectorize():
    """Tf-idf vectorization"""
    with open('contexts.txt', 'r', encoding='utf-8') as f:
        words = []
        contexts = []
        both = []
        for line in f:
            splitted = line.split()
            words.append(splitted[0])
            contexts.append(' '.join(splitted[1:]))
            both.append(line.strip())

    df = pd.DataFrame([], columns=['VOC', 'DEF'])
    df['VOC'] = words
    df['DEF'] = contexts
    df['BOTH'] = both

    # bow = CountVectorizer(tokenizer=tokenize)
    # bowed = bow.fit_transform(df['DEF'])
    # control_bowed = bow.transform(control)

    tfidf = TfidfVectorizer(tokenizer=tokenize)
    tfidfed = tfidf.fit_transform(df['BOTH'])

    return tfidf, df, tfidfed

def train(df, tfidfed):
    """Train algorithm k-means"""
    kmeans = KMeans(n_clusters=1500, random_state=0, init='k-means++').fit(tfidfed)
    df['LAB'] = kmeans.labels_

    df.to_csv('new.csv')

    joblib.dump(kmeans, 'kmeans.pkl')

def predict(req, tfidf, df):
    """Prediction"""
    kmeans = joblib.load('kmeans.pkl')

    control = tfidf.transform(req)
    preds = kmeans.predict(control)

    print (preds)

if __name__ == '__main__':
    # create_contexts()
    # lemmatize('-cld', 'ozheg_for_mystem.txt', 'output_mystem.txt')
    # clear_data()
    # lemmatize('-cldi', 'clear_contexts.txt', 'pos_contexts.txt')
    # remove_stopwords()
    # preprocess()
    tfidf, df, tfidfed = vectorize()
    # train(df, tfidfed)
    word = ['говорить']
    predict(word, tfidf=tfidf, df=df)