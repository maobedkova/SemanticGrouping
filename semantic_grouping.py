import pandas as pd
import numpy
import re
import os
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import pickle

'''Create dataframe from file'''
def create_contexts():
    ozhegov = pd.read_csv('OZHEGOV.txt', sep='|', header=0)
    df = pd.DataFrame(ozhegov, columns=['VOCAB', 'DEF'])
    df['CONTEXT'] = df['VOCAB'] + ' ' + df['DEF']
    with open('ozheg_for_mystem.txt', 'w', encoding='utf-8') as f:
        for line in numpy.array(df['CONTEXT']):
            if not isinstance(line, float):
                f.write(re.sub('[,\)\(;:\]\[]', '', ' '.join(line.split())) + '\n')

'''Lemmatization'''
def mystem_parsing(a, b, c):
    os.system('mystem ' + a + ' ' + b + ' ' + c)

'''Clear data from punctuation and not Russian symbols'''
def clear_data():
    with open('output_mystem.txt') as f:
        text = [re.sub('(-{.+?}| - |[0-9=><{}?a-zA-Z+/\"])', '', line) for line in f]
    with open('clear_contexts.txt', 'w') as w:
        w.write(re.sub(' +', ' ', ''.join(text)))

'''Remove stop words (all pos that are not adjective, adverb, noun, verb, comparative)'''
def remove_stopwords():
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

'''Select only high frequency words'''
def preprocessing():
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

'''Common tokenization'''
def tokenize(text):
    new_text = text.split()
    return new_text

'''Tf-idf vectorization'''
def vectorization():
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

'''Train algorithm k-means'''
def modelling(df, tfidfed):
    kmeans = KMeans(n_clusters=1500, random_state=0, init='k-means++').fit(tfidfed)
    df['LAB'] = kmeans.labels_

    df.to_csv('new.csv')

    joblib.dump(kmeans, 'kmeans.pkl')

'''Prediction'''
def prediction(req, tfidf, df):
    kmeans = joblib.load('kmeans.pkl')

    control = tfidf.transform(req)
    preds = kmeans.predict(control)

    print (preds)

if __name__ == '__main__':
    # create_contexts()
    # mystem_parsing('-cld', 'ozheg_for_mystem.txt', 'output_mystem.txt')
    # clear_data()
    # mystem_parsing('-cldi', 'clear_contexts.txt', 'pos_contexts.txt')
    # remove_stopwords()
    # preprocessing()
    tfidf, df, tfidfed = vectorization()
    # modelling(df, tfidfed)
    prediction(['говорить'], tfidf=tfidf, df=df)

'''
Попробовать иерархическую кластеризацию.

Cверху нам потом понадобится, фильтрация по морфологии.

дать просто слово, без характеристики.

Интерактивность.
'''