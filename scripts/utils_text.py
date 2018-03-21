import itertools
import random
import re
import time

import gensim
import nltk
import numpy as np
import pandas as pd
import sklearn
from gensim import corpora
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from nltk import ngrams, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import *
from nltk.tag import AffixTagger
from scipy import sparse
from scipy.stats import kurtosis, skew
from sklearn.decomposition import (NMF, PCA, LatentDirichletAllocation,
                                   TruncatedSVD)
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.preprocessing import (MinMaxScaler, Normalizer,
                                   PolynomialFeatures, StandardScaler)

seed = 1337

lemmatizer = WordNetLemmatizer()
stemmer = snowball.SnowballStemmer('english')
stopwords_eng = stopwords.words('english')
words = re.compile(r"\w+", re.I)
# model = KeyedVectors.load_word2vec_format('../data/embeddings/wiki.en',
#                                           binary=False)


def lowercase(df2):
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i in text_feats:
        df[i] = df[i].str.lower()
    return df


def unidecode(df2):
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i in text_feats:
        df[i] = df[i].str.encode('ascii', 'ignore')
    return df


def remove_nonalpha(df2):
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i in text_feats:
        df[i] = df[i].str.replace('\W+', ' ')
    return df


def repair_words(df2):
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i in text_feats:
        df[i] = df[i].apply(lambda x: (''.join(''.join(s)[:2]
                                               for _, s in itertools.groupby(x))))
    return df


def concat_words(df2):
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i in text_feats:
        df[i] = df[i].apply(lambda x: (' '.join(i for i in x)))
    return df


def tokenize(df2):
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i in text_feats:
        df[i] = df[i].apply(lambda x: word_tokenize(x))
    return df


def ngram(df2, n):
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i in text_feats:
        df[i] = df[i].apply(lambda x: [i for i in ngrams(word_tokenize(x), n)])
    return df


def skipgram(df2, ngram_n, skip_n):
    def random_sample(words_list, skip_n):
        return [words_list[i] for i in sorted(random.sample(range(len(words_list)), skip_n))]

    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i in text_feats:
        df[i] = df[i].apply(
            lambda x: [i for i in ngrams(word_tokenize(x), ngram_n)])
        df[i] = df[i].apply(lambda x: random_sample(x, skip_n))
    return df


def chargram(df2, n):
    def chargram_generate(string, n):
        return [string[i:i + n] for i in range(len(string) - n + 1)]
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i in text_feats:
        df[i] = df[i].apply(lambda x: [i for i in chargram_generate(x, 3)])
    return df


def remove_stops(df2, stopwords):
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i in text_feats:
        df[i] = df[i].apply(
            lambda x: [i for i in word_tokenize(x) if i not in stopwords])
    return df


def remove_extremes(df2, stopwords, min_count=3, max_frequency=0.75):
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i in text_feats:
        df[i] = df[i].apply(
            lambda x: [i for i in word_tokenize(x) if i not in stopwords])
    tokenized = []
    for i in text_feats:
        tokenized += df[i].tolist()
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=min_count, no_above=max_frequency)
    dictionary.compactify()
    df = df2.copy()
    for i in text_feats:
        df[i] = df[i].apply(lambda x: [i for i in word_tokenize(x) if i not in stopwords and i not in
                                       list(dictionary.token2id.keys())])
    return df


def chop(df2, n):
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i in text_feats:
        df[i] = df[i].apply(lambda x: [i[:n] for i in word_tokenize(x)])
    return df


def stem(df2):
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i in text_feats:
        df[i] = df[i].apply(lambda x: ' '.join(
            [stemmer.stem(i) for i in word_tokenize(x)]))
    return df


def lemmat(df2):
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i in text_feats:
        df[i] = df[i].apply(lambda x: ' '.join(
            [lemmatizer.lemmatize(i) for i in word_tokenize(x)]))
    return df


def extract_entity(df2):
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i in text_feats:
        df[i] = df[i].apply(lambda x: word_tokenize(x))
        df[i] = df[i].apply(lambda x: nltk.pos_tag(x))
        df[i] = df[i].apply(lambda x: [i[1:] for i in x])
    return df


def doc_features(df2):
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i, col in enumerate(text_feats):
        df['num_characters_{}'.format(i)] = df[col].map(
            lambda x: len(str(x)))  # length of sentence
        df['num_words_{}'.format(i)] = df[col].map(
            lambda x: len(str(x).split()))  # number of words
        df['num_spaces_{}'.format(i)] = df[col].map(lambda x: x.count(' '))
        df['num_alpha_{}'.format(i)] = df[col].apply(
            lambda x: sum(i.isalpha()for i in x))
        df['num_nonalpha_{}'.format(i)] = df[col].apply(
            lambda x: sum(1 - i.isalpha()for i in x))
    return df


def bag_of_words(df2, column, params=None):
    df = df2.copy()
    cv = CountVectorizer(params)
    bow = cv.fit_transform(df[column])
    return bow


def tf_idf(df2, column, params=None):
    df = df2.copy()
    tf = TfidfVectorizer(params)
    tfidf = tf.fit_transform(df[column])
    return tfidf


def PCA_text(df2, ndims, column, use_tfidf=True, params=None):
    df = df2.copy()
    if use_tfidf:
        bow = CountVectorizer(params).fit_transform(df[column])
    else:
        bow = CountVectorizer(params).fit_transform(df[column])
    pca_bow = PCA(ndims, random_state=seed).fit_transform(bow)
    pca_bow = pd.DataFrame(pca_bow)
    pca_bow.columns = ['PCA_dim{}_{}'.format(x, column) for x in range(pca_bow.shape[1])]
    return pca_bow


def SVD_text(df2, ndims, column, use_tfidf=True, params=None):
    df = df2.copy()
    if use_tfidf:
        bow = CountVectorizer(params).fit_transform(df[column])
    else:
        bow = CountVectorizer(params).fit_transform(df[column])
    svd_bow = TruncatedSVD(ndims, random_state=seed).fit_transform(bow)
    svd_bow = pd.DataFrame(svd_bow)
    svd_bow.columns = ['SVD_dim{}_{}'.format(x, column) for x in range(svd_bow.shape[1])]
    return svd_bow


def LDA_text(df2, ntopics, column, use_tfidf=True, params=None):
    df = df2.copy()
    if use_tfidf:
        bow = CountVectorizer(params).fit_transform(df[column])
    else:
        bow = CountVectorizer(params).fit_transform(df[column])
    lda_bow = LatentDirichletAllocation(
        ntopics, random_state=seed, n_jobs=4).fit_transform(bow)
    lda_bow = pd.DataFrame(lda_bow)
    lda_bow.columns = ['LDA_dim{}_{}'.format(x, column) for x in range(lda_bow.shape[1])]
    return lda_bow


def LSA_text(df2, ndims, column, use_tfidf=True, params=None):
    cv = CountVectorizer(params)
    svd = TruncatedSVD(ndims, random_state=1337)
    normalizer = Normalizer(copy=False)
    df = df2.copy()
    if use_tfidf:
        bow = CountVectorizer(params).fit_transform(df[column])
    else:
        bow = CountVectorizer(params).fit_transform(df[column])
    svd_bow = svd.fit_transform(bow)
    normed_bow = normalizer.fit_transform(svd_bow)
    lsa_bow = pd.DataFrame(normed_bow)
    lsa_bow.columns = ['LSA_dim{}_{}'.format(x, column) for x in range(lsa_bow.shape[1])]
    return lsa_bow
