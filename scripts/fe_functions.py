import itertools
import random
import re

import category_encoders as ce
import nltk
import numpy as np
import pandas as pd
import sklearn
from fancyimpute import KNN
from gensim import corpora
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from nltk import ngrams, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import *
from nltk.tag import AffixTagger
from scipy.spatial import distance
from scipy.stats import boxcox
from sklearn.decomposition import PCA, LatentDirichletAllocation, TruncatedSVD
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.preprocessing import (MinMaxScaler, Normalizer,
                                   PolynomialFeatures, StandardScaler)
from textstat.textstat import textstat

seed = 1337


# Vol 1
def label_encode(df2):
    df = df2.copy()
    categorical_features = df.select_dtypes(
        include=['category']).columns.values
    df[categorical_features] = df[categorical_features].apply(
        lambda x: x.cat.codes)
    return df


def hash_encode1(df2):
    df = df2.copy()
    categorical_features = df.select_dtypes(
        include=['category']).columns.values
    hashing_encoder = ce.HashingEncoder(n_components=len(
        categorical_features), cols=categorical_features.tolist())
    df[categorical_features] = hashing_encoder.fit_transform(
        df[categorical_features])
    return df


def hash_encode2(df2):
    df = df2.copy()
    categorical_features = df.select_dtypes(
        include=['category']).columns.values
    hashing_encoder = FeatureHasher(n_features=len(
        categorical_features), input_type='string')
    df[categorical_features] = pd.DataFrame(hashing_encoder.fit_transform(
        df[categorical_features].as_matrix()).toarray())
    return df


def count_encode(df2):
    df = df2.copy()
    categorical_features = df.select_dtypes(
        include=['category']).columns.values
    for i in categorical_features:
        df[i] = df[i].astype('object').replace(df[i].value_counts())
    return df


def labelcount_encode(df2):
    df = df2.copy()
    categorical_features = df.select_dtypes(
        include=['category']).columns.values
    for cat_feature in categorical_features:
        cat_feature_value_counts = df[cat_feature].value_counts()
        value_counts_list = cat_feature_value_counts.index.tolist()
        value_counts_range_rev = list(
            reversed(range(len(cat_feature_value_counts))))  # for ascending ordering
        # for descending ordering
        value_counts_range = list(range(len(cat_feature_value_counts)))
        labelcount_dict = dict(zip(value_counts_list, value_counts_range))
        df[cat_feature] = df[cat_feature].map(labelcount_dict)
    return df


def target_encode(df2):
    df = df2.copy()
    categorical_features = df.select_dtypes(
        include=['category']).columns.values
    for cat_feature in categorical_features:
        group_target_mean = df.groupby([cat_feature])['target'].mean()
        df[cat_feature] = df[cat_feature].astype(
            'object').replace(group_target_mean)
    return df


# Vol 2
def polynomial_encode(df2):
    df = df2.copy()
    categorical_features = df.select_dtypes(
        include=['category']).columns.values
    df[categorical_features] = df[categorical_features].apply(
        lambda x: x.cat.codes)
    poly = PolynomialFeatures(degree=2, interaction_only=False)
    df = pd.DataFrame(poly.fit_transform(df))
    return df


def nan_encode(df2):
    df = df2.copy()
    missing_cols = np.sum(pd.isnull(df))[np.sum(
        pd.isnull(df)) >= 1].index.tolist()
    for i in missing_cols:
        df[i] = df[i].replace(df[i].cat.categories.tolist(), 0)
        df[i] = df[i].replace(np.nan, 1)
    return df


def group_featurebyfeature_encode(df2, newvar_name, var1, var2, transformation):
    df = df2.copy()
    categorical_features = df.select_dtypes(
        include=['category']).columns.values
    # label encode categorical features if to be used on categorical features too
    df[categorical_features] = df[categorical_features].apply(
        lambda x: x.cat.codes)
    # determine groups based on var1, then apply a chosen transformation to the groups based on values of var2
    df['{}'.format(newvar_name)] = (df.groupby(var1))[
        var2].transform('{}'.format(transformation))
    return df


def impute_explicit_numerical(df2):
    df = df2.copy()
    df.fillna(-999, inplace=True)  # impute with a specified value
    return df


def impute_mean_numerical(df2):
    df = df2.copy()
    numerical_features = df.select_dtypes(include=['number']).columns.values
    for i in numerical_features:
        # impute with mean of each column
        mean = df[i][~np.isnan(df[i])].mean()
        df[i] = df[i].replace(np.nan, mean)
    return df


def impute_median_numerical(df2):
    df = df2.copy()
    numerical_features = df.select_dtypes(include=['number']).columns.values
    for i in numerical_features:
        # impute with median of each column
        mean = df[i][~np.isnan(df[i])].median()
        df[i] = df[i].replace(np.nan, mean)
    return df


def impute_knn_numerical(df2):
    df = df2.copy()
    numerical_features = df.select_dtypes(include=['number']).columns.values
    # impute with mean using KNN algorithm for 5 closest rows
    dfknn = pd.DataFrame(KNN(k=5).complete(df), columns=df2.columns)
    return dfknn


def round_numerical(df2, precision):
    df = df2.copy()
    df = df.round(precision)
    return df


def bin_numerical(df2, step):
    df = df2.copy()
    numerical_features = df.select_dtypes(include=['number']).columns.values
    for i in numerical_features:
        feature_range = np.arange(0, np.max(df[i]), step)
        df[i] = pd.cut(df[i], feature_range, right=True)
        df[i] = pd.factorize(df[i], sort=True)[0]
    return df


def scale_standard_numerical(df2):
    df = df2.copy()
    df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df2.columns)
    return df


def scale_minmax_numerical(df2):
    df = df2.copy()
    df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df2.columns)
    return df


# Vol 3
def locally_linear_embedding_others(df2, n):
    df = df2.copy()
    # specifying the number of manifold dimensions, to which data is mapped
    lle = LocallyLinearEmbedding(n_components=n, random_state=seed)
    df = pd.DataFrame(lle.fit_transform(df))
    return df


def spectral_embedding_others(df2, n):
    df = df2.copy()
    # specifying the number of manifold dimensions, to which data is mapped
    se = SpectralEmbedding(n_components=n, random_state=seed)
    df = pd.DataFrame(se.fit_transform(df))
    return df


def tsne_embedding(df2, n):
    df = df2.copy()
    # specifying the number of manifold dimensions, to which data is mapped
    tsne = TSNE(n_components=n, random_state=seed)
    df = pd.DataFrame(tsne.fit_transform(df))
    return df


def randomtrees_embedding_others(df2):
    df = df2.copy()
    rte = RandomTreesEmbedding(random_state=seed)
    df = pd.DataFrame(rte.fit_transform(df).toarray())
    return df


def row_statistics_others(df2):
    df = df2.copy()
    df['zeros'] = np.sum(df == 0, axis=1)
    df['non-zeros'] = np.sum(df == 0, axis=1)
    df['NaNs'] = np.sum(np.isnan(df), axis=1)
    df['negatives'] = np.sum(df < 0, axis=1)
    df['sum_row'] = df.sum(axis=1)
    df['mean_row'] = df.mean(axis=1)
    df['std_row'] = df.std(axis=1)
    df['max_row'] = np.amax(df, axis=1)
    return df


def interactions_others(df2):
    df = df2.copy()
    cols = df2.columns
    for comb in itertools.combinations(cols, 2):
        feat = comb[0] + "_plus_" + comb[1]
        # addition can be changed to any other interaction like subtraction, multiplication, division
        df[feat] = df[comb[0]] + df[comb[1]]
    return df


def target_engineering_others(df2):
    df = df2.copy()
    df['target'] = np.log(df['target'])  # log-transform
    df['target'] = (df['target'] ** 0.25) + 1
    df['target'] = df['target'] ** 2  # square-transform
    df['target'], _ = boxcox(df['target'])  # Box-Cox transform

    # Bin target variable in case of regression
    target_range = np.arange(0, np.max(df['target']), 100)
    df['target'] = np.digitize(df.target.values, bins=target_range)
    return df


# Vol 4 - Text

stemmer = snowball.SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
stopwords_eng = stopwords.words('english')
words = re.compile(r"\w+", re.I)

model = KeyedVectors.load_word2vec_format(
    '/media/w/1c392724-ecf3-4615-8f3c-79368ec36380/DS Projects/Kaggle/Quora/data/embeddings/GoogleNews-vectors-negative300.bin', binary=True)

# Cleaning


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
    # https://www.analyticsvidhya.com/blog/2014/11/text-data-cleaning-steps-python/
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i in text_feats:
        df[i] = df[i].apply(lambda x: (''.join(''.join(s)[:2]
                                               for _, s in itertools.groupby(x))))
    return df

# Tokenizing


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
    # http://stackoverflow.com/questions/18658106/quick-implementation-of-character-n-grams-using-python
    def chargram_generate(string, n):
        return [string[i:i + n] for i in range(len(string) - n + 1)]
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i in text_feats:
        df[i] = df[i].apply(lambda x: [i for i in chargram_generate(x, 3)])
    return df

# Removing


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

# Roots


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
        df[i] = df[i].apply(lambda x: [stemmer.stem(i)
                                       for i in word_tokenize(x)])
    return df


def lemmat(df2):
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i in text_feats:
        df[i] = df[i].apply(lambda x: [lemmatizer.lemmatize(i)
                                       for i in word_tokenize(x)])
    return df

# Enriching


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


def get_readability(df2):
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i, col in enumerate(text_feats):
        df['flesch_reading_ease{}'.format(i)] = df[col].apply(
            lambda x: textstat.flesch_reading_ease(x))
        df['smog_index{}'.format(i)] = df[col].apply(
            lambda x: textstat.smog_index(x))
        df['flesch_kincaid_grade{}'.format(i)] = df[col].apply(
            lambda x: textstat.flesch_kincaid_grade(x))
        df['coleman_liau_index{}'.format(i)] = df[col].apply(
            lambda x: textstat.coleman_liau_index(x))
        df['automated_readability_index{}'.format(i)] = df[col].apply(
            lambda x: textstat.automated_readability_index(x))
        df['dale_chall_readability_score{}'.format(i)] = df[col].apply(
            lambda x: textstat.dale_chall_readability_score(x))
        df['difficult_words{}'.format(i)] = df[col].apply(
            lambda x: textstat.difficult_words(x))
        df['linsear_write_formula{}'.format(i)] = df[col].apply(
            lambda x: textstat.linsear_write_formula(x))
        df['gunning_fog{}'.format(i)] = df[col].apply(
            lambda x: textstat.gunning_fog(x))
        df['text_standard{}'.format(i)] = df[col].apply(
            lambda x: textstat.text_standard(x))
    return df

# Similarities & transformations


def token_similarity(df2):

    # https://www.kaggle.com/the1owl/quora-question-pairs/matching-que-for-quora-end-to-end-0-33719-pb
    def word_match_share(row, col1, col2, stopwords):
        q1words = {}
        q2words = {}
        for word in str(row[col1]).lower().split():
            if word not in stopwords:
                q1words[word] = 1
        for word in str(row[col2]).lower().split():
            if word not in stopwords:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            return 0
        shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
        shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
        R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / \
            (len(q1words) + len(q2words))
        return R

    df = df2.copy()
    df['word_match_share'] = df.apply(lambda x: word_match_share(x, 'question1', 'question2', stopwords_eng),
                                      axis=1, raw=True)
    return df


def word2vec_embedding(df2, model, num_words, num_dims):
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    for i in text_feats:
        df[i] = df[i].apply(lambda x: " ".join(
            [stemmer.stem(i) for i in word_tokenize(x)]))
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(df['question1'] + df['question2'])
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((num_words, num_dims))
    for word, i in word_index.items():
        if word in model.vocab:
            embedding_matrix[i] = model.word_vec(word)
    return pd.DataFrame(embedding_matrix)


def distances(df2, model):

    # https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question/blob/master/feature_engineering.py
    def sent2vec(s):
        words = str(s).lower().encode().decode('utf-8')
        words = word_tokenize(words)
        words = [w for w in words if w not in stopwords_eng]
        words = [w for w in words if w.isalpha()]
        M = []
        for w in words:
            try:
                M.append(model[w])
            except Exception as e:
                print(e)
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        return v / np.sqrt((v ** 2).sum())

    df = df2.copy()
    question1_vectors = np.zeros((df.shape[0], 300))
    for i, q in (enumerate(df.question1.values)):
        question1_vectors[i, :] = sent2vec(q)
    question2_vectors = np.zeros((df.shape[0], 300))
    for i, q in (enumerate(df.question2.values)):
        question2_vectors[i, :] = sent2vec(q)
    df['cosine_distance'] = [distance.cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                     np.nan_to_num(question2_vectors))]
    df['jaccard_distance'] = [distance.jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                       np.nan_to_num(question2_vectors))]
    df['hamming_distance'] = [distance.hamming(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                       np.nan_to_num(question2_vectors))]
    return df


def bag_of_words(df2):
    df = df2.copy()
    cv = CountVectorizer()
    bow = cv.fit_transform(df.question1 + df.question2).toarray()
    return pd.DataFrame(bow, columns=cv.get_feature_names())


def tf_idf(df2):
    df = df2.copy()
    tf = TfidfVectorizer()
    tfidf = tf.fit_transform(df.question1 + df.question2).toarray()
    return pd.DataFrame(tfidf, columns=tf.get_feature_names())


def PCA_text(df2, ndims):
    df = df2.copy()
    bow = CountVectorizer().fit_transform(df.question1 + df.question2).toarray()
    pca_bow = PCA(ndims, random_state=seed).fit_transform(bow)
    return pd.DataFrame(pca_bow)


def SVD_text(df2, ndims):
    df = df2.copy()
    bow = CountVectorizer().fit_transform(df.question1 + df.question2)
    svd_bow = TruncatedSVD(ndims, random_state=seed).fit_transform(bow)
    return pd.DataFrame(svd_bow)


def LDA_text(df2, ntopics):
    df = df2.copy()
    bow = CountVectorizer().fit_transform(df.question1 + df.question2)
    lda_bow = LatentDirichletAllocation(
        ntopics, random_state=seed).fit_transform(bow)
    return pd.DataFrame(lda_bow)


def LDA_text2(df2, ntopics):
    cv = CountVectorizer(stop_words='english', min_df=1, max_df=0.999)
    lda = LatentDirichletAllocation(ntopics, random_state=seed, n_jobs=1)
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    cv.fit(df.question1 + df.question2)
    bow = cv.transform(df.question1 + df.question2)
    lda.fit(bow)
    ldas = []
    for i in text_feats:
        bow_i = cv.transform(df[i])
        ldas.append(pd.DataFrame(lda.transform(bow_i), index=df[i]))
    return ldas


def LSA_text(df2, ndims):
    cv = CountVectorizer(stop_words='english', min_df=1, max_df=0.999)
    svd = TruncatedSVD(ndims, random_state=1337)
    normalizer = Normalizer(copy=False)
    df = df2.copy()
    text_feats = df.select_dtypes(include=['object']).columns.values
    cv.fit(df.question1 + df.question2)
    bow = cv.transform(df.question1 + df.question2)
    svd.fit(bow)
    transformed_bow = svd.transform(bow)
    normed_bow = normalizer.fit(transformed_bow)
    svds = []
    for i in text_feats:
        bow_i = cv.transform(df[i])
        svd_i = svd.transform(bow_i)
        normed_i = pd.DataFrame(normalizer.transform(svd_i), index=df[i])
        svds.append(normed_i)
    return svds


# Projection onto circle
def polar_coords_column(df2, colname, normalize=True):
    df = df2.copy()
    max_val = np.max(df['{}'.format(colname)])
    val_range = np.linspace(0, 360, max_val + 1)
    cat_feature_value_counts = df['{}'.format(colname)].value_counts()
    value_counts_list = cat_feature_value_counts.index.tolist()
    angle_dict = dict(zip(value_counts_list, val_range))

    df['{}_raw'.format(colname)] = df['{}'.format(colname)].map(angle_dict)
    df['{}_sin'.format(colname)] = np.sin(df['{}_raw'.format(colname)])
    df['{}_cos'.format(colname)] = np.cos(df['{}_raw'.format(colname)])
    df.drop(['{}_raw'.format(colname)], axis=1, inplace=True)
    if normalize:
        df['{}_sin'.format(colname)] = (df['{}_sin'.format(colname)] - np.min(df['{}_sin'.format(colname)])) / \
            ((np.max(df['{}_sin'.format(colname)])) -
             np.min(df['{}_sin'.format(colname)]))
        df['{}_cos'.format(colname)] = (df['{}_cos'.format(colname)] - np.min(df['{}_cos'.format(colname)])) / \
            ((np.max(df['{}_cos'.format(colname)])) -
             np.min(df['{}_cos'.format(colname)]))
    return df
