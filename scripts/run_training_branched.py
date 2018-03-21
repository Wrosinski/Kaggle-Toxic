import gc
import os
from argparse import ArgumentParser

import keras
import keras_models
import numpy as np
import pandas as pd
import utils
from gensim.models import KeyedVectors
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD, Adadelta, Adam, Nadam, RMSprop
from keras.preprocessing import sequence, text
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--model_name",
                    help="Provide model name based on its definition.",
                    type=str, default='')
parser.add_argument("--kfold_run",
                    help="Whether to run KFold. Bagging is run if not.",
                    type=int, default=0)
parser.add_argument("--batch_size",
                    help="Model batch size.",
                    type=int, default=512)
parser.add_argument("--optimizer",
                    help="Optimizer to use during training.",
                    type=str, default='Nadam')
parser.add_argument("--importance",
                    help="Whether to train with Importance Sampling.",
                    type=int, default=0)
parser.add_argument("--stratify",
                    help="Whether to train StratifiedKFold.",
                    type=int, default=0)
parser.add_argument("--data_type",
                    help="Type of data to use during training.",
                    type=str, default='BasicClean')
parser.add_argument("--save_models",
                    help="Whether to save models during training.",
                    type=int, default=0)
parser.add_argument("--save_oof",
                    help="Whether to save OOF after training.",
                    type=int, default=0)
parser.add_argument("--load_models",
                    help="Whether load checkpoints from already trained models.",
                    type=int, default=0)
parser.add_argument("--prepare_submission",
                    help="Whether to prepare submission based on model output.",
                    type=int, default=0)
parser.add_argument("--gpu",
                    help="CUDA device ID, on which training will be performed.",
                    type=str, default='0')
args = parser.parse_args()
print('\nRunning with parameters: {}\n'.format(args))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)


n_folds = 5
n_bags = 1
split_size = 0.1
max_features = 300000
sequence_length = 196
embedding_dim = 300
bidirectional = True
run_prefix = '256_FastText300k_'
embedding_filename = 'FastText_300dim_embeddingBasic300k'

src = '/home/w/Projects/Toxic/data/'


if bidirectional and 'LSTM' in args.model_name or bidirectional and 'GRU' in args.model_name:
    run_prefix = 'Bidirectional{}'.format(run_prefix)
if args.kfold_run:
    general_run_name = '{}{}fold_BS{}_{}'.format(
        run_prefix, n_folds, args.batch_size, args.optimizer)
else:
    general_run_name = '{}{}bag_BS{}_{}'.format(
        run_prefix, n_bags, args.batch_size, args.optimizer)


if len(args.data_type) > 0:
    general_run_name += '_{}'.format(args.data_type)
if args.importance:
    general_run_name += '_ImportanceTrain'
if args.stratify and args.kfold_run:
    general_run_name += '_Stratified'

run_name = '{}{}'.format(args.model_name, general_run_name)
print('Run name: {}'.format(run_name))


model_callbacks = [EarlyStopping(monitor='val_loss', patience=9, verbose=1),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1,
                                     patience=4, min_lr=1e-5)]

if args.optimizer == 'Adam':
    optimizer = Adam(lr=1e-3, decay=1e-3)
    # optimizer = 'adam'
if args.optimizer == 'Nadam':
    optimizer = Nadam(lr=1e-3, schedule_decay=1e-3)
    # optimizer = 'nadam'
if args.optimizer == 'SGD':
    optimizer = SGD(lr=1e-2, momentum=0.9,
                    decay=1e-4, nesterov=True)


train, test = utils.load_data(src, mode=args.data_type)
target_columns = ['toxic', 'severe_toxic',
                  'obscene', 'threat', 'insult', 'identity_hate']

src_features = '/home/w/Projects/Toxic/data/features/'
data_badwords300 = pd.read_pickle(src_features + 'data_Binary300Badwords.pkl')
data_badwordsCount = pd.read_pickle(src_features + 'data_BadwordsCount.pkl')
data_textStatistics = pd.read_pickle(src_features + 'data_TextStatistics.pkl')
data_transformations = pd.read_pickle(
    src_features + 'data_TransformationsFeats20dim_SVDLSA.pkl')

X = pd.concat([data_badwords300,
               data_textStatistics, data_transformations], axis=1)
X['badwordsCount'] = data_badwordsCount

X_train_mlp = X.iloc[:train.shape[0], :]
X_test_mlp = X.iloc[train.shape[0]:, :]
features = np.setdiff1d(X_train_mlp.columns, target_columns)
X_train_mlp = X_train_mlp[features]
X_test_mlp = X_test_mlp[features]


del X, test
del data_badwords300, data_badwordsCount
del data_textStatistics, data_transformations
gc.collect()

train, test = utils.load_data(src, mode=args.data_type)
print(train.shape, test.shape)
list_classes = ["toxic", "severe_toxic",
                "obscene", "threat", "insult", "identity_hate"]
list_sentences_train = train["comment_text"].fillna("CVxTz").values
list_sentences_test = test["comment_text"].fillna("CVxTz").values

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train.comment_text.tolist() +
                       test.comment_text.tolist())
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index)) + 1

X_train = sequence.pad_sequences(
    list_tokenized_train, maxlen=sequence_length)  # [:1000]
y_train = train[list_classes].values  # [:1000]
X_test = sequence.pad_sequences(
    list_tokenized_test, maxlen=sequence_length)  # [:1000]
print(X_train.shape, y_train.shape, X_test.shape)

del train, test, list_tokenized_train, list_tokenized_test
gc.collect()


embedding_matrix = pd.read_pickle(
    '../data/embeddings/{}.pkl'.format(embedding_filename))


model_parameters = {
    'lstm_units': 256,
    'bidirectional': bidirectional,
    'nb_words': nb_words,
    'embedding_dim': embedding_dim,
    'embedding_matrix': embedding_matrix,
    'sequence_length': sequence_length,
    'optimizer': optimizer,
    'num_columns': X_train_mlp.shape[1],
}

pipeline_parameters = {
    'model_name': getattr(keras_models, args.model_name),
    'predict_test': True,
    'number_epochs': 1000,
    'batch_size': args.batch_size,
    'seed': 1337,
    'shuffle': True,
    'verbose': True,
    'run_save_name': run_name,
    'load_keras_model': args.load_models,
    'save_model': args.save_models,
    'save_history': True,
    'save_statistics': True,
    'output_statistics': True,
    'src_dir': os.getcwd(),
}

if args.kfold_run:
    oof_train, oof_test = utils.run_parametrized_kfold([X_train, X_train_mlp],
                                                       y_train,
                                                       [X_test, X_test_mlp],
                                                       pipeline_parameters=pipeline_parameters,
                                                       model_parameters=model_parameters,
                                                       model_callbacks=model_callbacks,
                                                       n_folds=n_folds,
                                                       importance_training=args.importance,
                                                       save_oof=args.save_oof)
    print(oof_train.shape, oof_test.shape)
else:

    X_tr, X_val, X_tr_mlp, X_val_mlp, y_tr, y_val = train_test_split(X_train,
                                                                     X_train_mlp,
                                                                     y_train,
                                                                     test_size=0.1,
                                                                     random_state=1337)

    oof_valid, oof_test = utils.run_parametrized_bagging([X_tr, X_tr_mlp],
                                                         y_tr,
                                                         [X_val, X_val_mlp],
                                                         y_val,
                                                         X_test=[
                                                             X_test, X_test_mlp],
                                                         pipeline_parameters=pipeline_parameters,
                                                         model_parameters=model_parameters,
                                                         model_callbacks=model_callbacks,
                                                         n_bags=n_bags,
                                                         split_size=split_size,
                                                         importance_training=args.importance)
    print(oof_valid.shape, oof_test.shape)


if args.prepare_submission:
    submission = utils.output_submission(
        oof_test.mean(axis=0), run_name, save=True)
