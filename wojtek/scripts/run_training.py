import gc
import os
from argparse import ArgumentParser

import keras
import keras_models_selected
import numpy as np
import pandas as pd
import utils
from gensim.models import KeyedVectors
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD, Adadelta, Adam, Nadam, RMSprop
from keras.preprocessing import sequence, text
from nltk.corpus import stopwords
from tqdm import tqdm
from utils import tf_roc_auc

parser = ArgumentParser()
parser.add_argument("--model_name",
                    help="Provide model name based on its definition.",
                    type=str, default='LSTMconcat2')
parser.add_argument("--run_name",
                    help="Provide current run name.",
                    type=str, default='R1')
parser.add_argument("--kfold_run",
                    help="Whether to run KFold. Bagging is run if not.",
                    type=int, default=0)
parser.add_argument("--batch_size",
                    help="Model batch size.",
                    type=int, default=256)
parser.add_argument("--optimizer",
                    help="Optimizer to use during training.",
                    type=str, default='Adam')
parser.add_argument("--importance",
                    help="Whether to train with Importance Sampling.",
                    type=int, default=0)
parser.add_argument("--stratify",
                    help="Whether to train StratifiedKFold.",
                    type=int, default=1)
parser.add_argument("--data_type",
                    help="Type of data to use during training.",
                    type=str, default='BasicClean2')
parser.add_argument("--save_models",
                    help="Whether to save models during training.",
                    type=int, default=0)
parser.add_argument("--save_oof",
                    help="Whether to save OOF after training.",
                    type=int, default=1)
parser.add_argument("--load_models",
                    help="Whether load checkpoints from already trained models.",
                    type=int, default=0)
parser.add_argument("--output_submission",
                    help="Whether to prepare submission based on model output.",
                    type=int, default=0)
parser.add_argument("--gpu",
                    help="CUDA device ID, on which training will be performed.",
                    type=str, default='0')
args = parser.parse_args()
print('\nRunning with parameters: {}\n'.format(args))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)


quick_run = 0
n_folds = 5
n_bags = 1
split_size = 0.1
run_seed = 1337

lstm_units = 128
dropout_rate = 0.20
learning_rate = 1e-3
character_level = False
bidirectional = True

max_features = 200000
max_features_k = int(max_features / 1e3)
sequence_length = 320
embedding_dim = 300
embedding_type = 'FastText2'
random_init = False


src = '../data/'
embedding_filename = '{}_{}_{}dim_{}k_{}len_random{}'.format(
    embedding_type, args.data_type, embedding_dim, max_features_k, sequence_length, int(random_init))
run_prefix = '{}_{}_KFold{}_{}_'.format(
    lstm_units, args.run_name, int(args.kfold_run), embedding_filename)


if bidirectional and 'LSTM' in args.model_name or bidirectional and 'GRU' in args.model_name:
    run_prefix = 'Bidirectional{}'.format(run_prefix)
if args.kfold_run:
    general_run_name = '{}{}fold_BS{}_{}{}'.format(
        run_prefix, n_folds, args.batch_size, args.optimizer, learning_rate)
else:
    general_run_name = '{}{}bag_BS{}_{}{}'.format(
        run_prefix, n_bags, args.batch_size, args.optimizer, learning_rate)


if args.importance:
    general_run_name += '_ImportanceTrain'
if args.stratify and args.kfold_run:
    general_run_name += '_Stratified'

run_name = '{}{}'.format(args.model_name, general_run_name)
print('Run name: {}'.format(run_name))


model_callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.3, verbose=1,
                                     patience=3, min_lr=1e-5)]

if args.optimizer == 'Adam':
    optimizer = Adam(lr=learning_rate)  # 5e-4
if args.optimizer == 'Nadam':
    optimizer = Nadam(lr=learning_rate)  # 5e-4
if args.optimizer == 'SGD':
    optimizer = SGD(lr=1e-3, momentum=0.9,
                    decay=1e-4, nesterov=True)


train, test = utils.load_data(src, mode=args.data_type)
print(train.shape, test.shape)
list_classes = ["toxic", "severe_toxic",
                "obscene", "threat", "insult", "identity_hate"]
list_sentences_train = train["comment_text"].fillna("CVxTz").values
list_sentences_test = test["comment_text"].fillna("CVxTz").values

tokenizer = text.Tokenizer(num_words=max_features, char_level=character_level)
tokenizer.fit_on_texts(train.comment_text.tolist() +
                       test.comment_text.tolist())
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index)) + 1

X_train = sequence.pad_sequences(
    list_tokenized_train, maxlen=sequence_length)
y_train = train[list_classes].values
X_test = sequence.pad_sequences(
    list_tokenized_test, maxlen=sequence_length)
if quick_run:
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    X_test = X_test[:5000]

print(X_train.shape, y_train.shape, X_test.shape)

if 'Hierarchical' in run_name:
    X_train = np.expand_dims(X_train, 1)
    X_test = np.expand_dims(X_test, 1)
    print(X_train.shape, y_train.shape, X_test.shape)

del train, test, list_tokenized_train, list_tokenized_test
gc.collect()


embedding_matrix = pd.read_pickle(
    '../data/embeddings/{}.pkl'.format(embedding_filename))


model_parameters = {
    'lstm_units': lstm_units,
    'dropout_rate': dropout_rate,
    'bidirectional': bidirectional,
    'nb_words': nb_words,
    'embedding_dim': embedding_dim,
    'embedding_matrix': embedding_matrix,
    'sequence_length': sequence_length,
    'optimizer': optimizer,
    'loss': tf_roc_auc,
}

pipeline_parameters = {
    'model_name': getattr(keras_models_selected, args.model_name),
    'predict_test': True,
    'number_epochs': 15,
    'batch_size': args.batch_size,
    'seed': run_seed,
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
    oof_train, oof_test = utils.run_parametrized_kfold(X_train, y_train, X_test,
                                                       pipeline_parameters,
                                                       model_parameters,
                                                       model_callbacks=model_callbacks,
                                                       n_folds=n_folds,
                                                       stratify=args.stratify,
                                                       importance_training=args.importance,
                                                       save_oof=args.save_oof)

    oof_test_mean = oof_test.mean(axis=-1)
    print(oof_train.shape, oof_test.shape, oof_test_mean.shape)

    if args.output_submission:
        submission = utils.output_submission(
            oof_test_mean, run_name, save=True)
else:
    oof_valid, oof_test = utils.run_parametrized_bagging(X_train, y_train,
                                                         X_test=X_test,
                                                         pipeline_parameters=pipeline_parameters,
                                                         model_parameters=model_parameters,
                                                         model_callbacks=model_callbacks,
                                                         n_bags=n_bags,
                                                         split_size=split_size,
                                                         importance_training=args.importance,
                                                         save_oof=args.save_oof)

    oof_test_mean = oof_test.mean(axis=0)
    print(oof_valid.shape, oof_test.shape, oof_test_mean.shape)

    if args.output_submission:
        submission = utils.output_submission(
            oof_test_mean, run_name, save=True)
