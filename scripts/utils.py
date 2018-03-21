import glob
import pprint

import numpy as np
import pandas as pd
import tflearn
from keras_pipeline import KerasPipeline

# Loss-related


def tf_roc_auc(y_true, y_pred):
    return tflearn.objectives.roc_auc_score(y_pred, y_true)

# Training-related


def pick_target_columns(df, target_columns, col='toxic'):
    if col == 'toxic':
        non_target_cols = [x for x in df.columns if col not in x]
        non_target_cols2 = [x for x in df.columns if 'severe' in x]
        non_target_cols.extend(non_target_cols2)
    else:
        non_target_cols = [x for x in df.columns if col not in x]
    target_cols = np.setdiff1d(df.columns, non_target_cols)
    return target_cols


def run_parametrized_kfold(X_train, y_train, X_test,
                           pipeline_parameters, model_parameters,
                           model_callbacks,
                           n_folds, n_bags=None, save_oof=True,
                           importance_training=False,
                           stratify=False,
                           MLP_array=None):

    pipeline = KerasPipeline(model_name=pipeline_parameters['model_name'],
                             predict_test=pipeline_parameters['predict_test'],
                             number_epochs=pipeline_parameters['number_epochs'],
                             batch_size=pipeline_parameters['batch_size'],
                             seed=pipeline_parameters['seed'],
                             shuffle=pipeline_parameters['shuffle'],
                             verbose=pipeline_parameters['verbose'],
                             run_save_name=pipeline_parameters['run_save_name'],
                             load_keras_model=pipeline_parameters['load_keras_model'],
                             save_model=pipeline_parameters['save_model'],
                             save_history=pipeline_parameters['save_history'],
                             save_statistics=pipeline_parameters['save_statistics'],
                             output_statistics=pipeline_parameters['output_statistics'],
                             src_dir=pipeline_parameters['src_dir'])

    if n_bags is not None:
        print('Running parametrized bagged KFold run.')
        model, oof_train, oof_test = pipeline.bagged_kfold_run(X_train, y_train,
                                                               X_test=X_test,
                                                               model_params=model_parameters,
                                                               model_callbacks=model_callbacks,
                                                               n_bags=n_bags,
                                                               n_folds=n_folds,
                                                               stratify=stratify,
                                                               index_number=None,
                                                               save_oof=save_oof,
                                                               importance_training=importance_training,
                                                               MLP_array=MLP_array)
    else:
        print('Running parametrized KFold run.')
        model, oof_train, oof_test = pipeline.kfold_run(X_train, y_train,
                                                        X_test=X_test,
                                                        model_params=model_parameters,
                                                        model_callbacks=model_callbacks,
                                                        n_folds=n_folds,
                                                        stratify=stratify,
                                                        index_number=None,
                                                        save_oof=save_oof,
                                                        importance_training=importance_training,
                                                        MLP_array=MLP_array)
    return oof_train, oof_test


def run_parametrized_bagging(X_train, y_train,
                             X_valid=None, y_valid=None,
                             X_test=None,
                             pipeline_parameters=None, model_parameters=None,
                             model_callbacks=None, n_bags=None,
                             user_split=True,
                             split_size=0.1,
                             importance_training=False,
                             save_oof=False):

    pipeline = KerasPipeline(model_name=pipeline_parameters['model_name'],
                             predict_test=pipeline_parameters['predict_test'],
                             number_epochs=pipeline_parameters['number_epochs'],
                             batch_size=pipeline_parameters['batch_size'],
                             seed=pipeline_parameters['seed'],
                             shuffle=pipeline_parameters['shuffle'],
                             verbose=pipeline_parameters['verbose'],
                             run_save_name=pipeline_parameters['run_save_name'],
                             load_keras_model=pipeline_parameters['load_keras_model'],
                             save_model=pipeline_parameters['save_model'],
                             save_history=pipeline_parameters['save_history'],
                             save_statistics=pipeline_parameters['save_statistics'],
                             output_statistics=pipeline_parameters['output_statistics'],
                             src_dir=pipeline_parameters['src_dir'])

    print('Running parametrized bagging')
    model, valid_preds, test_preds = pipeline.bag_run(X_train, y_train,
                                                      X_valid=X_valid, y_valid=y_valid,
                                                      X_test=X_test,
                                                      model_params=model_parameters,
                                                      model_callbacks=model_callbacks,
                                                      n_bags=n_bags,
                                                      user_split=user_split,
                                                      split_size=split_size,
                                                      index_number=None,
                                                      importance_training=importance_training,
                                                      save_oof=save_oof)
    return valid_preds, test_preds


def load_data(src, mode='SpacyClean'):
    if mode == 'BasicClean':
        print('Load data with basic cleaning.')
        X_train = pd.read_pickle(src + 'train_basic_clean.pkl')
        X_test = pd.read_pickle(src + 'test_basic_clean.pkl')
    if mode == 'BasicClean2':
        print('Load data with basic cleaning with non-alphanumeric contained.')
        X_train = pd.read_pickle(src + 'train_basic_clean2.pkl')
        X_test = pd.read_pickle(src + 'test_basic_clean2.pkl')
    if mode == 'SpacyClean':
        print('Load data cleaned with Spacy.')
        X_train = pd.read_pickle(src + 'train_spacy_clean.pkl')
        X_test = pd.read_pickle(src + 'test_spacy_clean.pkl')
    if mode == 'TextacyClean':
        print('Load data cleaned with Spacy + Textacy.')
        X_train = pd.read_pickle(src + 'train_textacy_clean.pkl')
        X_test = pd.read_pickle(src + 'test_textacy_clean.pkl')
    if mode == 'TextacyFullclean':
        print('Load data cleaned with Spacy + Textacy and most probable words kept.')
        X_train = pd.read_pickle(src + 'train_textacy_fullclean.pkl')
        X_test = pd.read_pickle(src + 'test_textacy_fullclean.pkl')
    return X_train, X_test


def output_submission(test_preds, name, save):
    submission = pd.read_csv('../input/sample_submission.csv')
    submission.iloc[:, 1:] = test_preds
    if save:
        submission.to_csv(
            '../submissions/{}.csv'.format(name), index=False)
    return submission


def load_predictions(src, load_oof=True, contains=None, contains2=None, not_contains=None):
    if load_oof:
        train_files = sorted(glob.glob('{}train/*.pkl'.format(src)))
    else:
        train_files = sorted(glob.glob('{}valid/*.pkl'.format(src)))
    test_files = sorted(glob.glob('{}test/*.pkl'.format(src)))
    if contains is not None:
        train_files = [x for x in train_files if contains in x]
        test_files = [x for x in test_files if contains in x]
        if contains2 is not None:
            train_files = [x for x in train_files if contains2 in x]
            test_files = [x for x in test_files if contains2 in x]
    if not_contains is not None:
        train_files = [x for x in train_files if not_contains not in x]
        test_files = [x for x in test_files if not_contains not in x]
    print('\nTrain files:\n')
    pprint.pprint(train_files)
    print('\nTest files:\n')
    pprint.pprint(test_files)
    valid_preds = []
    test_preds = []
    for i in train_files:
        valid_preds.append(pd.read_pickle(i))
    for i in test_files:
        test_preds.append(pd.read_pickle(i))
    return valid_preds, test_preds


def save_parameter_dict(filename, dictionary):
    with open(filename, 'w') as f:
        for key, value in dictionary.items():
            f.write('%s : %s\n' % (key, value))
    return
