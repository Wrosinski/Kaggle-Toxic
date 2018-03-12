import os
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.utils import shuffle
from utils import pick_target_columns

plt.style.use('ggplot')


class GBMPipeline(object):

    def __init__(self,
                 use_lgb=True,
                 predict_test=False,
                 objective=None,
                 eval_function=None,
                 seed=None,
                 shuffle=True,
                 verbose=True,
                 run_save_name=None,
                 save_model=True,
                 save_history=True,
                 src_dir=None,
                 save_statistics=False,
                 output_statistics=False,
                 output_importance=False
                 ):

        self.use_lgb = use_lgb
        self.predict_test = predict_test
        self.objective = objective
        self.eval_function = eval_function
        self.seed = seed
        self.shuffle = shuffle
        self.verbose = verbose
        self.run_save_name = run_save_name

        self.src_dir = src_dir if src_dir is not None else os.getcwd()
        self.save_model = save_model
        self.save_history = save_history
        self.save_statistics = save_statistics if run_save_name is not None else False
        self.output_statistics = output_statistics
        self.output_importance = output_importance

        self.oof_train = None
        self.oof_test = None

        self.i = 1
        self.prefix = None
        self.start_time = time.time()
        self.checkpoints_dst = self.src_dir + '/checkpoints/'

        self.history = {}
        self.losses = []
        self.best_iter = []
        self.predictions_valid = []
        self.predictions_test = []

    def bag_run(self,
                X_train, y_train=None,
                X_valid=None, y_valid=None,
                X_test=None, y_test=None,
                n_bags=2,
                model_params=None,
                train_params=None,
                split_size=0.2,
                output_submission=False,
                save_preds=True):

        if isinstance(y_train, pd.core.frame.Series):
            y_train = y_train.values
        if isinstance(y_train, pd.core.frame.Series):
            y_valid = y_valid.values
        if isinstance(y_train, pd.core.frame.Series):
            y_test = y_test.values

        index = 0

        if self.use_lgb:
            print('Using LightGBM')
            prefix = 'LGB'
        else:
            print('Using XGBoost')
            prefix = 'XGB'

        if self.save_model or self.output_statistics:
            os.makedirs('{0}{1}'.format(
                self.checkpoints_dst, self.run_save_name), exist_ok=True)

        print('Running bagging (currently just one bag).')
        print('Start training with parameters: {} \n \n'.format(model_params))
        print('X_train shape: {}'.format(X_train.shape))
        if X_test is not None:
            print('X_test shape: {}'.format(X_test.shape))

        if X_valid is not None and y_valid is not None:
            print('Validating on subset of data specified by user.')
            print('X_valid shape: {}'.format(X_valid.shape))
        else:
            if self.seed is not None:
                print('Splitting data - validation split size: {}, split seed: {}'.format(
                    split_size, self.seed))
            else:
                print(
                    'Splitting data - validation split size: {}, seed not set.'.format(split_size))

            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_size=split_size, random_state=self.seed)

        col = ['toxic', 'severe_toxic', 'obscene',
               'threat', 'insult', 'identity_hate']
        valid_predictions = np.zeros((X_valid.shape[0], len(col)))
        test_predictions = np.zeros((X_test.shape[0], len(col)))

        for i, j in enumerate(col):
            print('Training model for column:', j)
            self.prefix = '{}_{}'.format(prefix, j)

            if self.use_lgb:
                lgb_train = lgb.Dataset(X_train, y_train[j])
                lgb_val = lgb.Dataset(X_valid, y_valid[j], reference=lgb_train)

                if self.objective is not None or self.eval_function is not None:
                    print('Using custom evaluation function.')
                    gbm = lgb.train(model_params, lgb_train, valid_sets=lgb_val,
                                    evals_result=self.history,
                                    num_boost_round=train_params['boost_round'],
                                    early_stopping_rounds=train_params['stopping_rounds'],
                                    verbose_eval=train_params['verbose_eval'],
                                    fobj=self.objective,
                                    feval=self.eval_function)
                else:
                    gbm = lgb.train(model_params, lgb_train, valid_sets=lgb_val,
                                    evals_result=self.history,
                                    num_boost_round=train_params['boost_round'],
                                    early_stopping_rounds=train_params['stopping_rounds'],
                                    verbose_eval=train_params['verbose_eval'],)

                valid_preds = gbm.predict(
                    X_valid, num_iteration=gbm.best_iteration)
                self.best_iter.append(gbm.best_iteration)

            else:
                dtrain = xgb.DMatrix(X_train, label=y_train[j])
                dval = xgb.DMatrix(X_valid, label=y_valid[j])
                watchlist = [(dtrain, 'train'), (dval, 'valid')]
                if self.objective is not None or self.eval_function is not None:
                    print('Using custom evaluation function.')
                    gbm = xgb.train(model_params, dtrain, evals=watchlist,
                                    evals_result=self.history,
                                    num_boost_round=train_params['boost_round'],
                                    early_stopping_rounds=train_params['stopping_rounds'],
                                    verbose_eval=train_params['verbose_eval'],
                                    obj=self.objective,
                                    feval=self.eval_function,
                                    maximize=True)
                else:
                    gbm = xgb.train(model_params, dtrain, evals=watchlist,
                                    evals_result=self.history,
                                    num_boost_round=train_params['boost_round'],
                                    early_stopping_rounds=train_params['stopping_rounds'],
                                    verbose_eval=train_params['verbose_eval'],)

                valid_preds = gbm.predict(
                    dval, ntree_limit=gbm.best_ntree_limit)
                self.best_iter.append(gbm.best_ntree_limit)

            if self.save_model:
                print('Saving model into file.')
                gbm.save_model(
                    '{0}{1}/{2}_{1}_{3}.txt'.format(self.checkpoints_dst,
                                                    self.run_save_name, self.prefix, self.i))
            if self.save_history:
                print('Saving model history into file.')
                pd.to_pickle(self.history, '{0}{1}/{2}_{1}_{3}_eval_history.pkl'.format(self.checkpoints_dst,
                                                                                        self.run_save_name, self.prefix,
                                                                                        self.i))
            if self.output_statistics:
                self.output_run_statistics(index)
            if self.output_importance:
                self.visualize_importance(gbm)

            if self.predict_test and X_test is not None:
                test_preds = self.predict_on_test(
                    X_test, gbm)
                test_predictions[:, i] = test_preds

            valid_predictions[:, i] = valid_preds

            index += 1
            self.i += 1

        print('Mean loss for current bagging run:',
              np.array(self.losses).mean(axis=0))

        if save_preds:
            pd.to_pickle(valid_predictions, 'predictions/valid/{}_{:.5f}.pkl'.format(
                self.run_save_name, np.array(self.losses).mean(axis=0)))
            if self.predict_test:
                pd.to_pickle(test_predictions, 'predictions/test/{}_{:.5f}.pkl'.format(
                    self.run_save_name, np.array(self.losses).mean(axis=0)))
        if output_submission:
            self.prepare_submission(test_predictions, save=True)

        if self.predict_test and X_test is not None:
            return valid_predictions, test_predictions, gbm
        else:
            return valid_predictions, gbm

    def full_train_run(self,
                       X_train, y_train=None,
                       X_test=None, y_test=None,
                       model_params=None,
                       train_params=None,
                       output_submission=False,
                       save_preds=True):

        if isinstance(y_train, pd.core.frame.Series):
            y_train = y_train.values
        if isinstance(y_train, pd.core.frame.Series):
            y_valid = y_valid.values
        if isinstance(y_train, pd.core.frame.Series):
            y_test = y_test.values

        if self.use_lgb:
            print('Using LightGBM')
            self.prefix = 'LGB'
        else:
            print('Using XGBoost')
            self.prefix = 'XGB'

        if self.save_model or self.output_statistics:
            os.makedirs('{0}{1}'.format(
                self.checkpoints_dst, self.run_save_name), exist_ok=True)

        print('Running bagging (currently just one bag).')
        print('X_train shape: {}'.format(X_train.shape))
        if X_test is not None:
            print('X_test shape: {}'.format(X_test.shape))

        print('Start training with parameters: {} \n \n'.format(model_params))

        if self.use_lgb:
            lgb_train = lgb.Dataset(X_train, y_train)

            if self.objective is not None or self.eval_function is not None:
                print('Using custom evaluation function.')
                gbm = lgb.train(model_params, lgb_train,
                                evals_result=self.history,
                                num_boost_round=train_params['boost_round'],
                                verbose_eval=train_params['verbose_eval'],
                                fobj=self.objective,
                                feval=self.eval_function)
            else:
                gbm = lgb.train(model_params, lgb_train,
                                evals_result=self.history,
                                num_boost_round=train_params['boost_round'],
                                verbose_eval=train_params['verbose_eval'],)

        else:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            watchlist = [(dtrain, 'train')]
            if self.objective is not None or self.eval_function is not None:
                print('Using custom evaluation function.')
                gbm = xgb.train(model_params, dtrain, evals=watchlist,
                                evals_result=self.history,
                                num_boost_round=train_params['boost_round'],
                                verbose_eval=train_params['verbose_eval'],
                                obj=self.objective,
                                feval=self.eval_function,
                                maximize=True)
            else:
                gbm = xgb.train(model_params, dtrain, evals=watchlist,
                                evals_result=self.history,
                                num_boost_round=train_params['boost_round'],
                                verbose_eval=train_params['verbose_eval'],)

        if self.save_model:
            print('Saving model into file.')
            gbm.save_model(
                '{0}{1}/{2}_{1}_{3}.txt'.format(self.checkpoints_dst,
                                                self.run_save_name, self.prefix, self.i))
        if self.save_history:
            print('Saving model history into file.')
            pd.to_pickle(self.history, '{0}{1}/{2}_{1}_{3}_eval_history.pkl'.format(self.checkpoints_dst,
                                                                                    self.run_save_name, self.prefix,
                                                                                    self.i))
        if self.output_statistics:
            self.output_run_statistics()
        if self.output_importance:
            self.visualize_importance(gbm)

        if self.predict_test and X_test is not None:
            test_preds = self.predict_on_test(
                X_test, gbm, output_submission)

        if self.predict_test and save_preds:
            pd.to_pickle(test_preds, 'preds/test/{}_{:.5f}.pkl'.format(
                self.run_save_name, np.array(self.losses).mean(axis=0)))

        if self.predict_test and X_test is not None:
            return test_preds, gbm
        else:
            return gbm

    def fold_run(self,
                 X_train, y_train=None,
                 X_test=None, y_test=None,
                 n_folds=5,
                 stratify=False,
                 model_params=None,
                 train_params=None,
                 output_submission=False,
                 save_oof=True,
                 additional_features=None):

        if isinstance(y_train, pd.core.frame.Series):
            y_train = y_train.values
        if isinstance(y_train, pd.core.frame.Series):
            y_valid = y_valid.values
        if isinstance(y_train, pd.core.frame.Series):
            y_test = y_test.values

        index = 0

        if self.use_lgb:
            print('Using LightGBM')
            prefix = 'LGB'
        else:
            print('Using XGBoost')
            prefix = 'XGB'

        if self.save_model or self.output_statistics:
            os.makedirs('{0}{1}'.format(
                self.checkpoints_dst, self.run_save_name), exist_ok=True)

        col = ['toxic', 'severe_toxic', 'obscene',
               'threat', 'insult', 'identity_hate']

        self.oof_train = np.zeros((X_train.shape[0], len(col)))
        print('OOF train predictions shape: {}'.format(self.oof_train.shape))
        print('X_train shape: {}'.format(X_train.shape))
        if X_test is not None:
            self.oof_test = np.zeros((X_test.shape[0], len(col), n_folds))
            print('OOF test predictions shape: {}'.format(self.oof_test.shape))
            print('X_test shape: {}'.format(X_test.shape))

        print('Running KFold run with {} folds'.format(n_folds))
        print('Start training with parameters: {} \n \n'.format(model_params))

        if stratify:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=self.shuffle,
                                 random_state=self.seed)
        else:
            kf = KFold(n_splits=n_folds, shuffle=self.shuffle,
                       random_state=self.seed)

        for train_index, val_index in kf.split(X_train, y_train):

            X_tr_, X_val_ = X_train.iloc[train_index,
                                         :], X_train.iloc[val_index, :]
            y_tr, y_val = y_train.iloc[train_index,
                                       :], y_train.iloc[val_index, :]

            for i, j in enumerate(col):
                print('Training model for column:', j)
                self.prefix = '{}_{}'.format(prefix, j)

                target_train_columns = pick_target_columns(X_train, col, j)
                if additional_features is not None:
                    target_train_columns.tolist().extend(additional_features.tolist())
                X_tr = X_tr_[target_train_columns]
                X_val = X_val_[target_train_columns]

                if self.use_lgb:
                    lgb_train = lgb.Dataset(X_tr, y_tr[j])
                    lgb_val = lgb.Dataset(X_val, y_val[j], reference=lgb_train)

                    if self.objective is not None or self.eval_function is not None:
                        print('Using custom evaluation function.')
                        gbm = lgb.train(model_params, lgb_train, valid_sets=lgb_val,
                                        evals_result=self.history,
                                        num_boost_round=train_params['boost_round'],
                                        early_stopping_rounds=train_params['stopping_rounds'],
                                        verbose_eval=train_params['verbose_eval'],
                                        fobj=self.objective,
                                        feval=self.eval_function)
                    else:
                        gbm = lgb.train(model_params, lgb_train, valid_sets=lgb_val,
                                        evals_result=self.history,
                                        num_boost_round=train_params['boost_round'],
                                        early_stopping_rounds=train_params['stopping_rounds'],
                                        verbose_eval=train_params['verbose_eval'],)

                    self.oof_train[val_index, i] = gbm.predict(
                        X_val, num_iteration=gbm.best_iteration)
                    self.best_iter.append(gbm.best_iteration)

                else:
                    dtrain = xgb.DMatrix(X_tr, label=y_tr[j])
                    dval = xgb.DMatrix(X_val, label=y_val[j])
                    watchlist = [(dtrain, 'train'), (dval, 'valid')]
                    if self.objective is not None or self.eval_function is not None:
                        print('Using custom evaluation function.')
                        gbm = xgb.train(model_params, dtrain, evals=watchlist,
                                        evals_result=self.history,
                                        num_boost_round=train_params['boost_round'],
                                        early_stopping_rounds=train_params['stopping_rounds'],
                                        verbose_eval=train_params['verbose_eval'],
                                        obj=self.objective,
                                        feval=self.eval_function,
                                        maximize=True)
                    else:
                        gbm = xgb.train(model_params, dtrain, evals=watchlist,
                                        evals_result=self.history,
                                        num_boost_round=train_params['boost_round'],
                                        early_stopping_rounds=train_params['stopping_rounds'],
                                        verbose_eval=train_params['verbose_eval'],)

                    self.oof_train[val_index, i] = gbm.predict(
                        dval, ntree_limit=gbm.best_ntree_limit)
                    self.best_iter.append(gbm.best_ntree_limit)

                if self.save_model:
                    print('Saving model into file.')
                    gbm.save_model(
                        '{0}{1}/{2}_{1}_{3}.txt'.format(self.checkpoints_dst,
                                                        self.run_save_name, self.prefix, self.i))
                if self.save_history:
                    print('Saving model history into file.')
                    pd.to_pickle(self.history, '{0}{1}/{2}_{1}_fold{3}_eval_history.pkl'.format(self.checkpoints_dst,
                                                                                                self.run_save_name, self.prefix,
                                                                                                self.i))
                if self.output_statistics:
                    self.output_run_statistics(index)
                if self.output_importance:
                    self.visualize_importance(gbm)

                if self.predict_test and X_test is not None:
                    self.oof_test[:, i, self.i -
                                  1] = self.predict_on_test(X_test, gbm)

                index += 1
            self.i += 1

        print('Mean loss for current KFold run:',
              np.array(self.losses).mean(axis=0))

        if output_submission:
            self.prepare_submission(self.oof_test.mean(axis=-1),
                                    save=True)

        if save_oof:
            pd.to_pickle(self.oof_train, 'oof/train/{}_{:.5f}.pkl'.format(
                self.run_save_name, np.array(self.losses).mean(axis=0)))
            if self.predict_test:
                pd.to_pickle(self.oof_test, 'oof/test/{}_{:.5f}.pkl'.format(
                    self.run_save_name, np.array(self.losses).mean(axis=0)))

        if self.predict_test and X_test is not None:
            return self.oof_train, self.oof_test, gbm
        else:
            return self.oof_train, gbm

    def predict_on_test(self, X_test, gbm=None):

        print('Predicting on test data.')

        if self.use_lgb:
            if gbm is None:
                gbm = lgb.Booster(
                    model_file='{0}{1}/{2}_{1}_{3}.txt'.format(self.checkpoints_dst,
                                                               self.run_save_name, self.prefix, self.i))
            test_preds = gbm.predict(
                X_test, num_iteration=gbm.best_iteration)

        else:
            if gbm is None:
                gbm = xgb.Booster(
                    model_file='{0}{1}/{2}_{1}_{3}.txt'.format(self.checkpoints_dst,
                                                               self.run_save_name, self.prefix, self.i))
            test_preds = gbm.predict(xgb.DMatrix(
                X_test), ntree_limit=gbm.best_ntree_limit)

        return test_preds

    def prepare_submission(self, test_predictions, save=True):
        print('Preparing submission.')
        col = ['toxic', 'severe_toxic', 'obscene',
               'threat', 'insult', 'identity_hate']
        subm = pd.read_csv('../input/sample_submission.csv')
        submid = pd.DataFrame({'id': subm["id"]})
        submission = pd.concat(
            [submid, pd.DataFrame(test_predictions, columns=col)], axis=1)
        if save:
            submission.to_csv('../submissions/{}_{}_loss_{:.5f}.csv'.format(self.run_save_name, self.i,
                                                                            np.array(self.losses).mean(axis=0)),
                              index=False)
        return submission

    def output_run_statistics(self, index):
        if self.use_lgb:
            lgb_min_loss = np.max(self.history['valid_0']['auc'])
            min_loss = lgb_min_loss
            self.losses.append(lgb_min_loss)
            print('Minimum validation split loss for current fold/bag: {} \n'.format(
                lgb_min_loss))
        else:
            xgb_min_loss = np.max(self.history['valid']['auc'])
            min_loss = xgb_min_loss

            self.losses.append(xgb_min_loss)
            print('Minimum validation split loss for current fold/bag: {} \n'.format(
                xgb_min_loss))
        print('Seconds it took to train the model: {} \n'.format(
            time.time() - self.start_time))
        print('Best iterations: {} \n'.format(self.best_iter))

        if self.save_statistics:
            with open('{0}{1}/{2}_{1}_{3}_run_stats_loss_{4:.5f}_round_{5}.txt'.format(
                    self.checkpoints_dst,
                    self.run_save_name, self.prefix, self.i,
                    min_loss, self.best_iter[index]), 'w') as text_file:
                if self.use_lgb:
                    text_file.write('Minimum validation split loss for current fold/bag: {} \n'.format(
                                    lgb_min_loss))
                else:
                    text_file.write('Minimum validation split loss for current fold/bag: {} \n'.format(
                                    xgb_min_loss))
                text_file.write('Seconds it took to train the model: {} \n'.format(
                    time.time() - self.start_time))
                text_file.write('Mean loss for current run: {}\n'.format(
                                np.array(self.losses).mean(axis=0)))
                text_file.write(
                    'Best iterations: {} \n'.format(self.best_iter))
        return

    def visualize_importance(self, gbm=None, features_number=25):

        print('Visualize model feature importance.')

        if self.use_lgb:
            if gbm is None:
                gbm = lgb.Booster(
                    model_file='{0}{1}/{2}_{1}_{3}.txt'.format(self.checkpoints_dst,
                                                               self.run_save_name, self.prefix, self.i))
            importance = gbm.feature_importance()
            names = gbm.feature_name()
        else:
            if gbm is None:
                gbm = xgb.Booster(
                    model_file='{0}{1}/{2}_{1}_{3}.txt'.format(self.checkpoints_dst,
                                                               self.run_save_name, self.prefix, self.i))
            importance = list(gbm.get_fscore().values())
            names = list(gbm.get_fscore().keys())

        df_importance = pd.DataFrame()
        df_importance['fscore'] = importance
        df_importance['feature'] = names
        df_importance.sort_values('fscore', ascending=False, inplace=True)
        df_importance = df_importance.iloc[:features_number, :]

        # plt.figure()
        fscore_plot = df_importance.plot(kind='barh', x='feature',
                                         y='fscore', legend=False, figsize=(14, 18))
        if self.use_lgb:
            plt.title('LightGBM Feature Importance by F-score')
        else:
            plt.title('XGBoost Feature Importance by F-score')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        fig = fscore_plot.get_figure()
        fig.savefig('{0}{1}/{2}_{1}_{3}_feature_importance.pdf'.format(self.checkpoints_dst,
                                                                       self.run_save_name, self.prefix, self.i),
                    bbox_inches='tight')
        plt.close(fig)
        return
