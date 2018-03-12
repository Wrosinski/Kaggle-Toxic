import gc
import itertools
import multiprocessing
import time
from collections import Counter

import numpy as np
import pandas as pd


def create_customer_feature_set(train):

    customer_feats = pd.DataFrame()
    customer_feats['customer_id'] = train.customer_id

    customer_feats['customer_max_ratio'] = train.customer_id / \
        np.max(train.customer_id)
    customer_feats['index_max_ratio'] = train.customer_id / \
        (train.index + 1e-14)
    customer_feats['customer_count'] = train.customer_id.map(
        train.customer_id.value_counts())

    customer_feats['cust_first'] = train.customer_id.apply(
        lambda x: int(str(x)[:1]))
    customer_feats['cust_2first'] = train.customer_id.apply(
        lambda x: int(str(x)[:2]))
    customer_feats['cust_3first'] = train.customer_id.apply(
        lambda x: int(str(x)[:3]))
    customer_feats['cust_4first'] = train.customer_id.apply(
        lambda x: int(str(x)[:4]))
    customer_feats['cust_6first'] = train.customer_id.apply(
        lambda x: int(str(x)[:6]))
    # customer_feats.cust_3first = pd.factorize(customer_feats.cust_3first)[0]

    customer_feats.drop(['customer_id'], axis=1, inplace=True)

    return customer_feats


def create_groupings_feature_set(data, features, transform=True):

    df_features = pd.DataFrame()

    year_mean = group_feat_by_feat(
        data, 'year', features, 'mean', transform)
    year_max = group_feat_by_feat(
        data, 'year', features, 'max', transform)
    year_count = group_feat_by_feat(
        data, 'year', features, 'count', transform)

    month_mean = group_feat_by_feat(
        data, 'month', features, 'mean', transform)
    month_max = group_feat_by_feat(
        data, 'month', features, 'max', transform)
    month_count = group_feat_by_feat(
        data, 'month', features, 'count', transform)

    market_mean = group_feat_by_feat(
        data, 'market', features, 'mean', transform)
    market_max = group_feat_by_feat(
        data, 'market', features, 'max', transform)
    market_count = group_feat_by_feat(
        data, 'market', features, 'count', transform)

    customer_mean = group_feat_by_feat(
        data, 'customer_id', features, 'mean', transform)
    customer_max = group_feat_by_feat(
        data, 'customer_id', features, 'max', transform)
    customer_count = group_feat_by_feat(
        data, 'customer_id', features, 'count', transform)

    df_features = pd.concat([year_mean, year_max, year_count, month_mean, month_max, month_count,
                             market_mean, market_max, market_count,
                             customer_mean, customer_max, customer_count], axis=1)

    del year_mean, year_max, year_count, month_mean, month_max, month_count, \
        market_mean, market_max, market_count, \
        customer_mean, customer_max, customer_count
    gc.collect()

    return df_features


def create_aggregated_lags(df, current_month,
                           only_target=False, features=None, agg_func='mean',
                           month_merge=False):

    assert(current_month > 0 and current_month < 16)
    if month_merge:
        df_result = df[df.date == current_month][['customer_id']]
    else:
        df_result = df[['customer_id']]

    print('Creating grouping features based on aggregated data before {} date.'.format(
        current_month))
    print('Beginning shape:', df_result.shape)
    if features is not None:
        if 'customer_id' not in features:
            features.append('customer_id')

    df_lag = df[df.date < current_month]

    if only_target:
        df_lag = df_lag[['customer_id', 'target']].groupby(
            'customer_id', as_index=False).agg('{}'.format(agg_func))
    else:
        if features is not None:
            df_lag = df_lag[features].groupby(
                'customer_id', as_index=False).agg('{}'.format(agg_func))

    df_lag.columns = ['{}_lag_agg'.format(
        x) if 'customer' not in x else x for x in df_lag.columns]
    df_result = df_result.merge(
        df_lag, on=['customer_id'], how='left', copy=False)

    to_drop = [x for x in df_result.columns if 'customer' in x]
    df_result.drop(to_drop, axis=1, inplace=True)
    print('Final shape:', df_result.shape)
    return df_result


def create_lag_features(df, current_month=1, start_lag=0, incremental=False,
                        only_target=False, features=None, agg_func='mean',
                        month_merge=False):

    if month_merge:
        df_result = df[df.date == current_month][['customer_id', 'target']]
    else:
        df_result = df[['customer_id', 'target']]

    lag_subset = np.arange(start_lag, current_month, 1)
    print('Beginning shape:', df_result.shape, 'Lag subset:', lag_subset)
    if features is not None:
        if 'customer_id' not in features:
            features.append('customer_id')

    if incremental:
        print('Creating grouping features based on incremental lags.')
    if not incremental:
        print('Creating grouping features based on non-incremental lags.')
        print('For non-incremental lags only mean aggregation can be used - switch to it.')
        agg_func = 'mean'

    for i in range(len(lag_subset)):
        if incremental:
            print('Dates subset:', lag_subset[lag_subset <= lag_subset[i]])
            df_lag = df[df.date <= lag_subset[i]]
        else:
            df_lag = df[df.date == lag_subset[i]]

        if only_target:
            df_lag = df_lag[['customer_id', 'target']].groupby(
                'customer_id', as_index=False).agg('{}'.format(agg_func))
        else:
            if features is not None:
                df_lag = df_lag[features].groupby(
                    'customer_id', as_index=False).agg('{}'.format(agg_func))

        df_lag.columns = ['{}_lag{}'.format(
            x, i) if 'customer' not in x else x for x in df_lag.columns]
        df_result = df_result.merge(
            df_lag, on=['customer_id'], how='left', copy=False)

    to_drop = [x for x in df_result.columns if 'customer' in x]
    to_drop.append('target')
    df_result.drop(to_drop, axis=1, inplace=True)
    print('Final shape:', df_result.shape)
    return df_result


def prepare_lags_data(train, test,
                      start_train=1, end_train=11,
                      start_test=12, end_test=15,
                      only_target=False, features=None,
                      incremental=False, agg_func='mean'):

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    print('Create training set.\n')
    for i in range(start_train, end_train + 1, 1):
        if incremental:
            lag_features = create_lag_features(
                train, i, start_train, incremental=incremental,
                only_target=only_target, features=features, agg_func=agg_func)
        else:
            lag_features = create_lag_features(
                train, i, i - 1, incremental=incremental,
                only_target=only_target, features=features, agg_func=agg_func)
        df_train = pd.concat([df_train, lag_features])
        print('Current train shape:', df_train.shape)

    print('\nCreate test set.\n')
    for i in range(start_test, end_test + 1, 1):
        if incremental:
            lag_features = create_lag_features(
                test, i, start_test, incremental=incremental,
                only_target=only_target, features=features, agg_func=agg_func)
        else:
            lag_features = create_lag_features(
                test, i, i - 1, incremental=incremental,
                only_target=only_target, features=features, agg_func=agg_func)
        df_test = pd.concat([df_test, lag_features])
        print('Current test shape:', df_test.shape)

    print('Final shapes:', df_train.shape, df_test.shape)
    df_train.drop(['target'], axis=1, inplace=True)
    df_train.reset_index(inplace=True, drop=True)
    df_test.drop(['target'], axis=1, inplace=True)
    df_test.reset_index(inplace=True, drop=True)
    return df_train, df_test


def prepare_aggregated_lags(train, test,
                            start_train=0, end_train=11,
                            start_test=12, end_test=15,
                            only_target=False,
                            features=None, agg_func='mean'):

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    print('Create training set.\n')
    for i in range(start_train, end_train + 1, 1):
        lag_features = create_aggregated_lags(
            train, i,
            only_target=only_target, features=features, agg_func=agg_func)
        df_train = pd.concat([df_train, lag_features])

    print('\nCreate test set.\n')
    for i in range(start_test, end_test + 1, 1):
        lag_features = create_aggregated_lags(
            test, i,
            only_target=only_target, features=features, agg_func=agg_func)
        df_test = pd.concat([df_test, lag_features])

    print('Final shapes:', df_train.shape, df_test.shape)
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)
    return df_train, df_test


def labelcount_encode(df, categorical_features, ascending=False):
    print('LabelCount encoding:', categorical_features)
    new_df = pd.DataFrame()
    for cat_feature in categorical_features:
        cat_feature_value_counts = df[cat_feature].value_counts()
        value_counts_list = cat_feature_value_counts.index.tolist()
        if ascending:
            # for ascending ordering
            value_counts_range = list(
                reversed(range(len(cat_feature_value_counts))))
        else:
            # for descending ordering
            value_counts_range = list(range(len(cat_feature_value_counts)))
        labelcount_dict = dict(zip(value_counts_list, value_counts_range))
        new_df[cat_feature] = df[cat_feature].map(
            labelcount_dict)
    new_df.columns = ['{}_lc_encode'.format(x) for x in new_df.columns]
    return new_df


def count_encode(df, categorical_features, normalize=False):
    print('Count encoding:', categorical_features)
    new_df = pd.DataFrame()
    for cat_feature in categorical_features:
        new_df[cat_feature] = df[cat_feature].astype(
            'object').map(df[cat_feature].value_counts())
        if normalize:
            new_df[cat_feature] = new_df[cat_feature] / np.max(new_df[cat_feature])
    new_df.columns = ['{}_count_encode'.format(x) for x in new_df.columns]
    return new_df


def target_encode(df_train, df_test, categorical_features, smoothing=10):
    print('Target encoding:', categorical_features)
    df_te_train = pd.DataFrame()
    df_te_test = pd.DataFrame()
    for cat_feature in categorical_features:
        df_te_train[cat_feature], df_te_test[cat_feature] = \
            target_encode_feature(
                df_train[cat_feature], df_test[cat_feature], df_train.target, smoothing)
    df_te_train.columns = ['{}_target_encode'.format(x) for x in df_te_train.columns]
    df_te_test.columns = ['{}_target_encode'.format(x) for x in df_te_test.columns]
    return df_te_train, df_te_test


def bin_numerical(df, cols, step):
    numerical_features = cols
    new_df = pd.DataFrame()
    for i in numerical_features:
        try:
            feature_range = np.arange(0, np.max(df[i]), step)
            new_df['{}_binned'.format(i)] = np.digitize(
                df[i], feature_range, right=True)
        except ValueError:
            df[i] = df[i].replace(np.inf, 999)
            feature_range = np.arange(0, np.max(df[i]), step)
            new_df['{}_binned'.format(i)] = np.digitize(
                df[i], feature_range, right=True)
    return new_df


def add_statistics(df, features_list):
    X = pd.DataFrame()
    X['sum_row_{}cols'.format(len(features_list))
      ] = df[features_list].sum(axis=1)
    X['mean_row{}cols'.format(len(features_list))
      ] = df[features_list].mean(axis=1)
    X['std_row{}cols'.format(len(features_list))
      ] = df[features_list].std(axis=1)
    X['max_row{}cols'.format(len(features_list))] = np.amax(
        df[features_list], axis=1)
    print('Statistics of {} columns done.'.format(features_list))
    return X


def feature_combinations(df, features_list):
    X = pd.DataFrame()
    for comb in itertools.combinations(features_list, 2):
        feat = comb[0] + "_" + comb[1]
        X[feat] = df[comb[0]] * df[comb[1]]
    print('Interactions on {} columns done.'.format(features_list))
    return X


def group_feat_by_feat(df, feature_by, features_to, statistic='mean', transform=False):
    X = pd.DataFrame()
    for i in range(len(features_to)):
        if statistic == 'mean':
            if transform:
                X['{0}_by_{1}_{2}'.format(feature_by, features_to[i], statistic)] = (
                    df.groupby(feature_by))[features_to[i]].transform('{}'.format(statistic))
            else:
                X['{0}_by_{1}_{2}'.format(feature_by, features_to[i], statistic)] = (
                    df.groupby(feature_by))[features_to[i]].mean()
        if statistic == 'max':
            if transform:
                X['{0}_by_{1}_{2}'.format(feature_by, features_to[i], statistic)] = (
                    df.groupby(feature_by))[features_to[i]].transform('{}'.format(statistic))
            else:
                X['{0}_by_{1}_{2}'.format(feature_by, features_to[i], statistic)] = (
                    df.groupby(feature_by))[features_to[i]].max()
        if statistic == 'min':
            if transform:
                X['{0}_by_{1}_{2}'.format(feature_by, features_to[i], statistic)] = (
                    df.groupby(feature_by))[features_to[i]].transform('{}'.format(statistic))
            else:
                X['{0}_by_{1}_{2}'.format(feature_by, features_to[i], statistic)] = (
                    df.groupby(feature_by))[features_to[i]].min()
        if statistic == 'count':
            if transform:
                X['{0}_by_{1}_{2}'.format(feature_by, features_to[i], statistic)] = (
                    df.groupby(feature_by))[features_to[i]].transform('{}'.format(statistic))
            else:
                X['{0}_by_{1}_{2}'.format(feature_by, features_to[i], statistic)] = (
                    df.groupby(feature_by))[features_to[i]].count()
    if not transform:
        X['{}'.format(feature_by)] = X.index
        X.reset_index(inplace=True, drop=True)
    print('Groupings of {} columns by: {} done using {} statistic.'.format(features_to, feature_by,
                                                                           statistic))
    return X


def group_feat_by_feat_multiple(df, feature_by, features_to, statistic='mean', transform=False):
    X = pd.DataFrame()
    if statistic == 'mean':
        if transform:
            X = (df.groupby(feature_by))[
                features_to].transform('{}'.format(statistic))
        else:
            X = (df.groupby(feature_by))[features_to].mean()
    if statistic == 'max':
        if transform:
            X = (df.groupby(feature_by))[
                features_to].transform('{}'.format(statistic))
        else:
            X = (df.groupby(feature_by))[features_to].max()
    if statistic == 'min':
        if transform:
            X = (df.groupby(feature_by))[
                features_to].transform('{}'.format(statistic))
        else:
            X = (df.groupby(feature_by))[features_to].min()
    if statistic == 'count':
        if transform:
            X = (df.groupby(feature_by))[
                features_to].transform('{}'.format(statistic))
        else:
            X = (df.groupby(feature_by))[features_to].count()
    X.columns = ['{}_by_{}_{}'.format(
        feature_by, i, statistic) for i in features_to]
    print('Groupings of {} columns by: {} done using {} statistic.'.format(features_to, feature_by,
                                                                           statistic))
    return X


def group_feat_by_feat_list(df, features_list, transformation):
    X = pd.DataFrame()
    for i in range(len(features_list) - 1):
        X['{0}_by_{1}_{2}'.format(features_list[i], features_list[i + 1], transformation)] = (
            df.groupby(features_list[i]))[features_list[i + 1]].transform('{}'.format(transformation))
    print('Groupings of {} columns done using {} transformation.'.format(
        features_list, transformation))
    return X


def feature_combinations_grouping(df, features_list, transformation):
    X = pd.DataFrame()
    for comb in itertools.combinations(features_list, 2):
        X['{}_by_{}_{}_combinations'.format(comb[0], comb[1], transformation)] = (
            df.groupby(comb[0]))[comb[1]].transform('{}'.format(transformation))
    print('Groupings of {} columns done using {} transformation.'.format(
        features_list, transformation))
    return X


def get_duplicate_cols(df):
    dfc = df.sample(n=10000)
    dfc = dfc.T.drop_duplicates().T
    duplicate_cols = sorted(list(set(df.columns).difference(set(dfc.columns))))
    print('Duplicate columns:', duplicate_cols)
    del dfc
    gc.collect()
    return duplicate_cols


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode_feature(trn_series=None,
                          tst_series=None,
                          target=None,
                          min_samples_leaf=100,
                          smoothing=10,
                          noise_level=1e-3):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[
        target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / \
        (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * \
        (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(
            columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(
            columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
