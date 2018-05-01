import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
import gc

path = '../../data/'


def lgb_modelfit_nocv(params, dtrain, dvalid, dtest, predictors, target='target', objective='binary', metrics='auc',
                      feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10,
                      categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric': metrics,
        'learning_rate': 0.01,
        # 'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
        'metric': metrics
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgtest = lgb.Dataset(dtest[predictors].values, label=dtest[target].values,
                         feature_name=predictors,
                         categorical_feature=categorical_features
                         )

    evals_results = {}

    bst = lgb.train(lgb_params,
                    xgtrain,
                    valid_sets=[xgtrain, xgvalid, xgtest],
                    valid_names=['train', 'valid', 'test'],
                    evals_result=evals_results,
                    num_boost_round=num_boost_round,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=10,
                    feval=feval)

    n_estimators = bst.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics + ":", evals_results['valid'][metrics][n_estimators - 1])

    # save model to file
    print('Save model...')
    bst.save_model(path + 'model.txt')

    # plotting
    print('Plot metrics during training...')
    ax = lgb.plot_metric(evals_results)
    plt.show()

    print('Plot feature importances...')
    ax = lgb.plot_importance(bst, max_num_features=10)
    plt.show()

    print('Plot 84th tree...')  # one tree use categorical feature to split
    ax = lgb.plot_tree(bst, tree_index=83, figsize=(20, 8), show_info=['split_gain'])
    plt.show()

    print('Plot 84th tree with graphviz...')
    graph = lgb.create_tree_digraph(bst, tree_index=83, name='Tree84')
    graph.render(view=True)

    # ax = lgb.plot_importance(bst)
    # plt.gcf().savefig('features_importance_kernel2.png')
    return bst


dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}

# dev_250w.csv
# test_250w.csv
# train_3750w.csv
# data_shuffled_4250w.csv

print('loading train data...')
train_df = pd.read_csv(path + "train_3750w.csv", dtype=dtypes,
                       usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])

print('loading valid data...')
val_df = pd.read_csv(path + "dev_250w.csv", dtype=dtypes,
                     usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])

print('loading test data...')
test_df = pd.read_csv(path + "test_250w.csv", dtype=dtypes,
                      usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])

print('loading real world test data...')
real_test_df = pd.read_csv(path + "test.csv", dtype=dtypes,
                           usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])

print("train size: ", len(train_df))
print("valid size: ", len(val_df))
print("test size : ", len(test_df))
print("real world test size : ", len(real_test_df))


# handle 'click_time'
def handle_click_time(df):
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')


handle_click_time(train_df)
handle_click_time(val_df)
handle_click_time(test_df)
handle_click_time(real_test_df)

target = 'is_attributed'
# predictors = ['app', 'device', 'os', 'channel', 'hour', 'day']
# categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
predictors = ['ip', 'app', 'device', 'os', 'channel', 'hour', 'day']
categorical = ['ip', 'app', 'device', 'os', 'channel', 'hour', 'day']

sub = pd.DataFrame()
sub['click_id'] = real_test_df['click_id'].astype('int')

gc.collect()

print("Training...")
start_time = time.time()

params = {
    'learning_rate': 0.15,
    # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 7,  # 2^max_depth - 1
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight': 99  # because training data is extremely unbalanced
}
bst = lgb_modelfit_nocv(params,
                        train_df,
                        val_df,
                        test_df,
                        predictors,
                        target,
                        objective='binary',
                        metrics='auc',
                        early_stopping_rounds=30,
                        verbose_eval=True,
                        num_boost_round=500,
                        categorical_features=categorical)

print('[{}]: model training time'.format(time.time() - start_time))
del train_df
del val_df
del test_df
gc.collect()

print("Predicting...")
sub['is_attributed'] = bst.predict(real_test_df[predictors])
print("writing...")
sub.to_csv('sub_lgb_basic5.csv', index=False)
print("done...")
