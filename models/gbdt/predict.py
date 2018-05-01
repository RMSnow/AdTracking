# encoding:utf-8
import pandas as pd
import time
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import gc

path = '../../data/'

dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}

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

bst = lgb.Booster(model_file=path + 'model/model.txt')

xgtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
xgvalid = lgb.Dataset(val_df[predictors].values, label=val_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
xgtest = lgb.Dataset(test_df[predictors].values, label=test_df[target].values,
                         feature_name=predictors,
                         categorical_feature=categorical)

bst.eval(xgtrain,'train')
bst.eval(xgvalid,'valid')
bst.eval(xgtest,'test')

'''
# Predict
print("Predicting...")
sub['is_attributed'] = bst.predict(real_test_df[predictors])
print("writing...")
sub.to_csv(path + 'sub/sub_lgb_basic7.csv', index=False)
print("done...")
'''