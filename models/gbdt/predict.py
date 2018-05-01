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

print('loading real world test data...')
real_test_df = pd.read_csv(path + "test.csv", dtype=dtypes,
                           usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])


# handle 'click_time'
def handle_click_time(df):
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')


handle_click_time(real_test_df)

target = 'is_attributed'
# predictors = ['app', 'device', 'os', 'channel', 'hour', 'day']
# categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
predictors = ['ip', 'app', 'device', 'os', 'channel', 'hour', 'day']
categorical = ['ip', 'app', 'device', 'os', 'channel', 'hour', 'day']

sub = pd.DataFrame()
sub['click_id'] = real_test_df['click_id'].astype('int')

gc.collect()

# Predict
bst = lgb.Booster(model_file=path + 'model.txt')
print("Predicting...")
sub['is_attributed'] = bst.predict(real_test_df[predictors])
print("writing...")
sub.to_csv(path + 'sub_lgb_basic5.csv', index=False)
print("done...")