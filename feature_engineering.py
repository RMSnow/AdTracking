

import pandas as pd
import numpy as np
import gc

print('load train...')

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32',
        'hour'          : 'uint8'
        }

train_df = pd.read_csv("train_sample.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

print('Extracting new features...')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
train_df['wday']  = pd.to_datetime(train_df.click_time).dt.dayofweek.astype('uint8')


basic_features = ['ip', 'app', 'device', 'os', 'channel', 'hour']

for feature in basic_features:
    gp = train_df[[feature,'day']].groupby(by=[feature])[['day']].count().reset_index().rename(index=str, columns={'day': feature + '_count'})
    train_df = train_df.merge(gp, on=[feature], how='left')

    del gp
    gc.collect()

for feature in basic_features:
    nfeature = feature + '_attributed_count'
    gp = train_df.loc[train_df['is_attributed'] == 1,[feature,'day']].groupby(by=[feature])[['day']].count().reset_index().rename(index=str, columns={'day': nfeature})
    train_df = train_df.merge(gp, on=[feature], how='left')
    del gp
    gc.collect()

    train_df[nfeature] = train_df[nfeature].fillna(0).astype(dtypes[feature])

for feature in basic_features:
    train_df[feature + '_frequency'] = train_df[feature + '_count'] / len(train_df)

for feature in basic_features:
    train_df[feature + '_conversion'] = train_df[feature + '_attributed_count'] / train_df[feature + '_count']

basic_features = ['ip', 'app', 'device', 'os', 'channel']

for feature in basic_features:
    gp = train_df[[feature, 'hour', 'day']].groupby(by=[feature, 'hour'])[['day']].count().reset_index().rename(index=str, columns={'day': feature + '_hour_count'})
    train_df = train_df.merge(gp, on=[feature,'hour'], how='left')

    del gp
    gc.collect()

for feature in basic_features:
    nfeature = feature + '_hour_attributed_count'
    gp = train_df.loc[train_df['is_attributed'] == 1, [feature, 'hour', 'day']].groupby(by=[feature, 'hour'])[['day']].count().reset_index().rename(index=str, columns={'day': nfeature})
    train_df = train_df.merge(gp, on=[feature,'hour'], how='left')

    del gp
    gc.collect()

    train_df[nfeature] = train_df[nfeature].fillna(0).astype(dtypes[feature])

for feature in basic_features:
    train_df[feature + '_hour_frequency'] = train_df[feature + '_hour_count'] / train_df['hour_count']

for feature in basic_features:
    train_df[feature + '_hour_conversion'] = train_df[feature + '_hour_attributed_count'] / train_df[feature + '_hour_count']

suffixes = ['_hour_count', '_hour_frequency', '_hour_conversion', '_hour_attributed_count']
for feature in basic_features:
    for suffix in suffixes:
        f = feature + suffix
        gp = train_df[[f, feature, 'hour']].groupby(by=[feature])[[f]].mean().reset_index().rename(index=str, columns={f: f+'_mean'})
        train_df = train_df.merge(gp, on=[feature], how='left')

        del gp
        gc.collect()

        gp = train_df[[f, feature, 'hour']].groupby(by=[feature])[[f]].std().reset_index().rename(index=str, columns={f: f+'_std'})
        train_df = train_df.merge(gp, on=[feature], how='left')

        del gp
        gc.collect()

        gp = train_df[[f, feature, 'hour']].groupby(by=[feature])[[f]].max().reset_index().rename(index=str, columns={f: f+'_max'})
        train_df = train_df.merge(gp, on=[feature], how='left')

        del gp
        gc.collect()

        train_df[f + '_mean'] = train_df[f + '_mean'].fillna(0)
        train_df[f + '_std'] = train_df[f + '_std'].fillna(0)
        train_df[f + '_max'] = train_df[f + '_max'].fillna(0)

print(train_df)



