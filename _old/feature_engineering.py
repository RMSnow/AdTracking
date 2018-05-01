

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
path = 'input/'


print('loading train data...')
train_df = pd.read_csv(path+"train.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

print('loading test data...')
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

len_train = len(train_df)
train_df=train_df.append(test_df)

del test_df
gc.collect()

print('Extracting new features...')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
train_df['wday']  = pd.to_datetime(train_df.click_time).dt.dayofweek.astype('uint8')
print("features about time have done..." + " features: %d" % len(train_df.keys()))

basic_features = ['ip', 'app', 'device', 'os', 'channel', 'hour']

for feature in basic_features:
    gp = train_df[[feature,'day']].groupby(by=[feature])[['day']].count().reset_index().rename(index=str, columns={'day': feature + '_count'})
    train_df = train_df.merge(gp, on=[feature], how='left')

    del gp
    gc.collect()
print("features counting have done..." + " features: %d" % len(train_df.keys()))

for feature in basic_features:
    nfeature = feature + '_attributed_count'
    gp = train_df.loc[train_df['is_attributed'] == 1,[feature,'day']].groupby(by=[feature])[['day']].count().reset_index().rename(index=str, columns={'day': nfeature})
    train_df = train_df.merge(gp, on=[feature], how='left')
    del gp
    gc.collect()

    train_df[nfeature] = train_df[nfeature].fillna(0).astype(dtypes[feature])
print("attributed features about time have done..." + " features: %d" % len(train_df.keys()))

for feature in basic_features:
    train_df[feature + '_frequency'] = train_df[feature + '_count'] / len(train_df)
print("features frequency have done..." + " features: %d" % len(train_df.keys()))

for feature in basic_features:
    train_df[feature + '_conversion'] = train_df[feature + '_attributed_count'] / train_df[feature + '_count']
print("features conversion have done..." + " features: %d" % len(train_df.keys()))

basic_features = ['ip', 'app', 'device', 'os', 'channel']

for feature in basic_features:
    gp = train_df[[feature, 'hour', 'day']].groupby(by=[feature, 'hour'])[['day']].count().reset_index().rename(index=str, columns={'day': feature + '_hour_count'})
    train_df = train_df.merge(gp, on=[feature,'hour'], how='left')

    del gp
    gc.collect()
print("features hour counting have done..." + " features: %d" % len(train_df.keys()))

for feature in basic_features:
    nfeature = feature + '_hour_attributed_count'
    gp = train_df.loc[train_df['is_attributed'] == 1, [feature, 'hour', 'day']].groupby(by=[feature, 'hour'])[['day']].count().reset_index().rename(index=str, columns={'day': nfeature})
    train_df = train_df.merge(gp, on=[feature,'hour'], how='left')

    del gp
    gc.collect()

    train_df[nfeature] = train_df[nfeature].fillna(0).astype(dtypes[feature])
print("attributed features counting have done..." + " features: %d" % len(train_df.keys()))

for feature in basic_features:
    train_df[feature + '_hour_frequency'] = train_df[feature + '_hour_count'] / train_df['hour_count']
print("features hour frequency have done..." + " features: %d" % len(train_df.keys()))

for feature in basic_features:
    train_df[feature + '_hour_conversion'] = train_df[feature + '_hour_attributed_count'] / train_df[feature + '_hour_count']
print("features hour conversion have done..." + " features: %d" % len(train_df.keys()))

suffixes = ['_hour_count', '_hour_frequency', '_hour_conversion', '_hour_attributed_count']
'''
for feature in basic_features:
    for suffix in suffixes:
        f = feature + suffix
        gp = train_df[[f, feature, 'hour']].groupby(by=['hour'])[[f]].mean().reset_index().rename(index=str, columns={f: f+'_hour_mean'})
        train_df = train_df.merge(gp, on=['hour'], how='left')

        del gp
        gc.collect()

        gp = train_df[[f, feature, 'hour']].groupby(by=['hour'])[[f]].std().reset_index().rename(index=str, columns={f: f+'_hour_std'})
        train_df = train_df.merge(gp, on=['hour'], how='left')

        del gp
        gc.collect()

        gp = train_df[[f, feature, 'hour']].groupby(by=['hour'])[[f]].max().reset_index().rename(index=str, columns={f: f+'_hour_max'})
        train_df = train_df.merge(gp, on=['hour'], how='left')

        del gp
        gc.collect()

        train_df[f + '_hour_mean'] = train_df[f + '_hour_mean'].fillna(0)
        train_df[f + '_hour_std'] = train_df[f + '_hour_std'].fillna(0)
        train_df[f + '_hour_max'] = train_df[f + '_hour_max'].fillna(0)
print("hour with features computing have done..." + " features: %d" % len(train_df.keys()))
'''
def hour_data_process(train_df, feature, suffixes):
    unique = train_df[feature].unique()
    df = pd.DataFrame(np.zeros((len(unique), 24)), index=unique, columns=range(24), dtype=np.uint32)
    df = df.stack().reset_index().rename(index=str, columns={'level_0': feature, 'level_1': 'hour'})

    for suffix in suffixes:
        nfeature = feature + suffix
        gp = train_df[[nfeature, feature, 'hour']].groupby(by=[feature, 'hour'], as_index=False).count()
        gp = df.merge(gp, on=[feature, 'hour'], how='left')
        gp[nfeature] = gp[nfeature].fillna(0)
        mean = gp.groupby(by=[feature])[[nfeature]].mean().rename(index=str,
                                                                      columns={nfeature: nfeature + '_mean'})
        std = gp.groupby(by=[feature])[[nfeature]].std().rename(index=str,
                                                                    columns={nfeature: nfeature + '_std'})
        max = gp.groupby(by=[feature])[[nfeature]].max().rename(index=str,
                                                                    columns={nfeature: nfeature + '_max'})
        if suffix.endswith('count'): max[nfeature + '_max'] = max[nfeature + '_max'].astype('uint32')
        gp = pd.concat([max, mean, std], axis=1, join='inner')
        train_df = train_df.merge(gp.reset_index(), on=[feature], how='left')

        del gp, mean, std, max
        gc.collect()

    del df
    gc.collect()
    return train_df


#for feature in basic_features:
#    train_df = hour_data_process(train_df, feature, suffixes)
#print("features with hour computing have done..." + " features: %d" % len(train_df.keys()))

print("\nAll features: ")
for key in train_df.keys(): print(key)
print("length: %d" % (len(train_df.keys())-2))

test_df = train_df[len_train:]
train_df = train_df[:len_train]
for key in test_df.keys(): print(key)
print('writing train file...')
train_df.to_csv("train_feature_engineering.csv",index=False)
print('writing test file...')
test_df.to_csv("test_feature_engineering.csv",index=False)