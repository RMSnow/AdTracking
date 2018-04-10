# encoding:utf-8

# todo：计算条件熵，归一化公式为 I(X;Y) / ( H(X) + H(Y) - I(X;Y) )
import pandas as pd
import numpy as np
import time
import gc
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('me/data/test.csv')

start_time = time.time()

basic_list = ['ip', 'app', 'device', 'os', 'channel']
basic_df = df[basic_list]
basic_array = basic_df.as_matrix()

poly = PolynomialFeatures(2, interaction_only=True)
extended_array = poly.fit_transform(basic_array)

features_name = basic_list.copy()
new_feature_name = []
# 编号为1，2，3，4，5
# 则组合顺序为：1., 1,2,3,4,5, 12,13,14,15, 23,24,25, 34,35, 45
# 索引：从6-15
for i in range(len(basic_list)):
    for j in range(i + 1, len(basic_list)):
        new_feature_name.append(basic_list[i] + '_' + basic_list[j])
        features_name.append(basic_list[i] + '_' + basic_list[j])

new_data = []
for i in range(6, 16):
    new_data.append(extended_array[:, i])

for i in range(10):
    df[new_feature_name[i]] = new_data[i]

df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
features_name.append('hour')
features_sum = len(features_name)

print()
print("--------------------")
print("Time: " + str(time.time() - start_time) + "s")

path = 'doc/corr/'
mi_df = pd.read_csv(path+'mi/mutual_info.csv',index_col=0)
mi_array = mi_df.as_matrix()
correlations = np.zeros(mi_array.shape)

def calc_ent(x):
    """
        calculate shanno ent of x
    """
    start_time = time.time()
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    print("\tTime of calc_ent:" + str(time.time() - start_time) + "s")
    return ent


start_time = time.time()
for i in range(features_sum):
    print("i = " + str(i))
    for j in range(i):
        h_x = calc_ent(df[features_name[i]].as_matrix())
        h_y = calc_ent(df[features_name[j]].as_matrix())
        correlations[i, j] = mi_array[i, j] / (h_x + h_y - mi_array[i, j])
        print("\tcorr[%d,%d] = %f" % (i, j, correlations[i, j]))
        print("\tj = " + str(j) + ", Time: " + str(time.time() - start_time) + "s")

correlations_df = pd.DataFrame(index=features_name)
for i in range(features_sum):
    correlations_df[features_name[i]] = correlations[:, i]
correlations_df.to_csv(path + 'corr/mi/normalized_mi.csv')