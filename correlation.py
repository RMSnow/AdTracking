# encoding:utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

path = 'me/data/'
df = pd.read_csv(path + 'train.csv', nrows=100000)

print("---------------源feature---------------")
print(df.head())
print()
print("---------------源feature相关性---------------")
df.corr().to_csv(path + 'corr/corr_basic.csv')
print(df.corr())
print()

basic_list = ['ip', 'app', 'device', 'os', 'channel']
basic_df = df[basic_list]
basic_array = basic_df.as_matrix()

poly = PolynomialFeatures(2, interaction_only=True)
extended_array = poly.fit_transform(basic_array)

new_feature_name = []
# 编号为1，2，3，4，5
# 则组合顺序为：1., 1,2,3,4,5, 12,13,14,15, 23,24,25, 34,35, 45
# 索引：从6-15
for i in range(len(basic_list)):
    for j in range(i + 1, len(basic_list)):
        new_feature_name.append(basic_list[i] + '_' + basic_list[j])

new_data = []
for i in range(6, 16):
    new_data.append(extended_array[:, i])

for i in range(10):
    df[new_feature_name[i]] = new_data[i]

print("---------------组合basic feature---------------")
print(df.head())
print()
print("---------------新feature相关性---------------")
df.corr().to_csv(path + 'corr/corr_extended.csv')
print(df.corr())

df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
df['wday'] = pd.to_datetime(df.click_time).dt.dayofweek.astype('uint8')

print("---------------添加temporal feature---------------")
print(df.head())
print()
print("---------------新feature相关性---------------")
df.corr().to_csv(path + 'corr/corr_temporal.csv')
print(df.corr())

correlations = df.corr()
names = correlations.index.tolist()
# plot correlation matrix
fig = plt.figure()  # 调用figure创建一个绘图对象
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)  # 绘制热力图，从-1到1
fig.colorbar(cax)  # 将matshow生成热力图设置为颜色渐变条
ticks = np.arange(0, len(names), 1)  # 生成0-9，步长为1
ax.set_xticks(ticks)  # 生成刻度
ax.set_yticks(ticks)
ax.set_xticklabels(names, rotation='vertical')  # 生成x轴标签
ax.set_yticklabels(names)
plt.savefig(path + 'corr/corr.png')
plt.show()
