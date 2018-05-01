# encoding:utf-8
import pandas as pd
import numpy as np
import time
import datetime

# sampling：
# （1）把train.csv打乱，取4250w的数据
# （2）把4250w的数据按时间排序，认为最后600w为"高质量数据"
# （3）把最后600w打乱，再分为3部分：100w + 250w（dev set） + 250w（test set）
# （4）故最终结果：
#     train set：3650w + 100w = 3750w
#     dev set：250w
#     test set：250w

# TODO: fix the size of data
# set_size = np.array([3750, 250, 250])

start_time = time.time()
path = '../../data/'


def time_description(op):
    print("成功处理 %s ,累计用时 %f s" % (op, (time.time() - start_time)))
    print()


def handle_operation(op):
    print("当前时间为：%s，正在处理%s..." % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), op))


operation = "train.csv"
handle_operation(operation)
df = pd.read_csv(path + operation)
time_description(operation)

# sampling
operation = "sample_df"
handle_operation(operation)
sample_df = df.sample(n=42500000)
time_description(operation)

# sorted
operation = "sorted_df"
handle_operation(operation)
sorted_df = sample_df.sort_values(by='click_time')
time_description(operation)

# train, dev, test - df
len_sample = len(sorted_df)
high_quality_df = sorted_df[(len_sample - 6000000):len_sample].sample(frac=1)
high_quality_train = high_quality_df[:1000000]
dev_df = high_quality_df[len(high_quality_train):(len(high_quality_train) + 2500000)]
test_df = high_quality_df[(len(high_quality_train) + len(dev_df)):]
train_df = sorted_df[:36500000].append(high_quality_train)

# dev-output
operation = 'dev_250w.csv'
handle_operation(operation)
dev_df.to_csv(path + operation)
time_description(operation)

# test-output
operation = 'test_250w.csv'
handle_operation(operation)
test_df.to_csv(path + operation)
time_description(operation)

# train-output
operation = 'train_3750w.csv'
handle_operation(operation)
train_df.to_csv(path + operation)
time_description(operation)

# data_shuffled-output
operation = 'data_shuffled_4250w.csv'
handle_operation(operation)
sample_df.to_csv(path + operation)
time_description(operation)

'''
/home/stu/.conda/envs/deep/bin/python /home/stu/Projects/Thdlee.Snow/TalkingData/preprocess/dataset/sampling.py
当前时间为：2018-05-01 16:45:30，正在处理train.csv...
成功处理 train.csv ,累计用时 233.386754 s

当前时间为：2018-05-01 16:49:23，正在处理sample_df...
成功处理 sample_df ,累计用时 258.572500 s

当前时间为：2018-05-01 16:49:48，正在处理sorted_df...
成功处理 sorted_df ,累计用时 339.554489 s

当前时间为：2018-05-01 16:51:14，正在处理dev_250w.csv...
成功处理 dev_250w.csv ,累计用时 357.362650 s

当前时间为：2018-05-01 16:51:27，正在处理test_250w.csv...
成功处理 test_250w.csv ,累计用时 370.246952 s

当前时间为：2018-05-01 16:51:40，正在处理train_3750w.csv...
成功处理 train_3750w.csv ,累计用时 554.877350 s

当前时间为：2018-05-01 16:54:44，正在处理data_shuffled_4250w.csv...
成功处理 data_shuffled_4250w.csv ,累计用时 776.042315 s


Process finished with exit code 0
'''
