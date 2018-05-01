import pandas as pd

train = pd.read_csv('input/train.csv', skiprows=range(1,144903891), nrows=40000000)
train.to_csv('input/train_sample.csv')