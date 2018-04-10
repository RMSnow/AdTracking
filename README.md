# Feature Engineering

## Data Fields

| Features        | Descriptions |
| --------------- | ------------ |
| ip              | Not Null     |
| app             | Not Null     |
| device          | Not Null     |
| os              | Not Null     |
| channel         | Not Null     |
| click_time      | Not Null     |
| attributed_time |              |
| is_attributed   | 0 / 1        |

## Feature Selection Categories

### 1 Basic Features (#5)

ip, app, device, os, channel

### 2 Basic Features' attributed contributors (#5 * 2)

For each **unique value** of every basic feature, count the number of click that **is attributed or not**.

### 3 Frequencies of Basic Features (#5)

For each **unique value** of every basic feature, calculate the value's frequency in the whole dataset.

### 4 Conversion Rate (#5)

For each **unique value** of every basic feature, calculate the values conversion rate (i.e. the fraction, `#is_attributed clicks / #clicks`).

### 5 Correlated Features' Combination (#n)

Select the features' combination whose features own a high value of correlation.

### 6 Temporal Extraction

As for different time span(whole time, minute, and hour), calculate `raw` , `average` and `standard deviation` of all the features above.

So the amount of all above features is `3 * (20 + n) * 3`.

### 7 Temporal Conversion Rate (#1)

Calculate every hour's conversion rate.

### 8 Others

eg: temporal interval

## Experiments

| Category |                 Features                 |   AUC on dev set   | AUC on test set |
| :------: | :--------------------------------------: | :----------------: | :-------------: |
|    1     |                 Basic #5                 | 0.974860783943336  |     0.9684      |
|   1,2    |                Add count                 | 0.9759619912016608 |                 |
|   1,2    |           Add attributed count           | 0.9840863114093297 |     0.6114      |
|   1,2    |      Add count and attributed count      | 0.9842108186300558 |                 |
|   1,3    |              Add Frequency               | 0.9759619912016608 |                 |
|  1,2,3   |  Add count, attributed count, frequency  | 0.9843591094736229 |                 |
| 1,2,3,4  | Add count, attributed count, frequency, conversion | 0.9842607399131638 |                 |
|          |              Add hour count              | 0.975100704348888  |                 |
|          |      Add hour attributed count auc       | 0.9952688459421872 |     0.6239      |