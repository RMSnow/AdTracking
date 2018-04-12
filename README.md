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

### Performance

| Category |                      Features                      |     AUC on dev set     | AUC on test set |
| :------: | :------------------------------------------------: | :--------------------: | :-------------: |
|    1     |                      Basic #5                      | **0.9774025893305988** |     0.9583      |
|    1     |                  Basic #6 (hour)                   |   0.9765065318797589   |     0.9563      |
|          |                      "Basic"                       |   0.974860783943336    |     0.9684      |
|   1,2    |                     Add count                      |   0.9759619912016608   |                 |
|   1,2    |                Add attributed count                |   0.9840863114093297   |     0.6239      |
|   1,2    |      Add attributed count (no hour's effect)       |   0.9840863114093297   |     0.6114      |
|   1,2    |           Add count and attributed count           |   0.9842108186300558   |                 |
|   1,3    |                   Add Frequency                    |   0.9759619912016608   |                 |
|  1,2,3   |       Add count, attributed count, frequency       |   0.9843591094736229   |                 |
|   1,4    |             Add count, conversion rate             |                        |                 |
| 1,2,3,4  | Add count, attributed count, frequency, conversion |   0.9842607399131638   |                 |
|   1,6    |                   Add hour count                   |   0.975100704348888    |                 |
|   1,6    |           Add hour attributed count auc            |   0.9952688459421872   |     0.7051      |
|   1,5    |                   Add ip_channel                   |   0.979456127558631    |     0.9622      |
|   1,5    |                  Add app_channel                   |   0.9783578247039497   |     0.9592      |

### Correlations of features

#### Mutual Information

`Normalized corr_rate = I(X;Y) / H(X,Y) = I(X;Y) / (H(X) + H(Y) - I(X;Y))`

Mask = 0.1

```
app & channel: 0.269795263371
app & ip_channel: 0.104454990715
app & device_channel: 0.263802502515
app & os_channel: 0.174031886398

channel & ip_app: 0.105000757011
channel & app_device: 0.256371969166
channel & app_os: 0.148765039685

ip_app & device_channel: 0.109073593121
ip_app & os_channel: 0.150919655929

ip_channel & app_device: 0.105314833255
ip_channel & app_os: 0.136361783362

app_device & os_channel: 0.169265661127

app_os & device_channel: 0.149640128183
```

Mask = 0.2

```
app & channel: 0.269795263371
app & device_channel: 0.263802502515

channel & app_device: 0.256371969166
```