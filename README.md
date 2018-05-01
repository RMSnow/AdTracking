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

| Num  | Category |                      Features                      |   AUC on dev set   | AUC on test set | AUC in Real World |
| ---- | :------: | :------------------------------------------------: | :----------------: | :-------------: | :---------------: |
| 1    |    1     |                      Basic #5                      | 0.9774025893305988 |                 |      0.9583       |
|      |          |                      Basic #5                      |      0.959279      |    0.963003     |      0.9559       |
| 2    |    1     |                  Basic #6 (hour)                   | 0.9765065318797589 |                 |      0.9563       |
|      |          |                Basic #7(hour, day)                 |      0.960369      |    0.964231     |      0.9563       |
| 3    |          |                      "Basic"                       | 0.974860783943336  |                 |    **0.9684**     |
| 4    |   1,2    |                     Add count                      | 0.9759619912016608 |                 |                   |
| 5    |   1,2    |                Add attributed count                | 0.9840863114093297 |                 |      0.6239       |
| 6    |   1,2    |      Add attributed count (no hour's effect)       | 0.9840863114093297 |                 |      0.6114       |
| 7    |   1,2    |           Add count and attributed count           | 0.9842108186300558 |                 |                   |
| 8    |   1,3    |                   Add Frequency                    | 0.9759619912016608 |                 |                   |
| 9    |  1,2,3   |       Add count, attributed count, frequency       | 0.9843591094736229 |                 |                   |
| 10   |   1,4    |             Add count, conversion rate             |                    |                 |                   |
| 11   | 1,2,3,4  | Add count, attributed count, frequency, conversion | 0.9842607399131638 |                 |                   |
| 12   |   1,6    |                   Add hour count                   | 0.975100704348888  |                 |                   |
| 13   |   1,6    |           Add hour attributed count auc            | 0.9952688459421872 |                 |      0.7051       |
| 14   |   1,5    |                   Add ip_channel                   | 0.979456127558631  |                 |      0.9622       |
| 15   |   1,5    |                  Add app_channel                   | 0.9783578247039497 |                 |      0.9592       |
| 16   |   1,5    |                   Add ip_device                    | 0.9801894479874732 |                 |      0.9628       |
| 17   |   1,5    |           Add All two degree of features           | 0.9813457066441867 |                 |      0.9646       |
| 18   |   1,5    |    Add All two degree of features(100,000,000)     |                    |                 |      0.9656       |
| 19   |   1,5    |   "Basic" + some features of **high_importance**   | 0.9863726922286226 |                 |    **0.9684**     |

**PS**: In `Basic#5`, `click_time` is transfered into `hour` and `day`.

### Plot Importance of Features

#### Num 3: `"Basic"`

![1](https://raw.githubusercontent.com/RMSnow/AdTracking/master/doc/img/features_importance_3.png)

#### Num 18: `Basic + All two degree of features`

![1](https://raw.githubusercontent.com/RMSnow/AdTracking/master/doc/img/features_importance_18.png)

Select `device_ip`, `channel_app`, `app_ip`, `device`, `os_ip` to add into `Num3: "Basic" `.

#### Num 19

![1](https://raw.githubusercontent.com/RMSnow/AdTracking/master/doc/img/features_importance_19.png)

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

Mask = 0

```
ip & app: 0.00927286991461
ip & device: 0.00694314823037
ip & os: 0.0295303991247
ip & channel: 0.01910882176
ip & app_device: 0.0116509685587
ip & app_os: 0.0352817805537
ip & app_channel: 0.0269119027086
ip & device_os: 0.0319996095279
ip & device_channel: 0.0232395465844
ip & os_channel: 0.0687097193232
ip & hour: 0.00842828568901

app & device: 0.0278221472785
app & os: 0.0138230238585
app & channel: 0.269795263371
app & ip_device: 0.0147259939819
app & ip_os: 0.0259135610616
app & ip_channel: 0.104454990715
app & device_os: 0.0146956128532
app & device_channel: 0.263802502515
app & os_channel: 0.174031886398
app & hour: 0.00196613604054

device & os: 0.0292843937519
device & channel: 0.0190570407725
device & ip_app: 0.010875061967
device & ip_os: 0.0123530449439
device & ip_channel: 0.0109356373285
device & app_os: 0.0164432432512
device & app_channel: 0.0215004570006
device & os_channel: 0.0157067148459
device & hour: 0.00218613697383

os & channel: 0.0111228058317
os & ip_app: 0.0392854321075
os & ip_device: 0.036077091733
os & ip_channel: 0.0587870031998
os & app_device: 0.0155230901264
os & app_channel: 0.0127829864518
os & device_channel: 0.0152616221357
os & hour: 0.000521350840747

channel & ip_app: 0.105000757011
channel & ip_device: 0.0241785886481
channel & ip_os: 0.0498727607888
channel & app_device: 0.256371969166
channel & app_os: 0.148765039685
channel & device_os: 0.0125333204848
channel & hour: 0.00297988945705

ip_app & device_os: 0.0420245202447
ip_app & device_channel: 0.109073593121
ip_app & os_channel: 0.150919655929
ip_app & hour: 0.0116179491756

ip_device & app_os: 0.0412224118725
ip_device & app_channel: 0.0327801472947
ip_device & os_channel: 0.0755055123239
ip_device & hour: 0.00926587918155

ip_os & app_device: 0.0292665954443
ip_os & app_channel: 0.0654641563309
ip_os & device_channel: 0.0560870467462
ip_os & hour: 0.0302603369959

ip_channel & app_device: 0.105314833255
ip_channel & app_os: 0.136361783362
ip_channel & device_os: 0.0618911697116
ip_channel & hour: 0.0232839554938

app_device & os_channel: 0.169265661127
app_device & hour: 0.00226247457287

app_os & device_channel: 0.149640128183
app_os & hour: 0.00113844687176

app_channel & device_os: 0.0147866713056
app_channel & hour: 0.00478454682739

device_os & hour: 0.000885698518659

device_channel & hour: 0.00329861922203

os_channel & hour: 0.00227965522251
```