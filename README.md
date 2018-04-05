## Feature Engineering

### Data Fields

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

### Feature Selection

#### Basic Features (#5)

ip, app, device, os, channel

#### Basic Features' attributed contributors (#5)

For each **unique value** of every basic feature, count the number of click that **is attributed**.

#### Frequencies of Basic Features (#5)

For each **unique value** of every basic feature, calculate the value's frequency in the whole dataset.

#### Conversion Rate (#5)

For each **unique value** of every basic feature, calculate the values conversion rate (i.e. the fraction, `#is_attributed clicks / #clicks`).

#### Correlated Features' Combination (#n)

Select the features' combination whose features own a high value of correlation.

#### Temporal Extraction

As for different time span(whole time, minute, and hour), calculate `raw` , `average` and `standard deviation` of all the features above.

So the amount of all above features is `3 * (20 + n) * 3`.

#### Temporal Conversion Rate (#1)

Calculate every hour's conversion rate.

#### Others

eg: temporal interval