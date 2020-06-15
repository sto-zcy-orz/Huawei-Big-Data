import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import KFold
import lightgbm as lgb

import warnings

warnings.filterwarnings('ignore')

# baseline只用到gps定位数据，即train_gps_path
train_gps_path = './data/train0523.csv'
test_data_path = './data/A_testData0531.csv'
order_data_path = './data/loadingOrderEvent.csv'
port_data_path = './data/port.csv'
split_data_path = './data/split'
split_last_path = './data/split_last'
file_list = os.listdir(split_data_path)
last_list = os.listdir(split_last_path)

# 取前1000000行
debug = True
NDATA = 1000000


test_data = pd.read_csv(test_data_path)
port_data = pd.read_csv(port_data_path)
def geohash_encode(latitude, longitude, precision=5):
    """
    Encode a position given in float arguments latitude, longitude to
    a geohash which will have the character count precision.
    """
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True
    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash += base32[ch]
            bit = 0
            ch = 0
    return ''.join(geohash)

def get_data(data, mode='train'):
    assert mode == 'train' or mode == 'test'

    if mode == 'train':
        data['vesselNextportETA'] = pd.to_datetime(data['vesselNextportETA'], infer_datetime_format=True)
    elif mode == 'test':
        data['temp_timestamp'] = data['timestamp']
        data['onboardDate'] = pd.to_datetime(data['onboardDate'], infer_datetime_format=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'], infer_datetime_format=True)
    data['longitude'] = data['longitude'].astype(float)
    data['loadingOrder'] = data['loadingOrder'].astype(str)
    data['latitude'] = data['latitude'].astype(float)
    data['speed'] = data['speed'].astype(float)
    data['direction'] = data['direction'].astype(float)

    return data

# 代码参考：https://github.com/juzstu/TianChi_HaiYang
def get_feature(df, mode='train'):
    assert mode == 'train' or mode == 'test'

    df.sort_values(['loadingOrder', 'timestamp'], inplace=True)
    # 特征只选择经纬度、速度\方向
    df['lat_diff'] = df.groupby('loadingOrder')['latitude'].diff(1)
    df['lon_diff'] = df.groupby('loadingOrder')['longitude'].diff(1)
    df['speed_diff'] = df.groupby('loadingOrder')['speed'].diff(1)
    df['diff_minutes'] = df.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds() // 60
    df['anchor'] = df.apply(lambda x: 1 if abs(x['lat_diff']) <= 0.03 and abs(x['lon_diff']) <= 0.03
                                           and abs(x['speed_diff']) <= 0.3 and x['diff_minutes'] <= 10 else 0, axis=1)

    if mode == 'train':
        group_df = df.groupby('loadingOrder')['timestamp'].agg(mmax='max', count='count', mmin='min').reset_index()
        # 读取数据的最大值-最小值，即确认时间间隔为label
        group_df['label'] = (group_df['mmax'] - group_df['mmin']).dt.total_seconds()
    elif mode == 'test':
        group_df = df.groupby('loadingOrder')['timestamp'].agg(count='count').reset_index()

    anchor_df = df.groupby('loadingOrder')['anchor'].agg('sum').reset_index()
    anchor_df.columns = ['loadingOrder', 'anchor_cnt']
    group_df = group_df.merge(anchor_df, on='loadingOrder', how='left')
    group_df['anchor_ratio'] = group_df['anchor_cnt'] / group_df['count']

    agg_function = ['min', 'max', 'mean', 'median']
    agg_col = ['latitude', 'longitude', 'speed', 'direction']

    group = df.groupby('loadingOrder')[agg_col].agg(agg_function).reset_index()
    group.columns = ['loadingOrder'] + ['{}_{}'.format(i, j) for i in agg_col for j in agg_function]
    group_df = group_df.merge(group, on='loadingOrder', how='left')

    return group_df

num = 0
train = pd.DataFrame(columns=['id', 'loadingOrder', 'first_geohash'])
print(len(file_list))
for file in file_list:
    num += 1
    print(num)
    file_path = os.path.join(split_data_path, file)
    train_data_o = pd.read_csv(file_path, nrows=1, header=None)
    train_data_o.columns = ['id', 'loadingOrder', 'carrierName', 'timestamp', 'longitude',
                          'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                          'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']
    train_o = pd.DataFrame({
        'id': train_data_o['id'][0],
        'loadingOrder': train_data_o['loadingOrder'][0],
        'first_geohash': [geohash_encode(train_data_o['longitude'][0], train_data_o['latitude'][0])],
    })
    train = train.append(train_o, ignore_index=True)

#train.to_csv('last_zuobiao.csv', index=True)
train.to_csv('first_geohash.csv', index=True)
print(train)

#for last
num = 0
train = pd.DataFrame(columns=['id', 'loadingOrder', 'last_geohash'])
print(len(last_list))
for file in last_list:
    num += 1
    print(num)
    file_path = os.path.join(split_last_path, file)
    train_data_o = pd.read_csv(file_path, nrows=1, header=None)
    train_data_o.columns = ['id', 'loadingOrder', 'carrierName', 'timestamp', 'longitude',
                          'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                          'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']
    train_o = pd.DataFrame({
        'id': train_data_o['id'][0],
        'loadingOrder': train_data_o['loadingOrder'][0],
        'last_geohash': [geohash_encode(train_data_o['longitude'][0], train_data_o['latitude'][0])],
    })
    train = train.append(train_o, ignore_index=True)

#train.to_csv('last_zuobiao.csv', index=True)
train.to_csv('last_geohash.csv', index=True)
print(train)
#train = pd.read_csv('train_fea.csv')





