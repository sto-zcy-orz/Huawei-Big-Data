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
sub = pd.read_csv('sub_features_new.csv')

#合并起点坐标
zuobiao = pd.read_csv('first_zuobiao.csv', index_col=0)
zuobiao = zuobiao.drop('id',axis=1)
sub = sub.merge(zuobiao, on='loadingOrder', how='left')

#合并终点坐标
zuobiao = pd.read_csv('last_zuobiao.csv', index_col=0)
zuobiao = zuobiao.drop('id',axis=1)
sub = sub.merge(zuobiao, on='loadingOrder', how='left')

first_x = sub['first_longitude'].tolist()
first_y = sub['first_latitude'].tolist()
last_x = sub['last_longitude'].tolist()
last_y = sub['last_latitude'].tolist()
lenx = len(first_x)
distance_df = pd.DataFrame({
    'loadingOrder': sub['loadingOrder'],
    'distance': [(first_x[i] - last_x[i]) ** 2 + (first_y[i] - last_y[i]) ** 2 for i in range(lenx)]
})
sub = sub.merge(distance_df, on='loadingOrder', how='left')
sub = sub.groupby('loadingOrder')[['distance']].agg(lambda x: x[0:1]).reset_index()
#train.to_csv('last_zuobiao.csv', index=True)
sub.to_csv('distance.csv', index=True)
print(sub)
#train = pd.read_csv('train_fea.csv')






