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
file_list = os.listdir(split_data_path)

# 取前1000000行
debug = True
NDATA = 1000000


test_data = pd.read_csv(test_data_path)


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
train = pd.DataFrame(columns=['id', 'loadingOrder', 'vesselMMSI', 'TRANSPORT_TRACE_start', 'TRANSPORT_TRACE_final'])
print(len(file_list))
for file in file_list:
    num += 1
    print(num)
    file_path = os.path.join(split_data_path, file)
    train_data_o = pd.read_csv(file_path, nrows=1, header=None)
    train_data_o.columns = ['id', 'loadingOrder', 'carrierName', 'timestamp', 'longitude',
                          'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                          'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']
    if ('-' not in str(train_data_o['TRANSPORT_TRACE'][0])): continue
    s = train_data_o['TRANSPORT_TRACE'][0].split('-')
    train_o = pd.DataFrame({
        'id': train_data_o['id'][0],
        'loadingOrder': train_data_o['loadingOrder'][0],
        'vesselMMSI': train_data_o['vesselMMSI'][0],
        'TRANSPORT_TRACE_start': [s[0].replace(" ", "").upper()],
        'TRANSPORT_TRACE_final': [s[-1].replace(" ", "").upper()]
    })
    train = train.append(train_o, ignore_index=True)

train.to_csv('transport_trace_fea.csv', index=True)
print(train)
#train = pd.read_csv('train_fea.csv')

test_data = get_data(test_data, mode='test')






#train = get_feature(train_data, mode='train')
test = get_feature(test_data, mode='test')
features = [c for c in train.columns if c not in ['loadingOrder', 'label', 'mmin', 'mmax', 'count']]




result = build_model(train, test, features, 'label', is_shuffle=True)

test_data = test_data.merge(result, on='loadingOrder', how='left')
test_data['ETA'] = (test_data['onboardDate'] + test_data['label'].apply(lambda x: pd.Timedelta(seconds=x))).apply(
    lambda x: x.strftime('%Y/%m/%d  %H:%M:%S'))
test_data.drop(['direction', 'TRANSPORT_TRACE'], axis=1, inplace=True)
test_data['onboardDate'] = test_data['onboardDate'].apply(lambda x: x.strftime('%Y/%m/%d  %H:%M:%S'))
test_data['creatDate'] = pd.datetime.now().strftime('%Y/%m/%d  %H:%M:%S')
test_data['timestamp'] = test_data['temp_timestamp']
# 整理columns顺序
result = test_data[
    ['loadingOrder', 'timestamp', 'longitude', 'latitude', 'carrierName', 'vesselMMSI', 'onboardDate', 'ETA',
     'creatDate']]

result.to_csv('result.csv', index=False)


