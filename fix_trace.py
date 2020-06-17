import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import KFold
import lightgbm as lgb
import warnings
from scipy import spatial

warnings.filterwarnings('ignore')

# baseline只用到gps定位数据，即train_gps_path
train_gps_path = './data/train0523.csv'
test_data_path = 'self_test.csv'
order_data_path = './data/loadingOrderEvent.csv'
port_data_path = 'myports_new.csv'
split_data_path = './data/split'
file_list = os.listdir(split_data_path)

port_data = pd.read_csv(port_data_path)

# 取前1000000行
debug = True
NDATA = 1000000


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


# 读取处理后的数据，并提取特征
train = pd.read_csv('sub_features_new.csv')

#合并正确的标签
train_old = pd.read_csv('features.csv')
label_true = pd.DataFrame({
    'loadingOrder': train_old['loadingOrder'],
    'label': train_old['label']
})
train = train.drop('label', axis=1)
train = train.merge(label_true, on='loadingOrder', how='left')

# 合并路由之类的数据
transport = pd.read_csv('transport_trace_fea.csv', index_col=0)
transport = transport.drop('id', axis=1)
train = train.merge(transport, on='loadingOrder', how='left')

#合并起点坐标
zuobiao = pd.read_csv('first_zuobiao.csv', index_col=0)
zuobiao = zuobiao.drop('id', axis=1)
train = train.merge(zuobiao, on='loadingOrder', how='left')

#合并终点坐标
zuobiao = pd.read_csv('last_zuobiao.csv', index_col=0)
zuobiao = zuobiao.drop('id', axis=1)
train = train.merge(zuobiao, on='loadingOrder', how='left')

df = train.groupby('loadingOrder')[['first_longitude', 'first_latitude', 'last_longitude', 'last_latitude']].agg(lambda x: round(x[0:1])).reset_index()

loadingOrder = df['loadingOrder'].tolist()
first_longitude = df['first_longitude'].tolist()
first_latitude = df['first_latitude'].tolist()
first_xy = list(zip(first_longitude, first_latitude))

last_longitude = df['last_longitude'].tolist()
last_latitude = df['last_latitude'].tolist()
last_xy = list(zip(last_longitude, last_latitude))

l_first = dict(zip(loadingOrder, first_xy))
l_last = dict(zip(loadingOrder, last_xy))

trace_all = [x.replace(" ", "").upper() for x in port_data['router'].tolist()]
final_x = port_data['longitude'].tolist()
final_y = port_data['latitude'].tolist()
port_dict_x = dict(zip(trace_all, final_x))
port_dict_y = dict(zip(trace_all, final_y))

xy = [[final_x[i], final_y[i]] for i in range(len(final_x))]
kd = spatial.KDTree(data=xy)

drop_far = pd.DataFrame(columns=['loadingOrder', 'drop_far'])
threshold = 3
num = 0

train = pd.DataFrame(columns=['id', 'loadingOrder', 'TRANSPORT_TRACE_start', 'TRANSPORT_TRACE_final'])

for file in file_list:
    num += 1
    print(num)
    file_path = os.path.join(split_data_path, file)
    train_data_o = pd.read_csv(file_path, nrows=1, header=None)
    train_data_o.columns = ['id', 'loadingOrder', 'carrierName', 'timestamp', 'longitude',
                          'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                          'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']
    if train_data_o['loadingOrder'][0] not in l_first.keys(): continue
    if ('-' not in str(train_data_o['TRANSPORT_TRACE'][0])):
        #first
        x, y = l_first[train_data_o['loadingOrder'][0]]
        v, ii = kd.query([[x, y]])
        #print(v, ii)
        s1 = trace_all[ii[0]]
        # first
        x, y = l_last[train_data_o['loadingOrder'][0]]
        v, ii = kd.query([[x, y]])
        #print(v, ii)
        s2 = trace_all[ii[0]]
        train_o = pd.DataFrame({
            'id': train_data_o['id'][0],
            'loadingOrder': train_data_o['loadingOrder'][0],
            'TRANSPORT_TRACE_start': [s1.replace(" ", "").upper()],
            'TRANSPORT_TRACE_final': [s2.replace(" ", "").upper()]
        })
        #print(train_o)
        train = train.append(train_o, ignore_index=True)
        continue
    s = train_data_o['TRANSPORT_TRACE'][0].split('-')
    train_o = pd.DataFrame({
        'id': train_data_o['id'][0],
        'loadingOrder': train_data_o['loadingOrder'][0],
        'TRANSPORT_TRACE_start': [s[0].replace(" ", "").upper()],
        'TRANSPORT_TRACE_final': [s[-1].replace(" ", "").upper()]
    })
    train = train.append(train_o, ignore_index=True)
print(num, df.shape[0])
train.to_csv('transport_trace_fea_fix.csv', index=True)