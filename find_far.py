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
transport = pd.read_csv('transport_trace_fea_fix.csv', index_col=0)
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

df = train.groupby('loadingOrder')[['TRANSPORT_TRACE_start', 'TRANSPORT_TRACE_final', 'first_longitude', 'first_latitude', 'last_longitude', 'last_latitude']].agg(lambda x: round(x[0:1])).reset_index()
df1 = train.groupby('loadingOrder')[['TRANSPORT_TRACE_start', 'TRANSPORT_TRACE_final']].agg(
        lambda x: x.value_counts().index[0]).reset_index()

print(df)
print(df1)
trace_all = [x.replace(" ", "").upper() for x in port_data['router'].tolist()]
final_x = port_data['longitude'].tolist()
final_y = port_data['latitude'].tolist()
port_dict_x = dict(zip(trace_all, final_x))
port_dict_y = dict(zip(trace_all, final_y))

drop_far = pd.DataFrame(columns=['loadingOrder', 'drop_far'])
threshold = 100
num = 0
for i in range(df.shape[0]):
    print(i)
    first_x, first_y = port_dict_x[df1['TRANSPORT_TRACE_start'][i]], port_dict_y[df1['TRANSPORT_TRACE_start'][i]]
    last_x, last_y = port_dict_x[df1['TRANSPORT_TRACE_final'][i]], port_dict_y[df1['TRANSPORT_TRACE_final'][i]]
    if ((first_x - df['first_longitude'][i])**2 + (first_y - df['first_latitude'][i])**2 > threshold) or ((last_x - df['last_longitude'][i])**2 + (last_y - df['last_latitude'][i])**2 > threshold):
        d = ({
            'loadingOrder': df['loadingOrder'][i],
            'drop_far': 'Dropped'
        })
        drop_far = drop_far.append(d, ignore_index=True)
        num += 1
    else:
        d = ({
            'loadingOrder': df['loadingOrder'][i],
            'drop_far': 'Nice'
        })
        drop_far = drop_far.append(d, ignore_index=True)

print(num, df.shape[0])
drop_far.to_csv('drop_far.csv', index=True)