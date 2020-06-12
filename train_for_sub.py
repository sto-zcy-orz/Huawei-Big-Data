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
port_data = pd.read_csv(port_data_path)

# 取前1000000行
debug = True
NDATA = 1000000

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

    #处理路由和识别码等数据

    #df['TRANSPORT_TRACE_start'] = df.apply(lambda x: x['TRANSPORT_TRACE'].split('-')[0])
    #df['TRANSPORT_TRACE_start'] = df.apply(lambda x: x['TRANSPORT_TRACE'].split('-')[-1])
    mmsi_df = df.groupby('loadingOrder')[['vesselMMSI', 'TRANSPORT_TRACE']].agg(lambda x: x.value_counts().index[0]).reset_index()
    group_df = group_df.merge(mmsi_df, on='loadingOrder', how='left')
    mmsi_df = pd.DataFrame({
        'loadingOrder': group_df['loadingOrder'],
        'TRANSPORT_TRACE_start': [x.split('-')[0] for x in group_df['TRANSPORT_TRACE']],
        'TRANSPORT_TRACE_final': [x.split('-')[-1] for x in group_df['TRANSPORT_TRACE']]
    })
    group_df = group_df.merge(mmsi_df, on='loadingOrder', how='left')
    group_df = group_df.drop('TRANSPORT_TRACE', axis=1)

    # 处理终点数据
    trace_final = [x.replace(" ", "").upper() for x in port_data['TRANS_NODE_NAME'].tolist()]

    final_x = port_data['LONGITUDE'].tolist()
    final_y = port_data['LATITUDE'].tolist()
    port_dict_x = dict(zip(trace_final, final_x))
    port_dict_y = dict(zip(trace_final, final_y))
    """zuobiao_df = group_df.groupby('loadingOrder')[['TRANSPORT_TRACE_final']].agg(
        lambda x: x.value_counts().index[0]).reset_index()"""
    zuobiao_df = pd.DataFrame({
        'loadingOrder': group_df['loadingOrder'],
        # 'TRANSPORT_TRACE_final': [x.replace(" ", "").upper() for x in group_df['TRANSPORT_TRACE_final']],
        'last_longitude': [round(port_dict_x[x]) for x in group_df['TRANSPORT_TRACE_final']],
        'last_latitude': [round(port_dict_y[x]) for x in group_df['TRANSPORT_TRACE_final']]
    })
    group_df = group_df.merge(zuobiao_df, on='loadingOrder', how='left')

    anchor_df = df.groupby('loadingOrder')['anchor'].agg('sum').reset_index()
    anchor_df.columns = ['loadingOrder', 'anchor_cnt']
    group_df = group_df.merge(anchor_df, on='loadingOrder', how='left')
    group_df['anchor_ratio'] = group_df['anchor_cnt'] / group_df['count']

    agg_function = ['min', 'max', 'mean', 'median']
    agg_col = ['latitude', 'longitude', 'speed', 'direction']

    group = df.groupby('loadingOrder')[agg_col].agg(agg_function).reset_index()
    group.columns = ['loadingOrder'] + ['{}_{}'.format(i, j) for i in agg_col for j in agg_function]
    group_df = group_df.merge(group, on='loadingOrder', how='left')

    group_df['vesselMMSI'] = group_df['vesselMMSI'].astype('category')
    group_df['TRANSPORT_TRACE_start'] = group_df['TRANSPORT_TRACE_start'].astype('category')
    group_df['TRANSPORT_TRACE_final'] = group_df['TRANSPORT_TRACE_final'].astype('category')

    return group_df

test_data = pd.read_csv(test_data_path)

def drop_for_status(data):
    data = data[data['status'] != 'Dropped']
    data = data.drop('status', axis=1)
    data = data.drop('rate', axis=1)
    data = data[data['label'] > 5 * 60 * 24 * 60]
    data = data[data['label'] < 50.0 * 60 * 24 * 60]
    #data = data[data['count'] > 30]
    #data = data.dropna(axis=0, how="any")
    #data = data[str(data['TRANSPORT_TRACE_start']) != 'nan']
    return data

#丢掉一些相对于测试集的异常值，置信区间用3sigma
def drop_Outliers(data, test):
    features = data.keys()
    for i in features:
        if (isinstance(data[i][0], str) == True): continue
        #if (i == 'first_longitude') or (i == 'first_latitude') or (i == 'last_longitude') or (i == 'last_latitude'): continue
        if (i == 'label'): continue
        mu, sigma = np.mean(test[i]), np.std(test[i])
        mu1, sigma1 = np.mean(data[i]), np.std(data[i])
        data = data[data[i] > min(mu1 - 4 * sigma1, mu - 4 * sigma)]
        data = data[data[i] < max(mu1 + 4 * sigma1, mu + 4 * sigma)]
    return data

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

#合并终点坐标
zuobiao = pd.read_csv('last_zuobiao.csv', index_col=0)
zuobiao = zuobiao.drop('id',axis=1)
train = train.merge(zuobiao, on='loadingOrder', how='left')

print(train)
print(train.shape)
#train.to_csv('train_fea.csv', index=True)
train = drop_for_status(train)

test_data = get_data(test_data, mode='test')

#train = get_feature(train_data, mode='train')
test = get_feature(test_data.copy(), mode='test')
test.to_csv('test_fea.csv', index=True)

#train = drop_Outliers(train, test)

print(train.shape)
features = [c for c in train.columns if c not in ['loadingOrder', 'label', 'mmin', 'mmax', 'count', 'vesselMMSI', 'TRANSPORT_TRACE_final']]
#features = ['vesselMMSI', 'TRANSPORT_TRACE_start', 'TRANSPORT_TRACE_final']
print(features)

# 改变训练集的正态分布
def change_distribute(data, target, features):
    for i in features:
        if (isinstance(data[i][0], str) == True): continue
        if (i == 'label'): continue
        mu, sigma = np.mean(data[i]), np.std(data[i])
        #print(mu, sigma)
        mu1, sigma1 = np.mean(target[i]), np.std(target[i])
        #print(mu1, sigma1)
        data[i] = (data[i] - mu)/sigma * sigma1 + mu1
        mu2, sigma2 = np.mean(data[i]), np.std(data[i])
        #print(mu2, sigma2)
    return data

#train = change_distribute(train, test, features)

def mse_score_eval(preds, valid):
    labels = valid.get_label()
    scores = mean_squared_error(y_true=labels/60/60, y_pred=preds/60/60)/len(preds)
    return 'mse_score', scores, False


def build_model(train, test, pred, label, seed=1080, is_shuffle=True):
    train_pred = np.zeros((train.shape[0],))
    test_pred = np.zeros((test.shape[0],))
    n_splits = 10
    # Kfold
    fold = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=seed)
    kf_way = fold.split(train[pred])
    # params
    params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'num_leaves': 31,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.7,
        'bagging_freq': 6,
        'seed': 8,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'nthread': 8,
        'verbose': 1,
    }
    # train
    for n_fold, (train_idx, valid_idx) in enumerate(kf_way, start=1):
        print('n_fold: %d'%(n_fold))
        train_x, train_y = train[pred].iloc[train_idx], train[label].iloc[train_idx]
        valid_x, valid_y = train[pred].iloc[valid_idx], train[label].iloc[valid_idx]
        # 数据加载
        n_train = lgb.Dataset(train_x, label=train_y)
        n_valid = lgb.Dataset(valid_x, label=valid_y)

        clf = lgb.train(
            params=params,
            train_set=n_train,
            num_boost_round=1000,
            valid_sets=[n_valid],
            early_stopping_rounds=100,
            verbose_eval=100,
            feval=mse_score_eval
        )
        train_pred[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        test_pred += clf.predict(test[pred], num_iteration=clf.best_iteration) / fold.n_splits

    test['label'] = test_pred

    return test[['loadingOrder', 'label']]

train['vesselMMSI'] = train['vesselMMSI'].astype('category')
train['TRANSPORT_TRACE_start'] = train['TRANSPORT_TRACE_start'].astype('category')
train['TRANSPORT_TRACE_final'] = train['TRANSPORT_TRACE_final'].astype('category')
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


