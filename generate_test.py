import pandas as pd
from tqdm import tqdm
import numpy as np
import os

from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.model_selection import KFold
import lightgbm as lgb

import warnings
import time
import random
warnings.filterwarnings('ignore')

# baseline只用到gps定位数据，即train_gps_path
os.chdir("E:\华为云")
train_gps_path = 'data/train0523.csv'
test_data_path = 'data/A_testData0531.csv'
self_test_data_path = 'data/self_test.csv'
order_data_path = 'data/loadingOrderEvent.csv'
port_data_path = 'data/port.csv'
split_data_path='data/split'
feature_path='src/data/features.csv'
header_flag=True

def get_data(file):
    global header_flag
    train_data = pd.read_csv(os.path.join(split_data_path,file),header=None)
    train_data.columns = ['id','loadingOrder','carrierName','timestamp','longitude',
                    'latitude','vesselMMSI','speed','direction','vesselNextport',
                    'vesselNextportETA','vesselStatus','vesselDatasource','TRANSPORT_TRACE']
    n=len(train_data["id"])
    k=int(random.uniform(10,30)*n//100)
    train_data['onboardDate']=''
    print(train_data['timestamp'][n-1])
    train_data['realETA']=train_data['timestamp'][n-1]
    tags=['loadingOrder','timestamp','longitude','latitude','speed','direction','carrierName','vesselMMSI','onboardDate','TRANSPORT_TRACE','realETA']
    test_data=train_data[tags][0:k]
    test_data.to_csv(self_test_data_path,index=False,mode='a',header=header_flag)
    if(header_flag):
        header_flag=False


file_list = os.listdir(split_data_path)
cnt=0
if os.path.exists(self_test_data_path):
    os.remove(self_test_data_path)
while True:
    file=random.choice(file_list)
    print(file)
    get_data(file)
    cnt+=1
    if(cnt==100):
        break