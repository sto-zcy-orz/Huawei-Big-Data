import pandas as pd
from tqdm import tqdm
import numpy as np
import os

from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')

# baseline只用到gps定位数据，即train_gps_path
os.chdir("E:\华为云")
train_gps_path = 'data/train0523.csv'
test_data_path = 'data/A_testData0531.csv'
order_data_path = 'data/loadingOrderEvent.csv'
port_data_path = 'data/port.csv'
split_data_path='data/split'

# debug = True
# NDATA = 1000
# train_data = pd.read_csv(train_gps_path,nrows=NDATA,header=None)
# print(train_data.head())


reader = pd.read_csv(train_gps_path, iterator=True)

loop = True
chunkSize = 1000000
n=0
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        n+=chunkSize
        chunk.columns = ['loadingOrder','carrierName','timestamp','longitude',
                  'latitude','vesselMMSI','speed','direction','vesselNextport',
                  'vesselNextportETA','vesselStatus','vesselDatasource','TRANSPORT_TRACE']
        a=chunk.groupby('loadingOrder')
        for k,_ in a:
            i=a.get_group(k)
            i.to_csv(os.path.join(split_data_path,k+".csv"), mode='a', header=False)
        # loop=False
        if(n%10000000==0):
            print(n)
    except StopIteration:
        loop = False
        print("Iteration is stopped.")
print(n)