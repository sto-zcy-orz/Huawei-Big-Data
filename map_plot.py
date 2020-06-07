import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import warnings
import random
warnings.filterwarnings('ignore')

# baseline只用到gps定位数据，即train_gps_path
os.chdir("E:\华为云")
train_gps_path = 'data/train0523.csv'
test_data_path = 'data/A_testData0531.csv'
order_data_path = 'data/loadingOrderEvent.csv'
port_data_path = 'data/port.csv'
split_data_path='data/split'

fig=plt.figure()

#鼠标事件回调函数
def call_back(event):
    axtemp = event.inaxes
    base_scale=2
    x_min, x_max = axtemp.get_xlim()
    y_min, y_max = axtemp.get_ylim()
    fanwei_x = (x_max - x_min)/2
    fanwei_y = (y_max - y_min)/2
    xdata = event.xdata
    ydata = event.ydata
    if event.button == 'up':
        scale_factor = 1 / base_scale
    elif event.button == 'down':
        scale_factor = base_scale
    else:
        scale_factor = 1

    axtemp.set(xlim=(xdata - fanwei_x*scale_factor, xdata + fanwei_x *scale_factor))
    axtemp.set(ylim=(ydata - fanwei_y*scale_factor , ydata +fanwei_y*scale_factor))
    fig.canvas.draw_idle()


reader = pd.read_csv(train_gps_path, iterator=True)

file_list = os.listdir(split_data_path)
flag = 0

def time_dis(x,y):
    return (data["timestamp"][y]-data["timestamp"][x]).total_seconds()

def dis_lon(x,y):
    k=abs(x-y)
    return min(360-k,k)**2

def dis_lat(x,y):
    k=abs(x-y)
    return min(180-k,k)**2

# for file in file_list:
while True:
    file=random.choice(file_list)
    data = pd.read_csv(os.path.join(split_data_path,file),header=None)
    data.columns = ['id','loadingOrder','carrierName','timestamp','longitude',
                  'latitude','vesselMMSI','speed','direction','vesselNextport',
                  'vesselNextportETA','vesselStatus','vesselDatasource','TRANSPORT_TRACE']
    data["timestamp"]=pd.to_datetime(data['timestamp'], infer_datetime_format=True)
    n=len(data["longitude"])
    print(file)
    s=0
    for i in range(1,n):
        s=max(s,dis_lon(data["longitude"][i],data["longitude"][i-1])+dis_lat(data["latitude"][i],data["latitude"][i-1]))
    if(s<30):
        continue #这行用来筛不对劲的数据
    fig,_=plt.subplots()
    m = Basemap()     # 实例化一个map
    m.drawcoastlines()  # 画海岸线
    m.drawmapboundary(fill_color='white')  
    m.fillcontinents(color='white',lake_color='white') # 画大洲，颜色填充为白色
    
    parallels = np.arange(-90., 90., 10.)  # 这两行画纬度，范围为[-90,90]间隔为10
    m.drawparallels(parallels,labels=[False, True, True, False])
    meridians = np.arange(-180., 180., 20.)  # 这两行画经度，范围为[-180,180]间隔为10
    m.drawmeridians(meridians,labels=[True, False, False, True])
    fig.canvas.mpl_connect('scroll_event', call_back)
    fig.canvas.mpl_connect('button_press_event', call_back)
    lon, lat = m(data["longitude"], data["latitude"])  # lon, lat为给定的经纬度，可以使单个的，也可以是列表

    n=len(lon)
    flag=1
    last=0
    c=''

    for i in range(1,n):
        if(time_dis(last,i)<86400 and i!=n-1):
            continue
        if(flag):
            c='g'
            flag=0
        else:
            c='r' if c!='r' else 'b'
        m.plot(lon[last:i+1],lat[last:i+1],c+".--")
        last=i
    print(s)
    plt.show()
