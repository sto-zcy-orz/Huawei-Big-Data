# 一些分析

## 实验

- 1、直接baseline：14W
- 2、baseline把100w个数据改成200w个数据：13W
- 3、处理100种运单：4.3W
- 4、处理1000种运单：4.5W
  - 增加运单却mse增加？
    - 1、可能是迭代的次数不够，没训练到位
    - 2、有很多垃圾数据影响(可能性较大)
- 5、处理20种运单：3.6W
  - 迷惑？
    - 1、垃圾数据太多？
- 6、处理“下采样数据版本1”，-->16894条数据：5.2W
- 7、把6的mse改成越小越好，并迭代2000次：23W
- 8、把7改成迭代100次：22W



通过上述实验得到结论，这次比赛的要点就在于数据集和测试集的分布不同

- 答案的时间间隔大概在[18,20]之间集中
  - 第3个实验得到的时间间隔在[19.6,19.7]之间
  - 第5个实验得到时间间隔就是18.0
- 分布差距很大，如果不改变分布，会越训练效果越差
  - 第7个实验得到的时间间隔集中在[0,7]，mu=4.06，sigma=3.36
- 9、把训练集强行对齐测试集的高斯分布，迭代3000次，mse也是越小越好：4.6W
  - 线下就在训练集上的mse大概200多
- 10、把第9次的时间区间改小[5,40]，并扔掉数据量<=30的：3.3W：
- 11、把上面的时间前改小到[5,35]：2.8W
- 12、把船的识别码，路由的首尾端加进来，在第11的基础上，训练['vesselMMSI', 'TRANSPORT_TRACE_start', 'TRANSPORT_TRACE_final', 'anchor_ratio', 'anchor_cnt', 'direction_min', 'direction_max', 'direction_median']这些特征：3.3W
- 13、在第12次的基础上，只训练['vesselMMSI', 'TRANSPORT_TRACE_start', 'TRANSPORT_TRACE_final']的特征：1.6W
- 14、yyh：直接按照路由归类，然后取平均值，测试集路由不清的取22：2.6W
- 15、时间限制在[5,30]，mse:1.8W
- 16、yyh：在14的基础上，把时间限制在[5,35]，mse:3w多?
  - 发现可能在[35，50]还有一堆
- 17、在15的基础上，把时间限制在[5,50]，mse:1.4w
- 18、把数据改成10+15+20+25+30，去掉时间限制，mse:2w
  - 后面觉得这里就是很可能就是高斯分布拟合没扔掉的问题
- 19、去掉高斯分布拟合，增加起始点的坐标和终止点的坐标，时间[1.5,∞),使用的特征扔掉了'vesselMMSI', 'TRANSPORT_TRACE_final'，mse: 5635.3760 
- 20、把时间设置在[5,∞)，mse:4689.6243 
- 21、之前的drop的方法扔多了，改进之后，mse：4228.0473 
- 22、在21的基础上加上了4sigma判断异常值的方法，mse:4978.2605 
  - 验证集中3sigma不能提升精度，4sigma可以
- 23、把判断异常值的去掉，时间限制在[5,50]，mse：4070.5402 
- 24、发现提交的那个程序其实没有把起点的坐标加进去，改了一下，mse：2424.7980 
- 25、不记得了，mse：2630.5850 
- 26、改成了5折交叉验证，mse：2298.5309 
- 27、加上了起点和终点的geohash，mse：3207.5456 
- 28、不加geohash，把数据条数[count]限制在20000以内，mse：2240.4072 
- 29、把所有特征给用上（此时mu偏移到了24），mse：6976.5583 
- 30、在28的基础上，把trace中有nan的给扔掉，mse：8530.9747 
- 31、在30的基础上，把anchor_cnt限制在1w以内(感觉这样可能可以解决一点塞港)，mse：2235.6425 
- 32、在31的基础上把nan给drop了，mse：2472.9858 
- 33~34、在31的基础上又对count和anchor_cnt限制了一下，试过一个1w、一个5000或只有1750，mse：2332.5266 ，2243.1052 
- 35、在31次的基础上，用最近港口修复了一下路由数据，mse：2248.3722 
- 36、在35的基础上，把起始或终止路由离起始点远的(大于某个阈值，好像取得是5)给扔掉，mse：2601.4556 
- 37、在36的基础上，把与测试集无关的航线扔掉，mse：5037.0797 
- 38、在37的基础上，保留与测试集无关，但是数据集中重复航线数>10且起始或终止路由离起始点b不远的，然后重新选了一下特征['loadingOrder', 'label', 'mmin', 'mmax', 'count', 'vesselMMSI', 'TRANSPORT_TRACE_final','carrierName', 'last_geohash', 'first_geohash', 'both_geohash',                                                   'anchor_cnt', 'latitude_median', 'latitude_min', 'latitude_max', 'latitude_mean']，mse：3053.1603 

## 之后的处理的打算

### 2020.6.7

- 可以参照海洋那个比赛，对经纬度处理
- 通过聚类的方式找到测试数据的相似类来训练）【回顾：直接用路由多分类】
- 增加特征
  - 像洋流，天气之类的，不过预计影响不大
  - 识别码，路由之类的【回顾：还行】
  - 物理学特征？

## 2020.6.8

- 线下搞个验证集【回顾：稍微能操作判断一下结果怎么样，不过线上线下因为分布不同，结果还是有点差距】
- 新建一个多分类任务预测航线目标港口【回顾：还行】
- 直接把训练集按照测试集的大小划分，让是分布相同【回顾：如果是直接高斯拟合不太行，但是数据集就那样划分还行】

## 2020.6.9

- 线下数据集内的mse通常在30以内，然后验证集都是3W以上
  - 把数据按照10%,20%。。。来采样训练【回顾：可能是关键】

## 2020.6.10

- 把起点和终点的坐标加进特征【回顾：也许还行】

## 2020.6.11

- 新建距离特征【回顾：还行】
- 把之前搞错的drop处理好【回顾：还行】
- 对着智慧海洋的top 1的方法硬剽



## 2020.6.15

- geohash的这样搞得数据集干净之后，深度学习也得数据集之后

# 特征的处理

- 测试集的port的终点坐标要基于路由算，路由根据port算，如果不对应人工加
- 可以根据测试集的数据分布扔掉一些相对异常值

(105485, 32)

- 第21次的时候，10w条数据，32个特征

测试集：222个测试运单号

https://blog.csdn.net/qq_26593695/article/details/106557941这个网址里分析的较详细

  

  # 数据清洗的想法

1、限制count小于多少：  大概可以搞掉一些塞港的问题，而且测试集最大的数量不超过1750(当然这是10%-50%)

2、限制时间在某个区间：数据中有几个时间是为0的，还有些时间很长

3、对于路由来说，判断和起点、终点的距离，如果有一个大于某个阈值就扔掉(发现经纬度坐标系>10的有三千多条)

4、搜一下港口对(不同的航线)的数量扔进一个桶，如果数量<一个阈值就扔掉(有很多只有1)

5、把不在测试数据中的港口对给扔掉

  # A榜答案分布

- 这个是mse：2240.4072 的

![mse:2240.4072 ](./mse:2240.4072.png)