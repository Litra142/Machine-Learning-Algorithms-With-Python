#导入相关的库
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import mode
#先设置成支持中文界面
mpl.rcParams['font.family'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False
#其实可以直接看数据相关的描述来看看数据是否有异常
#编写相关的函数
def my_describe(data):
    #每一行数据的最大值和最小值
    data_max = np.max(data,axis =0)
    print("数据的最大值是：")
    print(data_max)
    data_min = np.min(data,axis = 0)
    print("数据的最小值是：")
    print(data_min)
    #数据的极差
    data_ptp = np.ptp(data,axis = 0)
    print("数据的极差是：")
    print(data_ptp)
    #每一行数据的平均值
    data_mean = np.mean(data,axis = 0)
    print("数据的平均值是：")
    print(data_mean)
    #每一行数据的中位数
    data_median = np.median(data,axis = 0)
    print("数据的中位数是：")
    print(data_median)
    #计算每一列数据的众数
    data_mode = mode(data,axis=0)
    print("数据的众数是：")
    print(data_mode)
    #每一行数据的标准差
    data_std = np.std(data,axis = 0)
    print("数据的标准差是：")
    print(data_std)
    #每一行数据的方差
    data_var = np.var(data,axis= 0)
    print("数据的方差是：")
    print(data_var)





