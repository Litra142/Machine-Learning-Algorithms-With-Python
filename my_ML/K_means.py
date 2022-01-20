import numpy as np
import pandas as pd
from tqdm import tqdm

class KMeans:
    """用python实现Kmeans算法"""
    def __init__(self,k,times):
        """初始化
        Parameters
        -----
        k: int
            聚类的个数
        times : int
            迭代的次数
        """
        self.k = k
        self.times = times

    def fit(self,X):
        """对参数传递进来的样本进行训练，得到相应模型
        Parameters
        -----
        X : 类数组类型，形状：[样本的数量，特征的数量]
            待训练的样本特征属性。
        """
        #将 X 转换为数组的类型
        X = np.asarray(X)
        #设置随机种子
        #设置随机种子的依据：只需要保证每次的随机种子相同，即可重现相同的随机结果呢
        np.random.seed(0)
        #随机产生k 个聚类中心
        self.cluster_centers_ = X[np.random.randint(0,len(X),self.k)]
        #创建为labels_，用来存放所属的簇，初始化为一个全为0的数组，之后进行迭代时再将迭代结果存放到labels_中
        self.labels_ = np.zeros(len(X))
        #进行迭代
        #用tqdm模块实现进度条显示
        for t in tqdm(range(self.times)):
            for index,x in enumerate(X):
                #计算每个点与就聚类中心的距离
                dis= np.sqrt(np.sum((x - self.cluster_centers_) **2 ,axis = 1))
                #将最小距离的索引复制给标签数组，索引的值就是当前点所属的簇，方位是[0,k -1]
                self.labels_[index] = dis.argmin()
            #循环遍历每一个数更新聚类中心
            for i  in range(self.k):
                #计算每个簇内所有点的均值，将其作为新一轮计算距离的聚类中心
                self.cluster_centers_[i] = np.mean(X[self.labels_ == i],axis =0)

    def predict(self,X):
        """根据参数传递地样本，对样本数据今昔那个预测（预测样本属于哪一个簇中）
        Parameters
        -----
        X : 类数组类型，形状： [样本数据，特征数量]
        Returns
        -----
        result: 数组类型
            预测的样本的特征属性
        """
        X = np.asarray(X)
        result = np.zeros(len(X))
        for index, x in enumerate(X):
        # 计算每个点与就聚类中心的距离
            dis = np.sqrt(np.sum((x - self.cluster_centers_) ** 2, axis=1))
        # 将最小距离的索引复制给标签数组，索引的值就是当前点所属的簇，方位是[0,k -1]
            result[index] = dis.argmin()
        return result

