import numpy as np
import pandas as pd
class Imputer:
    """利用python实现处理数据的缺失值"""
    def fit(self,X):
        """通过传进的参数对模型进行训练
        X:类数组形式，形状：[样本数量，特征数量]
            待训练的特征（属性）
        """
        #计算缺失数据的数量
        na_num = X.isnull().sum(axis= 0)
        #计算缺失值的占比
        self.m = na_num / len(X)
        #分别计算数据的平均值和中位数
        self.X_mean = np.mean(X,axis = 0)
        self.X_median = np.median(X,axis = 0)

    def transform(self,X,strategy = 'mean',axis = 0):
        """通过传进来的参数对数据进行处理
        Parameters
        -----
        X:类数组形式，形状：[样本数量，特征数量]
            待训练的特征（属性）
        strategy:object
            替换值得类型
        axis :int
            处理的维度
        #在sklearn的实现是这样的：missing_values='NaN', strategy='mean', axis=0
        """
        #根据缺失数据占比来判断用什么值来替代
        if self.m >= 0.4:
        #缺失数据占比达到40%时，使用0来替换缺失数据
            X = X.fillna(0)
        else:
            if strategy == 'mean':
                X = X.fillna(self.X_mean)
            if strategy == 'median':
                X = X.fillna(self.X_median)
        return X
    def __repr__(self):
        return "Imputer()"

class StandarScaler:
    """该类对数据进行标准化处理"""

    def fit(self, X):
        """根据每个传递的样本，计算每个特征列的均值与标准差
        Parameters
        -----
        X:类数组类型
            训练数据，用来计算均值和标准差
        """
        X = np.asarray(X)
        self.std_ = np.std(X,axis = 0)
        self.mean_ = np.mean(X,axis = 0)
        return self

    def transform (self, X):
        """
        对给定的数据X，进行标准化处理（将X的每一列都变成标准正态分布的数据）
        parameter
        -----
        X: 类数组形式
            待转化的数据
        Returns
        -----
        result:类数组类型
            参数转换成标准正太分布后的结果
        """
        result = (X - self.mean_) / self.std_
        return result

    def __repr__(self):
        return "StandarScaler()"

class MinMaxScaler:
    """该类对数据进行归一化处理"""

    def fit(self, X):
        """根据每个传递的样本，计算每个特征列的均值与标准差
        Parameters
        -----
        X:类数组类型
            训练数据，用来计算均值和标准差
        """
        X = np.asarray(X)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        """
        对给定的数据X，进行标准化处理（将X的每一列都变成标准正态分布的数据）
        parameter
        -----
        X: 类数组形式
            待转化的数据
        Returns
        -----
        result:类数组类型
            参数转换成标准正太分布后的结果
        """
        return (X - self.min_) / (self.max_ - self.max_)

    def __repr__(self):
        return "MinMaxScaler()"


#用独热编码实现离散数据连续化
def OneHotCode(self,X):
    """用独热编码将离散数据连续化
    Parameters
    -----
    X:类数组形式
        待转化的数据
    Returns
    -----
    返回转化后的数据
    """
    X = pd.get_dummies(X)
    return X

class EquiWeigth:
    """用python实现等宽连续数值离散化"""
    def fit(self,data,k):
        """将传进来的数据进行等宽发离散法
        Parameters
        -----
        data:数组形式，形状：[样本数量]
            待转换的连续数值
        k:int
            离散后的数据段
        """
        self.k = k
        self.data = data

    def transform(self):
        d =  pd.cut(self.data,self.k,labels = range(self.k))
        return d

    def __repr__(self):
        return "EquiWeigth()"


class EquiFrequent:
    def fit(self, data, k):
        """将传进来的数据进行等宽发离散法
        Parameters
        -----
        data:数组形式，形状：[样本数量]
            待转换的连续数值
        k:int
            离散后的数据段
        """
        self.k = k
        self.data = data

    def transform(self):
        w = self.data.quanties([1.0 * i / self.k for i in range(self.k + 1)])
        w[0] = w[0] * (1 - 1e-10)
        d = pd.cut(self.data, w ,labels=range(self.k))
        return d

    def __repr__(self):
        return "Equifrequent()"

#几种距离计算的实现
#数据相似度分析
def euclidean(p,q):
    """计算欧几里德距离"""
    #两组数据集数目不一定相同，所以先计算两者间都对应的有的数
    same = 0
    for i in p:
        for j in q:
            same +=1
    #计算欧式距离,并将其标准化
    e = sum([(p[i] - q[i])**2 for i in range(same)])
    dis = 1/(1+e**0.5)
    return dis

def pearson(p,q):
    """计算皮尔逊相关度"""
    same = 0
    for i in p:
        for j in q:
            same +=1
    n = same
    # 分别求出p,q的和
    p_sum = sum([p[1] for i in range(n)])
    q_sum = sum([q[1] for i in range(n)])
    # 分别求出pq,的平方和
    p_square_sum = sum([p[1] ** 2 for i in range(n)])
    q_square_sum = sum([q[i] ** 2 for i in range(n)])
    # 根据公式，求出p q的乘积和
    pq_sum = sum([p[i] * q[i] for i in range(n)])
    # 求出pearnson的相关系数
    up = pq_sum - p_sum * q_sum / n
    dowm = ((p_square_sum - pow(p_sum, 2) / n) * (q_square_sum - pow(q_sum, 2) / n)) ** 0.5
    if dowm == 0:
        return 0
    result = up / dowm
    return result

#余弦相似度分析
def consine(p,q):
    """利用余弦相似度分析两组数据的相似度"""
    same = 0
    for i in p:
        for j in q:
            same += 1
    n = same
    #先在之前的基础上稍加修改即可
    # 分别求出pq,的平方和
    p_square_sum = sum([p[1] ** 2 for i in range(n)])
    q_square_sum = sum([q[i] ** 2 for i in range(n)])
    # 根据公式，求出p q的乘积和
    pq_sum = sum([p[i] * q[i] for i in range(n)])
    # 求出pearnson的相关系数
    up =pq_sum
    dowm = (p_square_sum**0.5)*(q_square_sum**0.5)
    if dowm == 0:
        return 0
    result = up / dowm
    return result