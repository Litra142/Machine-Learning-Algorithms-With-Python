import numpy as np
import pandas as pd
from .metrics import accuracy_score
#创建knn的类
class Knn_Classification:
    """使用python实现k近邻算法（实现分类），考虑权重，使用距离的倒数作为权重"""

    def __init__(self,k):
        """初始化方法
        Parameter
        ------
        k:int
            邻居的个数
        """
        self.k = k

    def fit(self,X,y):
        """训练方法
        Parameters
        ------
        X:类数组类型,形状：[样本数量，特征数量]
          待训练的样本特征（属性）

        y:数组类型,形状：[样本数量]
          每个样本的目标值（标签）
        """
        #将x，y转换为ndarray数组类型
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self,X):
        """根据参数传递样本，进而对样本进行预测
        Parameters
        ------
        X:类数组类型,形状：[样本数量，特征数量]
          待预测的样本特征（属性）
        Returns
        ------
        result:数组类型
            预测的结果。
        """
        X = np.asarray(X)
        result = []
        for x in X:
            #对于测试集一行样本，依次与训练集中所有样本求距离，x对应到坐标轴上就是一个点
            #由于x是一行的数据，而self.X是多行的形式（在这个样本中是120*4行的数据）
            #所以在进行不同行的运算时，numpy会启发广播机制
            #计算传进来的点与训练数据的距离
            dis = np.sqrt(np.sum((x - self.X)**2,axis = 1))
            #之后根据k近邻算法的规则，求最近的k个点，所以要进行排序操作，找到这k个点对应的标签
            #argsort()返回排序的结果，并且依次返回每个元素排序前在原数组中的位置（索引）
            index = dis.argsort()
            #取前k个，进行截断，用切片
            index = index[:self.k]
            #计算所有邻居的
            #找到在距离最近的k个节点对应的标签，再进行判断每一个类别出现的次数
            #统计每一个标签出现的次数，bincount()可以返回数组中每个元素出现的次数，但是有个要求，
            #必须是非负的整数# 返回数组中每个整数元素出现次数，元素必须是非负整数\n",
            #考虑权重，只需要在bincount 中传入weights = 1/dis[index]
            count= np.bincount(self.y[index], weights = 1/ dis[index])
            #找到count中值最大的索引，因为这个索引就代表几出现的次数
            #返回ndarray数组中值最大元素对应的索引,该索引就是我们判定的类别
            result.append(count.argmax())
        return( np.asarray(result))

    def score(self,y_true,y_predict):
        return accuracy_score(y_true,y_predict)

    def __repr__(self):
        return "Knn_Classification()"

class Knn_Regression:
    """使用python实现k近邻算法（实现回归），考虑权重，默认使用距离的倒数作为权重
    该算法用于回归预测，根据前三个特征属性，寻找最近的k个邻居，然后再根据k个邻居的第四个特征属性，
    去预测当前样本的第4个特征值
    """

    def __init__(self,k):
        """初始化方法
        Parameter
        ------
        k:int
            邻居的个数
        """
        self.k = k
    def fit(self,X,y):
        """训练方法
        Parameters
        ------
        X:类数组类型,形状：[样本数量，特征数量]
          待训练的样本特征（属性）

        y:数组类型,形状：[样本数量]
          每个样本的目标值（标签）
        """
        #将x，y转换为ndarray数组类型
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self,X):
        """根据参数传递样本，进而对样本进行预测
        Parameters
        ------
        X:类数组类型,形状：[样本数量，特征数量]
          待预测的样本特征（属性）
        Returns
        ------
        result:数组类型
            预测的结果。
        """
        X = np.asarray(X)
        result = []
        for x in X:
            dis = np.sqrt(np.sum((x - self.X)**2,axis = 1))
            index = dis.argsort()
            #取前k个，进行截断，用切片
            index = index[:self.k]
            #考虑截距
            s = np.sum(1/(dis[index] + 0.01))   # 距离倒数之和，最后加一个一很小的数就是为了避免距离为0的情况
            weight = (1/(dis[index]+0.001))/s     # 距离倒数/k个节点的距离倒数之和，（这样就可以得到每条边的权重）
            #计算最近的几个点他们对应的第四个值得平均数即使可得预测值 #邻居节点标签纸*对应权重相加求和
            result.append(np.sum(self.y[index] *weight))
        return( np.asarray(result))

    def score(self, X_test, y_test):
        """根据测试数据集，X_test,和y_test确定当前模型的精确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "Knn_Regression()"
