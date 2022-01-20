#利用逻辑回归进行二分类算法
import numpy as np
import pandas as pd
from .metrics import r2_score
#创建类
class LogisticRegression:
    """利用python实现逻辑回归算法，二分类问题"""

    def __init__(self,alpha,times):
        """初始化算法
        Parameters
        -----
        alpha:float
            学习率，决定梯度下降的步幅
        times:int
            迭代次数
        """
        self.alpha= alpha
        self.times = times

    def sigmod(self,z):
        """sigmod函数的实现
        Parameters
        -----
        z :float
            自变量，值为:z = w.T*x
        Returns
        -----
        p:float,值为[0,1]之间
            返回类别为1 的概率值，用来作为结果的预测值
            当z>=0.5(z>=0)时，返回类别1，否则，返回类别2
        """
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self,X,y):
        """根据传进来的训练数据，对模型进行训练。
        Parameter
        -----
        X: 类数组形式，形状：[样本数量，特征数量]
            待训练的样本特征属性
        y:类数组形式，形状：[特征数量]
            每个样本的目标值
        """
        X = np.asarray(X)
        y = np.asarray(y)
        #在梯度下降的过程中，权重的初始值并不重要，我们在这里将他初始化为0即可，但需要注意都是，权重的数量需要比特征的数量多1
        #创建权重的向量，初始化为0，长度比特征数量多1，多出来的一个是截距
        self.w_ = np.zeros(1 + X.shape[1])
        #创建损失列表，用来存放梯度下降过程中每次迭代后的损失值
        #我们可以通过每次的损失值来判断每次都迭代时是不是损失值都在进行减少
        self.loss_ = []
        for i in range(self.times):
            #求出相应的z的值
            z = np.dot(X,self.w_[1:]) +self.w_[0]
            #计算概率值（结果判定为类别1 的概率值）
            p = self.sigmod(z)
            #根据逻辑回归的代价函数（目标函数）,计算损失值
            #计算公式为：J(w) = - sum(yi* log(s(zi)) + (1-yi) *log(1 - s(zi)))(i 从1 到 n,n为样本的数量）
            cost = -np.sum(y * np.log(p)+(1 - y)*np.log(1 - p))
            self.loss_.append(cost)
            #调整权重值，根据公式：调整为：权重（y 列） = 权重（y列） + 学习率 * sum((y - s(z)) *x(j))
            self.w_[0] += self.alpha * np.sum(y - p)
            self.w_[1:] += self.alpha * np.dot(X.T, y - p)

    def predict_proba(self,X):
        """根据参数传递的样本，对样本数据进行预测（这里的预测时预测它是类别1 和类别2 的概率分别是多少）
        Parameters
        -----
        X:类数组类型，形状：[样本数量，特征数量]
            待测试的样本的特征（属性）
        Returns
        -----
        result:数组类型
            预测结果（概率值） z = np.dot(X, self.w_[1:]) + self.w_[0]\n",
    "        p = self.sigmoid(z)\n",
        """
        #将X转换为数组类型
        X = np.asarray(X)
        #计算z的值
        z  = np.dot(X,self.w_[1:]) + self.w_[0]
        p = self.sigmod(z)
        #将预测结果变成二维数组的结构便于后续的拼接
        p = p.reshape(-1,1)
        #将两个数组进行拼接，并返回
        return np.concatenate([1-p,p],axis = 1)

    def predict(self,X):
        """根据参数传递的样本，对样本数据进行预测属于哪一个类别
        Parameters
        -----
        X:类数组形式，形状：[样本数量，特征数量]
            待测试的样本特征（属性）
        Returns
        -----
        result:数组形式，
            预测的结果（分类值）
        """
        return np.argmax(self.predict_proba(X),axis = 1)

    def score(self,X_test,y_test):
        """根据测试数据集，X_test,和y_test确定当前模型的精确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test,y_predict)

    def __repr__(self):
        return "LogisticRegression()"
