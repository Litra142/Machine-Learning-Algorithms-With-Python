import numpy as np
import math
import random

# softmax分类器介绍：Softmax回归可以用于多类分类问题，Softmax代价函数与logistic 代价函数在形式上非常类似，
# 只是在Softmax损失函数中对类标记的 k 个可能值进行了累加。
# 注意在Softmax回归中将 x 分类为类别 j  的概率为：
# p(y(i)=j|x(i);theta)= exp(theta(j).T * x(i))/ sum(exp(theta(l) * x(i))     (注意：l = 1,2,3...k)
# 代价函数为：
# J(theta) = -1/m (sum(sum(1{y(i) = j} log(exp(theta(j).T * x(i))/ sum(exp(theta(l) * x(i)))
# 其中，第一个求和：i = 1,2,3...m    第二个求和：j = 1,2,3...k
# 求导后得到：∇theta(j)J(theta) = -x(i)(1{y(i) = j}-p(y(i) = j|x(i);theta)) + lambda*theta(j)  其中1{y(i) = j}为示例函数

class Softmax():
    """用python实现softmax回归分类器"""
    def __init__(self,alpha = 0.000001,n_iters = 100000,weight_lambda = 0.01):
        """初始化
        Parameters
        -----
        alpha :float
            学习率
        n_iters:int
            最大迭代次数
        weight_lambda :float
            衰退权重
        """
        self.alpha = alpha
        self.n_iters = n_iters
        self.weight_lambda = weight_lambda

    def calc_e(self,x,l):
        """根据公式计算"""
        # 公式为：exp(theta(l) * x(i))
        theta_l= self.w[l]
        product = np.dot(theta_l,x)
        return math.exp(product)

    def calc_probability(self,x,j):
        """根据公式计算将 x 分类为类别 j  的概率"""
        # 公式为：p(y(i)=j|x(i);theta)= exp(theta(j).T * x(i))/ sum(exp(theta(l) * x(i))     (注意：l = 1,2,3...k)
        molecule = self.calc_e(x,j)
        denominator = sum([self.calc_e(x,i) for i in range(self.k)])
        return molecule / denominator

    def calc_partial_derivative(self,x,y,j):
        """根据公式计算损失函数的导数"""
        # 公式：∇theta(j)J(theta) = -x(i)(1{y(i) = j}-p(y(i) = j|x(i);theta)) + lambda*theta(j)  其中1{y(i) = j}为示例函数
        first = int(y == j)   # 计算示性函数
        second = self.calc_probability(x,j)  # 计算后面的概率，公式：p(y(i) = j|x(i);theta)
        # 结果：
        result = -x*(first -second) + self.weight_lambda*self.w[j]
        return result

    def predict_(self,x):
        """预测"""
        result = np.dot(self.w,x)
        row,column = result.shape
        # 找出最大值所在的列
        _positon = np.argmax(result)
        m, n = divmod(_positon ,column)     # divmod() 函数把除数和余数运算结果结合起来，返回一个包含商和余数的元组
        return m

    def fit(self,features,labels):
        """模型训练
        Parameters
        -----
        features:待训练特征数据
        labels:待训练数据对应的标签
        """
        self.k = len(set(labels))   # 用k来保存标签的种类
        # 初始化一个w数组
        self.w = np.zeros((self.k,len(features[0])+1))
        i_iter = 0  # 初始化已经迭代的次数为0
        while i_iter < self.n_iters:
            i_iter += 1      # 迭代次数加1
            index = random.randint(0,len(labels) - 1)   # 随机索引
            x = features[index]                         # 随机取出特征数据中的值
            y = labels[index]                            # 取出对应的标签
            x = list(x)
            x.append(1.0)
            x = np.asarray(x)  # 将x转换为数组形式
            # 计算每一种标签对应数据的损失函数的导数
            derivatives = [self.calc_partial_derivative(x,y,j) for j in range(self.k)]
            for j in range(self.k):
                # 更新权重
                self.w[j] -= self.alpha * derivatives[j]

    def predict(self,features):
        """模型预测
        Parameters
        -----
        features:待预测的特征数据
        Returns
        -----
        label:对应的标签
        """
        labels = []
        for feature in features:
            # feature的shape为784
            x = feature.tolist()
            x.append(1)
            # 将X转换成矩阵的形式
            x = np.matrix(x)
            # 求X的转置
            x = np.transpose(x)
            labels.append(self.predict_(x))
        return labels
    
    def __repr__(self):
        return (Softmax())










