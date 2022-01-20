#梯度下降
import numpy as np
import pandas as pd
from.metrics import r2_score
# 算法的实现
class LR_GradientDescent:
    """用python实现线性回归算法（梯度下降)"""

    def __init__(self, alpha, times):
        """"初始化算法
        Parameters
        -----
        alpha :float
            学习率，用来控制步长，（权重调整的幅度）
        times :int
            循环迭代的次数
            """
        self.alpha = alpha
        self.times = times

    def fit(self, X, y):
        """根据提供的训练数据，对参数进行训练
        Paramenters
            ------
        X:类数组类型。形状：[样本数量，特征数量]
                待训练的特征属性

        y : 类数组类型，形状：[样本数量]
            目标值（标签的属性）
        """
        # 将 X 转化为数组的形式
        X = np.asarray(X).copy()
        # 将 y 转化为数组的形式
        y = np.asarray(y).copy()
        # 创建权重的值，初始值为1（或任何其他的值,长度比特征数量多1，多出的就是截距）
        self.w_ = np.zeros(1 + X.shape[1])
        # 创建损失列表，用来保存每次迭代后的损失数量,损失计算1/2（sum（预测值-真实值）**2）
        self.loss_ = []
        # 进行循环，多次迭代，在每次迭代的过程中，不断取调整权重值，使得损失值不断减小
        for i in range(self.times):
            # y_hat为预测值
            # 由于self.w_的行的数量比X多一列
            y_hat = np.dot(X, self.w_[1:]) + self.w_[0]
            # 计算真实值与预测值之间的差距
            error = y - y_hat
            # 将损失值加入到损失列表中
            self.loss_.append(np.sum(error ** 2) / 2)
            # 根据差距调整权重w_,根据公式，调整为:权重(j) = 权重（j)+ 学习率 * sum((y - y_hat)*x(j))
            #更新权重
            self.w_[0] += self.alpha * np.sum(error)
            self.w_[1:] += self.alpha * np.dot(X.T, error)

    def predict(self, X):
        """根据参数传递的样本，对样本数据进行预测
        Parameters
        -----
        X:类数组类型，形状：[样本数量，特征数量]
            待测试的样本
        Returns
        -----
        result :数组类型
            预测的结果
        """
        X = np.asarray(X)
        result = np.dot(X, self.w_[1:]) + self.w_[0]
        return result

    def score(self,X_test,y_test):
        """根据测试数据集，X_test,和y_test确定当前模型的精确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test,y_predict)

    def __repr__(self):
        return "LR_GradientDescent()"

#最小二乘法
#最小二乘法的思想，将训练集的特征和标签转换成矩阵的形式，之后利用正则化方程，求出数据集的最佳权重，之后再他预测的会员嫁，
#代入最佳权值进行预测，得到预测结果，因为是矩阵的运算，所以我们需要将W0也与矩阵的一行数据进行计算，所以就u需要再原数据集中
#添加一列全为1的列

class LR_LeastSquareMethod2:
    """使用python实现线性回归（最小二乘法的实现）"""
    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None  #系数 θ0
        self.interception_ = None  #截距 [θ1,θ2,...θn].T  系数可以描述我们这个特征对于最终样本的贡献程度
        self._thete = None #整体的θ

    def fit(self,X,y):
        """根据提供的训练数据X，对模型进行训练。

        Paramenters
        ------
        X:类数组类型。形状：[样本数量，特征数量]
            特征矩阵，用来对样本进行训练。

        y : 类数组类型，形状：[样本数量]

        """
        # 考虑截距，在原数据中加入一列全为1的列
        X_b = np.hstack([np.ones((len(X),1)),X])    #np.hstack()水平(按列顺序)把数组给堆叠起来
        #np.linalg.inv()矩阵求逆
        #列出正规方程解
        self._theta =np.linalg.inv( X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.interception_ = self._theta[0]    #截距
        self.coef_= self._theta[1:] #系数
        return self

    def predict(self, X_predict):
        """通过传递的样本X，对样本数据进行预测"""
        X_b = np.hstack([np.ones((len(X_predict),1)),X_predict])
        return X_b.dot(self._theta)

    def score(self,X_test,y_test):
        """根据测试数据集，X_test,和y_test确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test,y_predict)

    def __repr__(self):
        return "LR_LeastSquareMethod2()"


#用python实现加权线性回归
#返回该条样本预测值
def lwlr(testpoint,X_train,y,k = 0.01):
    #由于 k= 1时相当于简单线性回归，而且哟有很高的欠拟合，k = 0.01时，数据拟合情况刚刚好，k = 0.03时，模型考虑了太多噪音数据，呈现过拟合
    #将训练数据集的特征数组和标签数组分别转换成矩阵的形式，方便后续进行计算 注意：这里的y 是y_train.T
    #y_train是m *1的形式,y_train.T是1*m的形式，而y_mat 是m*1的形式
    X_mat = np.mat(X_train)
    y_mat = np.mat(y).T
    m = np.shape(X_train)[0]
    #创建为单位矩阵，再mat 转换数据格式，因为后面是与原数据矩阵运算，所以这里是为了后面运算且不带其他影响
    weigths = np.mat(np.eye(m))
    #利用高斯核公式创建去那种W，遍历数据，给每个数据一个权重
    for j in range(m):
        #高斯公式1
        diffmat = testpoint - X_mat[j,:]
        #高斯核公式2，矩阵*矩阵.T,转行向量为一个值，权重值以指数形式递减
        weigths[j,j] = np.exp(diffmat*diffmat.T/(-2.0*k**2))
        #求回归系数公式1
        xTx = X_mat.T*(weigths*X_mat)
        #判断是否有逆矩阵
        if np.linalg.det(xTx) == 0.0:
            print("行列式为0，奇异矩阵，不能做逆")
            return
        #如果矩阵有逆的话，解线性方程组
        ws = xTx.I*(X_mat.T*(weigths*y_mat))
        #返回预测值
        return testpoint *ws
    #循环所有点求出所有点的预测值
def lwlrTest(X_test,X_train,y,k = 0.01):
    # 传入的k值决定了样本的权重，1和原来一样一条直线，0.01拟合程度不错，0.003纳入太多噪声点过拟合了
    m = np.shape(X_test)[0]
    yHat = np.zeros(m)
    for i in range(m):
        #返回该样本的预测目标值
        yHat[i] = lwlr(X_test[i],X_train,y,k)
    return yHat

#线性回归的正则化--用岭回归实现
def ridgeRegres(X_mat,y_mat,lam = 0.2):
    #用python实现岭回归
    xTx = X_mat.T*X_mat
    #创建一个矩阵，先让他初始化为一个对角矩阵
    m = np.eye(np.shape(X_mat)[1])
    #回归系数计算公式的第一步
    denom = xTx +m*lam
    #判断denom是否不可逆（即不是满秩矩阵
    if np.linalg.det(denom) == 0.0:
        print("该矩阵不是满秩矩阵，不能进行逆运算")
        return
    ws = denom.I*(xTx.T*y_mat)
    return ws

def ridgrTest(X_train,y_train):
    #将传进来的数据转换为矩阵的额形式，方便之后进行计算
    X_mat = np.mat(X_train)
    y_mat = np.mat(y_train)
    #进行岭回归的数据需要进行相应的数据标准化，让每一个特征在有相同的重要性
    y_mean = np.mean(y_mat)
    #进行标准化后的标签数据
    y_mat = y_mat - y_mean
    X_means = np.mean(X_mat,0)
    X_var = np.var(X_mat,0)
    X_mat = (X_mat - X_means) / X_var
    #使用30个lam值
    numTestPts = 30
    w_mat = np.zeros((numTestPts,np.shape(X_mat)[1]))
    #测试不同的lam值，获得不用的系数
    for i in range(numTestPts):
        ws = ridgeRegres(X_mat,y_mat,np.exp(i-10))
        w_mat[i,:] = ws.T
    return w_mat








