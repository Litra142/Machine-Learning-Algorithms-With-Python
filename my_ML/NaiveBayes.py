import numpy as np

class Gaussian_NB:
    def __init__(self):
        """初始化"""
        self.num_of_samples = None  # 总样本数
        self.num_of_class = None
        self.class_name = []
        self.prior_prob = []
        self.X_mean = []   # 每一列样本数据的平均值
        self.X_var = []    # 每一列样本的方差

    def _SepByClass(self,X, y):
        """按照类别分割数据
        Pararmeters
        -----
        X:类数组类型，形状：[样本数量，特征数量]
            未分类的特征
        y :类数组类型，形状:[样本数量]
            未分类的属性
        Returns
        -----
        data_byclass :字典类型
            分类完成的数据
        """

        self.num_of_samples = len(y)    # 样本总数量

        # 为了避免X和 y的数据的规格不一样，所以对y稍加处理
        y = y.reshape(X.shape[0],1)
        data = np.hstack((X,y))     # 将特征和属性合并成完整数据
        data_byclass = {}           # 初始化分类数据，为一个空字典
        # 提取各类别数据，字典的键为类别名，值为对应的分类数据
        for i in range(len(data[:,-1])):
            if i in data[:,-1]:
                data_byclass[i] = data[data[:,-1] == i]
        self.class_name =list(data_byclass.keys()) # 类别名
        self.num_of_class = len(data_byclass.keys()) # 类别总数

        return data_byclass

    def _calc_PriorProb(self,y_byclass):
        """计算y的先验概率（使用拉普拉斯平滑）
        Parameters
        -----
        y_byclass:
            当前类别下的目标
        Returns
        ------
        result :float
            类别的先验概率
        """
        # 如果Xi是离散的值，可以假设Xi符合多项式分布，这样得到P(Xj=X(test)j|Y=Ck) 是在样本类别Ck中，特征X(test)j出现的频率。
        # 某些时候，可能某些类别在样本中没有出现，这样可能导致P(Xj=X(test)j|Y=Ck)为0，这样会影响后验的估计，为解决这种情况，引入了拉普拉斯平滑
        # 其中λ 为一个大于0的常数，常常取为1。
        # 所以计算公式：（当前类别下的样本数+1）/（总样本数+类别总数）
        result = (len(y_byclass) + 1) / (self.num_of_samples +self.num_of_class)
        return result

    def _calc_XMean(self,X_byclass):
        """计算各类别特征各维度的平均值
        Parameters
        -----
        X_byclass:
            当前类别下的特征
        Returns
        ------
        X_mean :
            该特征各个维度的平均值
        """
        X_mean = []
        for i in range(X_byclass.shape[1]):
            X_mean.append(np.mean(X_byclass[:,i]))
        return X_mean

    def _calc_XVar(self,X_byclass):
        """计算各类别特征各维度的方差
        Parameters
        -----
        X_byclass:
            当前类别下的特征
        Returns
        ------
        X_var :
            该特征各个维度的方差
        """
        X_var = []
        for i in range(X_byclass.shape[1]):
            X_var.append(np.var(X_byclass[:, i]))
        return X_var

    def _calc_GaussianProb(self,X_new,mean,var):
        """计算训练集特征（符合正态分布）在各类别下的条件概率
        Parameters
        -----
        X_new :类数组类型，形状：[特征数量]
            新样本的特征
        mean:float
            训练集特征的平均值
        var:float
            训练集特征的方差
        Returns
        -----
        gaussian_prob:
            输出新样本的特征在相应训练集中的分布概率
        """
        # 如果Xi是连续值，通常取Xi的先验概率为正态分布，即在样本类别Ck中，Xj的值符合正态分布。
        # 根据计算公式（高斯分布的公式）：(np.exp(-(X_new-mean)**2/(2*var)))*(1/np.sqrt(2*np.pi*var))
        # 初始化一个用来存放条件概率的列表
        gaussian_prob = []
        for a,b,c in zip(X_new, mean, var):    # zip(X_new,mean,var)可以同时访问三个序列
            formula1 = np.exp(-(a - b) ** 2 / (2*c))    # 公式左边部分
            formula2 = 1 / np.sqrt(2 * np.pi*c)         # 公式右边部分
            gaussian_prob.append(formula1 * formula2)

        return gaussian_prob

    def fit(self,X,y):
        """训练数据集
        Parameters
        -----
        X:类数组类型，形状 :[样本数量，特征数量]
            待训练的数据集的特征
        y:类数组类型，形状：[样本数量]
            训练数据集的目标
        Returns
        ------
        self.prior_prob:目标的先验概率
        self.X_mean:特征的平均值
        self.X_var:特征的方差
        """
        # 首先将传进来的X,y转换成array的形式
        X = np.asarray(X,np.float32)
        y = np.asarray(y, np.float32)
        # 将数据分类（即算出数据又多少种类别，并划分开来）
        data_byclass = self._SepByClass(X,y)
        # 计算各类别数据的目标先验概率，特征平均值和方差
        for data  in data_byclass.values():
            X_byclass = data[:,:-1]
            y_byclass = data[:,-1]
            # 求先验概率
            self.prior_prob.append(self._calc_PriorProb(y_byclass))
            # 求特征的平均值,方差
            self.X_mean.append(self._calc_XMean(X_byclass))
            self.X_var.append(self._calc_XVar(X_byclass))

        return self.prior_prob,self.X_mean,self.X_var

    def predict(self,X_new):
        """预测数据
        Parameters
        -----
        X_new:类数组类型，形状：[特征数量 ]
            新样本的特征
        Returns
        -----
        self.class_name:新样本最有可能的目标
        """
        # 同样的，也是将X_new转换成numpy数组）的形式
        X_new = np.asarray(X_new,np.float32)
        # 初始化后验概率
        posteriori_prob = []
        # 计算后验概率
        for i,j,o in zip(self.prior_prob,self.X_mean,self.X_var):
            # 根据传进来的先验概率，均值和方差计算后验概率，返回后验概率最大的类别作为最终的预测结果
            gaussian = self._calc_GaussianProb(X_new, j, o)
            posteriori_prob.append(np.log(i) + sum(np.log(gaussian)))
            idx = np.argmax(posteriori_prob)

        return self.class_name[idx]

    def __repr__(self):
        return(Gaussian_NB())
















