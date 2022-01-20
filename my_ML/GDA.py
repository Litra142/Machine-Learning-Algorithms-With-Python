# 用python实现高斯线性判别分析
# GDA原理分析：高斯判别分析也是用于分类。
# 对于两类样本，其服从伯努利分布，而对每个类中的样本，假定都服从高斯分布
# y∼Bernouli(ϕ)
# x|y=0∼N(μ0,Σ)
# x|y=1∼N(μ1,Σ)
# 这样，根据训练样本，估计出先验概率以及高斯分布的均值和协方差矩阵（注意这里两类内部高斯分布的协方差矩阵相同），
# 即可通过如下贝叶斯公式求出一个新样本分别属于两类的概率，进而可实现对该样本的分类。
# 　p(y|x)=p(x|y)p(y)p(x)
# y=argmax yp(y|x)=argmax yp(x|y)p(y)p(x)=argmax yp(x|y)p(y)
import numpy as np

class GDA:

    """利用python实现高斯判别分析"""
    def __init__(self,train_data,train_label):
        """初始化
        Parameters
        -----
        train_data:类数组类型，形状:[样本数量，特征数量]
            待训练的样本
        train_label: 数组类型，形状:[样本数量]
        """
        self.Train_Data = train_data
        self.Train_Label = train_label
        self.postive_num = 0 # 初始化正样本的数量为 0
        self.negetive_num = 0  # 初始化负样本的数量为 0
        postive_data = []
        negetive_data = []
        # 计算正负样本的数量以及将其分别分类到相应的列表中
        for (data,label) in zip(self.Train_Data,self.Train_Label):
            if label == 1:
                self.postive_num +=1
                postive_data.append(list(data))
            else:
                self.negetive_num += 1
                negetive_data.append(list(data))
        # 计算正负样本的二项分布的概率
        # 这里计算出来的row和col分别时样本总数量以及特征数量
        row,col = np.shape(train_data)
        # row,col = train_data.shape
        self.postive = self.postive_num *1.0 / row   # 根据公式计算出正样本在总样本数的比例
        # self.negetive = self.negetive_num * 1.0 / row
        self.negetive = 1 - self.postive    # 负样本在总样本数的比例

        # 根据公式计算正负样本的高斯分布的均值向量
        postive_data = np.array(postive_data)  # 先将样本转换为数组的形式，方便后续的操作
        negetive_data = np.array(negetive_data)

        postive_data_sum = np.sum(postive_data,0)   # 分别计算正负样本数据总和
        negetive_data_sum = np.sum(negetive_data, 0)

        self.mu_positive = postive_data_sum * 1.0 / self.postive_num     # 均值向量
        self.mu_negetive = negetive_data_sum * 1.0 / self.negetive_num

        # 计算高斯分布的协方差矩阵
        positive_deta = postive_data-self.mu_positive   # 根据参数计算公式的一部分：x(i)- mu(i)
        negetive_deta = negetive_data - self.mu_negetive
        self.sigma = []
        for deta in positive_deta:
            deta = deta.reshape(1,col)
            ans = deta.T.dot(deta)  # 根据协方差矩阵计算公式 sigma = sum(x(i)- mu(i)*x(i)- mu(i).T)
            self.sigma.append(ans)

        for deta in negetive_deta:
            deta = deta.reshape(1,col)
            ans = deta.T.dot(deta)
            self.sigma.append(ans)
        # 将sigma转换成数组的形式
        self.sigma = np.array(self.sigma)
        self.sigma = np.sum(self.sigma,0)
        self.sigma = self.sigma/ row
        self.mu_positive = self.mu_positive.reshape(1,col)
        self.mu_negetive = self.mu_negetive.reshape(1, col)

    def Gaussian(self,x,mean,cov):
        """自定义的高斯分布概率密度函数
        Parameters
        -----
        x:输入数据
        mean:均值向量
        cov:协方差矩阵
        """
        # 数据的维度
        dim = np.shape(cov)[0]
        # cov的行列式为0 时的措施
        covdet = np.linalg.det(cov + np.eye(dim) * 0.001)
        covinv = np.linalg.inv(cov + np.eye(dim) * 0.001)
        xdiff = (x -mean).reshape((1,dim))
        # 概率密度,公式：1/((2pi)**n/2*|sigma|**1/2 )exp(-1/2(x - mu).T *sigma.I(x - mu))
        prob = 1.0 / (np.power(np.power(2 * np.pi, dim) * np.abs(covdet), 0.5)) * \
               np.exp(-0.5 * xdiff.dot(covinv).dot(xdiff.T))[0][0]
        return prob

    def predict(self,test_data):
        """根据传进来的数据进行预测(通过贝叶斯公式求出一个新样本分别属于两类的概率，进而可实现对该样本的分类。)"""
        predict_label = []
        for data in test_data:
            positive_pro = self.Gaussian(data,self.mu_positive,self.sigma)
            negetive_pro = self.Gaussian(data, self.mu_negetive, self.sigma)
            if positive_pro >= negetive_pro:
                predict_label.append(1)
            else:
                predict_label.append(0)
        return predict_label

    def __repr__(self):
        return(GDA())








