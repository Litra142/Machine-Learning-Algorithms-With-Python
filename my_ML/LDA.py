#利用LDA实现线性判别分析
import numpy as np
import pandas as pd
import matplotlib as mpl

#类间距离SW，类内距离SB矩阵
#S_b*u = lambda * S_w * u  ==> S_w^(-1) * S_b * u = lambda*u
class LDA(object):
    """利用python实现LDA"""
    def __init__(self,num_class = 2,out_dim = 2):
        """初始化
        Parameters
        -----
        num_class :int类型
            默认解决2分类问题
        out_dim :int 类型
            默认将数据处理到二维的平面
        """
        self.num_class = num_class
        self.out_dim = out_dim
        self.W = None               #用来降维的特征向量
        self.eig_pairs = None       #特征值
        self.reduced_data = None    #降维后的数据（reduced_X,y)

    def fit(self,X,y):
        """将传进来的数据进行降维处理
        Parameters
        -----
        X：类数组形式，形状：[样本数量，特征数量]
            待转换的数据
        y:类数组类型，形状：[样本数量]
            数据的标签（默认值从0开始）
        """
        m = X.shape[1]
        #计算均值
        class_mean = self.__calc_class_mean(X,y)
        #计算类间距离
        S_b =  self.__calc_Sb(X,y,class_mean)
        #计算类内距离
        S_w = self.__calc_Sw(X,y,class_mean)
        #得到特征值[w1,w2,...,wi],和特征向量[:,i]
        #numpy.linalg模块中，eigvals函数可以计算矩阵的特征值，而eig函数可以返回一个包含特征值和对应的特征向量的元组
        #使用numpy.linalg模块中的pinv函数进行求解广义逆矩阵，inv函数只接受方阵作为输入矩阵，而pinv函数则没有这个限制
        eig_vals,eig_vecs = np.linalg.eig(np.linalg.pinv(S_w).dot(S_b))
        eig_pairs = [(eig_vals[i],eig_vecs[:,i]) for i in range(len(eig_vals))]
        #返回排序后的结果
        eig_pairs = sorted(eig_pairs,key = lambda x:x[0],reverse = True)
        #默认取前两个特征值对应的特征向量
        out_vecs = []
        for dim in range(self.out_dim):
            out_vecs.append(eig_pairs[dim][1].reshape(m,1))
        self.W = np.hstack(out_vecs)
        self.eig_pairs = eig_pairs
        #将原来的数据映射到得到的维度
        self.reduced_X = X.dot(self.W)
        self.reduced_data = (self.reduced_X,y)
        return self

    def predict(self,X):
        pass

    def __calc_class_mean(self,X,y):
        class_mean = []
        for label in range(self.num_class):
            idx = (y == label)
            vec = np.mean(X[idx], axis=0)
            class_mean.append(vec)
        return np.array(class_mean)

    def __calc_Sb(self, X, y, class_mean):
        """计算类间矩阵，可以通过总体散度矩阵进行优化"""
        m = X.shape[1]
        S_b = np.zeros((m, m))
        all_mean = np.mean(X, axis=0)
        for k in range(self.num_class):
            class_k_mean = class_mean[k]
            n_k = sum(y == k)
            all_mean, class_k_mean = all_mean.reshape(m, 1), class_k_mean.reshape(m, 1)
            S_b += n_k *(class_k_mean - all_mean).dot((class_k_mean - all_mean).T)
        return S_b

    def __calc_Sw(self,X,y,class_mean):
        """类内矩阵"""
        m = X.shape[1]
        S_w = np.zeros((m,m))
        for k in range(self.num_class):
            class_k_mat = np.zeros((m,m))
            class_k_mean = class_mean[k]
            for x_i in X[y == k]:
                x_i ,class_k_mean = x_i.reshape(m,1),class_k_mean.reshape(m,1)
                # class_k_mat += (x_i.dot(x_i.T) - class_k_mean.dot(class_k_mean.T))
                class_k_mat += (x_i - class_k_mean).dot((x_i - class_k_mean).T)

            S_w += class_k_mat
        return S_w

    def plot(self,y):
        """默认降维到二维数据进行可视化"""
        assert self.num_class == 2,\
            "只对降到二维的数据进行可视化"
        for label,marker,color in zip([0,1],('^','s'),("blue","yellow")):
            plt.scatter(x= self.reduced_X[:,0][y == label],
                        y = self.reduced_X[:,1][y == label],
                        marker = marker,
                        color = color,
                        alpha = 0.5,
                        label = str(label)
                        )
        plt.xlabel("X1")
        plt.ylabel("x2")
        plt.title("LDA")
        plt.legend()
        plt.show()

    def __repr__(self):
        return "LDA()"




