import numpy as np
import pandas as pd

class PCA:
    """用python实现PCA（主成分分析法）"""
    def __init__(self,n_components):
        """初始化PCA"""
        #进行断言
        assert n_components >= 1,\
            "维度必须大于1"
        #将用户传进来的n_components传进到self.n_components中
        self.n_components = n_components
        #设置一个主成分
        self.components_ = None

    def fit(self,X,alpha = 0.01,n_iters = 1e4):
        """获得数据的前n个主成分"""
        assert self.n_components <= X.shape[1],\
            "维度必须小于原本数据的列数（特征数）"

        def demean( X):
            """进行均值归零"""
            # X是一个矩阵，所以i在这里表示的是，将X的每一个样本 - 每一个特征对应的均值（即：一列数据的均值）
            # 将数据进行了向量化了
            return X - np.mean(X, axis=0)

        # 梯度上升法：目标函数f： sum(Xw**2)/len(X)
        def f(w, X):
            return np.sum(X.dot(w) ** 2) / len(X)

        def df(w,X):
            #求目标函数f的梯度df，根据公式推导，最终的结果可以是：
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def direction(w):
            #将w单位化（求它的单位方向向量）
            return w / np.linalg.norm(w)

        def first_component(X,initial_w,alpha = 0.01,n_iters = 1e4,epsilon = 1e-8):
            #求第一主成分
            w = direction(initial_w)   #首先将w单位向量化
            cur_iter = 0
            while cur_iter < n_iters:
                #当循环次数小于我们给定的值时，执行：
                gradient = df(w,X)
                last_w = w
                w = w + alpha * gradient
                w = direction(w)
                if(abs(f(w,X) - f(last_w,X)))< epsilon:
                    #f增加的限度没有超过我们规定的限度时（即：在梯度上升中有点爬不动了）
                    break
                cur_iter += 1
            return w
        #求前n个主成分
        #先对数据进行demean
        X_pca = demean(X)
        self.components_ = np.empty(shape = (self.n_components,X.shape[1]))
        for i in range(self.n_components):
            #确定初始搜索的方向initial_w,但不能将它设置为0
            initial_w = np.random.random(X_pca.shape[1])
            #求X_pca的第一主成分w
            w = first_component(X_pca,initial_w,alpha,n_iters)
            #将w放在self.components_的第i行的位置
            self.components_[i,:] = w
            #将X_pca减去X_pca和刚才求得的样本在w的方向的分量，得到新的X_pca,进行下次循环，依次类推
            X_pca = X_pca - X_pca.dot(w).reshape(-1,1)*w
        return self

    def transform(self,X):
        """将给定的X，映射到pca求得的各个主成分分量中"""
        assert X.shape[1] == self.components_.shape[1]
        return X.dot(self.components_.T)

    def inverse_transform(self,X):
        """将映射后的低维数据重新返回到高维数据来，但是，还是会造成数据的损失"""
        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_compoents = %d" %self.n_components
