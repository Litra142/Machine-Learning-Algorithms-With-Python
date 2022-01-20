import numpy as np

class GMM():
    """高斯混合模型的python实现"""
    def __init__(self,k = 3,max_iter = 1000,tolerance = 1e-6):
        """初始化
        Parameters
        -----
        k:单高斯模型个数（或者说是聚类中心个数）
        max_iter:最大迭代次数
        tolerance:收敛阈值，当前与上一次迭代之差小于该阈值时认为收敛
        """
        self._k = k
        self._max_iter = max_iter
        self._tolerance = tolerance
        self._gmm_params = list()       # 保存所有高斯模型的参数
        self._resp = None               # 保存响应度
        self._sample_cluster = None     # 保存每个样本所属聚类中心
        self._cur_max_resp = None       # 本轮迭代最大响应度
        self._prev_max_resp = None      # 上一轮迭代最大响应度

    def fit(self,X):
        """训练GMM模型"""
        # 初始化高斯混合模型参数
        self._init_params(X)
        for i in range(self._max_iter):
            # E步: E步的核心就是缺失数据到完全数据的转化，即计算条件概率P(Z|Y,theta(i))
            #     其中，P(Z|Y,theta(i))是给定观测数据Y和当前参数估计theta(i)下隐变量Z的条件概率分布。
            #     记theta(i)为第i次迭代参数theta的估值，在第i+1次迭代的E步，根据准则函数计算: Q(theta,theta(i))
            # 事实上，E步主要求解隐变量条件概率密度P(Z|Y,theta(i))，这一步实现了缺失数据—>完全数据的转化，进而构造准则函数Q，为MLE求解参数（即M步）作准备；
            self._e_step(X)
            # M步: 1）利用MLE求解参数 2)更新各参数
            self._m_step(X)
            if i > 2:
                # 计算响应度之差
                # linalg.norm用来求范数
                resp_diff = np.linalg.norm(self._cur_max_resp - self._prev_max_resp)
                # 如果响应度之差小于给定的阈值，则认为收敛，不再训练下去
                if resp_diff < self._tolerance:
                    break
            self._prev_max_resp = self._cur_max_resp

    def predict(self,X):
        """为每一个样本赋值一个聚类中心"""
        # 使用更新后的参数重新给样本指定中心
        self._e_step(X)
        return self._sample_cluster

    def _e_step(self,X):
        """期望最大化的E步"""
        # E步: E步的核心就是缺失数据到完全数据的转化，即计算条件概率P(Z|Y,theta(i))
        #     其中，P(Z|Y,theta(i))是给定观测数据Y和当前参数估计theta(i)下隐变量Z的条件概率分布。
        #     记theta(i)为第i次迭代参数theta的估值，在第i+1次迭代的E步，根据准则函数计算: Q(theta,theta(i))
        # 事实上，E步主要求解隐变量条件概率密度P(Z|Y,theta(i))，这一步实现了缺失数据—>完全数据的转化，进而构造准则函数Q，
        # 为MLE求解参数（即M步）作准备；
        n_sample = X.shape[0]          # 样本的个数
        # 每一列为一个高斯模型的概率密度,将其初始化为0
        likelihoods = np.zeros((n_sample,self._k))
        for i in range(self._k):
            likelihoods[:,i] = self._alpha[i] * self._gaussian_pdf(X,self._gmm_params[i])
        # 混合高斯模型，每行求和
        sum_likelihoods = np.sum(likelihoods,axis = 1)
        # 计算响应度
        self._resp = likelihoods / np.expand_dims(sum_likelihoods,axis = 1)
        # 为每一个样本赋值一个聚类中心，使用响应度最大的中心
        self._sample_cluster = self._resp.argmax(axis = 1)
        # 保存每轮迭代的最大响应度，用于收敛判断
        self._cur_max_resp = np.amax(self._resp,axis = 1)

    def _m_step(self, X):
        """期望最大化的M步"""
        n_sample = X.shape[0]
        for i in range(self._k):
            # 第i个高斯模型对应的响应度
            resp = np.expand_dims(self._resp[:, i], axis=1)
            # 更新均值
            mean = np.sum(resp * X, axis=0) / np.sum(resp)
            # 更新协方差
            covar = (X - mean).T.dot((X - mean) * resp) / np.sum(resp)
            self._gmm_params[i]["mean"] = mean
            self._gmm_params[i]["cov"] = covar
        # 更新高斯模型的权重
        self._alpha = np.sum(self._resp, axis=0) / n_sample

    def _init_params(self,X):
        """初始化高斯模型参数"""
        n_sample = X.shape[0]
        # 初始每个高斯模型权重均等
        self._alpha = np.ones(self._k) / self._k
        for _ in range(self._k):
            params = dict()
            # 初始化高斯模型的均值和协方差
            rng = np.random.default_rng()
            params["means"] = X[rng.choice(np.arange(n_sample))]
            params["cov"] = np.cov(X.T)
            self._gmm_params.append(params)

    def _gaussian_pdf(self, X, params):
        """给定数据集和高斯模型参数，计算高斯模型概率"""

        n_sample, n_feature = X.shape
        mean = params["mean"]
        covar = params["cov"]
        # 协方差矩阵的行列式
        determinant = np.linalg.det(covar)
        # 概率密度
        likelihood = np.zeros(n_sample)
        # 概率密度前面的系数
        coeff = 1 / np.sqrt(np.power(2 * np.pi, n_feature) * determinant)
        for i, x in enumerate(X):
            # 概率密度指数部分
            exponent = np.exp(-0.5 * (x - mean).T.dot(np.linalg.pinv(covar)).dot(x - mean))
            likelihood[i] = coeff * exponent
        return likelihood

    def __repr__(self):
        return(GMM())






