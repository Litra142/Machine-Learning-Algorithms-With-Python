import numpy as np
from .CART import CARTRegression
from .util import SquareLoss, CrossEntropyLoss

class GBDT(object):
    """梯度提升树python实现"""
    def __init__(self, n_estimator=10, learning_rate=0.01):
        """
        n_estimator:
            残差树个数
        learning_rate:
            学习率
        """
        self._n_estimator = n_estimator
        self._lr = learning_rate
        # 存储残差树
        self._trees = list()

    def fit(self, X, y):
        """模型训练"""
        pass

    def predict(self, x):
        """给定输入样本，预测输出"""
        pass

class GBDTClassification(GBDT):
    """分类"""
    def __init__(self, n_estimator=10, learning_rate=0.01, min_sample=2, min_gain=0.1, max_depth=10):
        # super.__init__.():就是执行父类的构造函数，使得我们能够调用父类GBDTClassification的属性。
        super(GBDTClassification, self).__init__(n_estimator, learning_rate)
        """
        Parameters
        ------
        n_estimator:
            残差树个数
        min_sample:
            当数据集样本数少于min_sample时不再划分
        min_gain:
            如果划分后收益不能超过该值则不进行划分
            对分类树来说基尼指数需要有足够的下降
            对回归树来说平方误差要有足够的下降
        max_depth:
            树的最大高度
        """
        self._min_sample = min_sample
        self._min_gain = min_gain
        self._max_depth = max_depth
        # 分类树损失函数维交叉熵损失
        self._loss = CrossEntropyLoss()

    def fit(self, X, y):
        """模型训练"""
        # 先对输入标签做one hot编码
        y = self._to_one_hot(y)
        n_sample, self._n_class = y.shape
        # 初始残差为每个类别的平均值
        residual_pred = np.full_like(y, np.mean(y, axis=0))

        for _ in range(self._n_estimator):
            # 遍历残差树
            label_trees = list()
            # 根据之前的残差对新的残差进行更新
            residual_update = np.zeros_like(residual_pred)
            # 每个类别分别学习提升树
            for j in range(self._n_class):
                residual_gradient = self._loss.calc_gradient(y[:, j], residual_pred[:, j])
                tree = CARTRegression(self._min_sample, self._min_gain, self._max_depth)
                # 每棵树以残差为目标进行训练
                tree.fit(X, residual_gradient)
                label_trees.append(tree)
                for i in range(n_sample):
                    residual_update[i, j] = tree.predict(X[i])
            self._trees.append(label_trees)
            residual_pred -= self._lr * residual_update

    def predict(self, x):
        """给定输入样本，预测输出"""
        y_pred = np.zeros(self._n_class)
        for label_trees in self._trees:
            for i in range(len(label_trees)):
                residual_update = label_trees[i].predict(x)
                y_pred[i] -= self._lr * residual_update
        # 返回概率值最大的类别，省略了指数计算
        return np.argmax(y_pred)

    def _to_one_hot(self, y):
        """将离散标签进行one hot编码"""
        n_col = np.amax(y) + 1
        one_hot = np.zeros((y.shape[0], n_col))
        # 将类别所在列置为1
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot

class GBDTRegression(GBDT):
    """回归树"""
    def __init__(self, n_estimator=10, learning_rate=0.01, min_sample=2, min_gain=0.1, max_depth=10):
        super(GBDTRegression, self).__init__(n_estimator, learning_rate)
        # 回归树损失函数维平方损失
        self._loss = SquareLoss()
        for _ in range(self._n_estimator):
            # 递归地建树
            tree = CARTRegression(min_sample, min_gain, max_depth)
            self._trees.append(tree)

    def fit(self, X, y):
        """模型训练"""
        n_sample = y.shape[0]
        residual_pred = np.zeros(n_sample)
        for i in range(self._n_estimator):
            residual_gradient = self._loss.calc_gradient(y, residual_pred)
            # 每棵树以残差为目标进行训练
            self._trees[i].fit(X, residual_gradient)
            residual_update = np.zeros(n_sample)
            for j in range(n_sample):
                residual_update[j] = self._trees[i].predict(X[j])
            residual_pred -= self._lr * residual_update

    def predict(self, x):
        """给定输入样本，预测输出"""
        y_pred = 0
        for tree in self._trees:
            residual_update = tree.predict(x)
            y_pred -= self._lr * residual_update
        return y_pred
