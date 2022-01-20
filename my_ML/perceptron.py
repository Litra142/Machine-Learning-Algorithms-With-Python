import numpy as np

class Perceptron():
    """感知机算法python实现"""
    def __init__(self, n_epoch=500, learning_rate=0.1, loss_tolerance=0.001):
        """
        n_epoch:
            训练时迭代多少个epoch之后终止训练
        learning_rate:
            学习率
        loss_tolerance:
            当前损失与上一个epoch损失之差的绝对值小于loss_tolerance时终止训练
        """
        self._n_epoch = n_epoch
        self._lr = learning_rate
        self._loss_tolerance = loss_tolerance

    def fit(self, X, y):
        """
        模型训练
        X:
            训练集，每一行表示一个样本，每一列表示一个特征或属性
        y:
            训练集标签
        """
        n_sample, n_feature = X.shape
        # 获取一个默认的随机数生成器
        rnd_val = 1 / np.sqrt(n_feature)
        rng = np.random.default_rng()
        # 均匀随机初始化权重参数
        self._w = rng.uniform(-rnd_val, rnd_val, size=n_feature)
        # 偏置初始化为0
        self._b = 0

        # 记录迭代次数
        num_epoch = 0
        # 记录当前的损失
        prev_loss = 0
        while True:
            cur_loss = 0
            # 误分类样本个数
            wrong_classify = 0
            for i in range(n_sample):
                # 遍历所有的样本，得到预测值y_pred = w * xi + b
                y_pred = np.dot(self._w, X[i]) + self._b
                # 计算损失和
                cur_loss += -y[i] * y_pred
                # 感知机只对误分类样本进行参数更新
                if y[i] * y_pred <= 0:
                    # 当样本的真实标签与模型的预测异号时，即模型分类错误，进行参数更新
                    self._w += self._lr * y[i] * X[i]
                    self._b += self._lr * y[i]
                    # 误分类样本个数累加
                    wrong_classify += 1
            num_epoch += 1
            # 计算损失之差
            loss_diff = cur_loss - prev_loss
            prev_loss = cur_loss

            # 训练终止条件：
            # 1. 训练epoch数达到指定的epoch数时停止训练
            # 2. 本epoch损失与上一个epoch损失差异小于指定的阈值时停止训练
            # 3. 训练过程中不再存在误分类点时停止训练

            if num_epoch >= self._n_epoch or abs(loss_diff) < self._loss_tolerance or wrong_classify == 0:
                break

    def predict(self, x):
        """给定输入样本，预测其类别"""
        y_pred = np.dot(self._w, x) + self._b
        return 1 if y_pred >= 0 else -1

    def __repr__(self):
        return (Perceptron())