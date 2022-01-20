import numpy as np
import matplotlib.pyplot as plt

class Loss(object):
    """损失函数基类"""
    def calc_loss(self, y, y_pred):
        """计算并返回损失"""
        return NotImplementedError()

    def calc_gradient(self, y, y_pred):
        """计算并返回梯度"""
        return NotImplementedError()

class SquareLoss(Loss):
    """平方损失函数"""
    def __init__(self):
        pass

    def calc_loss(self, y, y_pred):
        return 0.5 * np.power(y_pred - y, 2)

    def calc_gradient(self, y, y_pred):
        return y_pred - y

class CrossEntropyLoss(Loss):
    """交叉熵损失"""
    def __init__(self):
        pass

    def calc_loss(self, y, p):
        # 避免对数值为0
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -y * np.log(p) - (1 - y) * np.log(1 - p)

    def calc_gradient(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -y / p + (1 - y) / (1 - p)

def plot_scatter(X, y):
    """绘制聚类散点图"""
    # 不妨取数据的前两列来绘制散点图
    x1 = X[:, 0]
    x2 = X[:, 1]
    for v in np.unique(y):
        x1_v = x1[y == v]
        x2_v = x2[y == v]
        plt.scatter(x1_v, x2_v)
    plt.show()