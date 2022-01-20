import numpy as np

class WeakClassifier():
    """adaboost弱分类器信息"""

    def __init__(self):
        # 划分特征索引
        self.feature_idx = None
        # 划分值
        self.fea_val = None
        # 取阈值方式，大于或小于划分值
        self.threshold_type = None
        # 弱分类器权重
        self.alpha = None


class Adaboost():
    """adaboost算法python实现"""

    def __init__(self, ncls = 30):
        """
        初始化
        Parameters
        -----
        ncls:
            训练多少个弱分类器
        """
        self._ncls = ncls
        # 将训练过程中的分类器保存到_classifier列表中
        self._classifier = list()

    def fit(self, X, y):
        """模型训练"""
        n_sample, n_feature = X.shape
        # 初始化每个样本权重均等
        w = np.ones(n_sample) / n_sample

        for _ in range(self._ncls):
            # 本次迭代对应的弱分类器
            ws = WeakClassifier()
            # 最小误差
            min_error = np.inf

            # 遍历每一维特征
            for i in range(n_feature):
                # 保存每一位特征中的所有取值
                feature_value = np.unique(X[:, i])
                # 遍历特征的每一个取值作为划分值.(也可以设计步长从特征最小值搜索到最大值)
                for fea_val in feature_value:
                    # 需要考虑大于或小于划分值两种情况下，哪种情况能使分类误差更小
                    # 所以分别计算在大于和小于两种情况下的分类误差，取最小值，作为划分的标准
                    for threshold_type in ["less", "great"]:
                        # "less"：小于，"great":大于
                        # 预测每个样本类别
                        # _stump_predict是一个辅助的预测函数
                        y_pred = self._stump_predict(X, i, fea_val, threshold_type)
                        # 预测错误为1，预测正确为0
                        error_sample = np.ones(n_sample)
                        error_sample[y_pred == y] = 0
                        # 以当前特征的当前划分值划分时，分类误差
                        err = np.dot(w, error_sample)
                        if err < min_error:
                            # 记录误差最小的划分
                            min_error = err
                            ws.feature_idx = i
                            ws.fea_val = fea_val
                            ws.threshold_type = threshold_type
            # 最佳划分情况下对样本类别的预测
            y_pred = self._stump_predict(X, ws.feature_idx, ws.fea_val, ws.threshold_type)

            # 计算弱分类器权重，最小误差可能为0,所以加上一个极小的数1e-15
            ws.alpha = 0.5 * np.log((1 - min_error) / (min_error + 1e-15))
            # 更新每一个样本的权重
            w *= np.exp(-ws.alpha * y * y_pred)
            w /= np.sum(w)

            self._classifier.append(ws)

    def predict(self, x):
        """预测样本类别"""
        y_pred = 0
        for cls in self._classifier:
            pred = 1
            if cls.threshold_type == "less":
                if x[cls.feature_idx] <= cls.fea_val:
                    pred = -1
            else:
                if x[cls.feature_idx] > cls.fea_val:
                    pred = -1
            y_pred += cls.alpha * pred
        return np.sign(y_pred)

    def _stump_predict(self, X, feature_idx, fea_val, threshold_type):
        """
        给定划分特征，划分值，划分类型预测数据集类别
        X:
            数据集
        feature_idx:
            划分索引
        fea_val:
            划分值
        threshold_type:
            划分类型，大于或小于
        """

        y_pred = np.ones(X.shape[0])      # 将预测结果全部初始化为1（全部预测错误）
        if threshold_type == "less":
            y_pred[X[:, feature_idx] <= fea_val] = -1
        else:
            y_pred[X[:, feature_idx] > fea_val] = -1
        return y_pred

    def __repr__(self):
        return("Adaboost()")

