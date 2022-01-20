import numpy as np
from random import randint
from .CART import CARTClassification  # 导入CART树方便后续建立随机森林
# 随机森林类

class RandomForest():
    """用python实现随机森林"""
    def __init__(self,trees_num= 20,max_depth = 20,leaf_min_size = 1,samples_ratio= 1,feature_ratio = 0.3):
        """初始化
        Parameters
        -----
        trees_num:int
            随机森林中树的数量
        max_depth: int
            树的最大深度
        leaf_min_size: int
            建树时，停止的分支样本的最小数量
        sample_ratio: float
            采样时，创建子树的比例（行比例）
        feature_ratio: float
            特征比例
        """
        self.trees_num = trees_num
        self.max_depth = max_depth
        self.leaf_min_size = leaf_min_size
        self.samples_spilt_ratio = samples_ratio
        self.feature_ratio = feature_ratio
        self.trees = list()  # 森林

    def sample_spilt(self,dataset):
        """有放回的采样，创建数据子集"""
        sample = list()  # 初始化样本列表
        n_sample = round(len(dataset) * self.samples_spilt_ratio)
        while len(sample) < n_sample:
            index = randint(0,len(dataset) - 2)  # 将数据集的索引打乱
            sample.append(dataset[index])
        return sample

    def built_randomforest(self,train,X_train,y_train):
        """建立建立随机森林"""
        max_depth = self.max_depth
        min_size = self.leaf_min_size
        n_trees = self.trees_num
        # 列采样，从所有列中随机选取m列，m < M
        n_features = int(self.feature_ratio *(len(train[0]) - 1))
        for i in range(n_trees):
            # 创建n_trees棵树
            # 选取样本
            sample = self.sample_spilt(train)
            # 创建一棵CART分类树
            self.cart_tree = CARTClassification(min_sample=2, min_gain=1e-6, max_depth=20)
            self.cart_tree.fit(X_train,y_train)
            self.trees.append(self.cart_tree)
        return self.trees

    def bagging_predict(self,onetestdata):
        """随机森林预测的多数表决"""
        # predictions = [self.predict(tree,onetestdata) for tree in self.trees]
        predictions = []
        for tree in self.trees:
            # 遍历随机森林
            # 统计每棵树的预测结果
            predictions.append(self.cart_tree.predict(onetestdata))
        return max(set(predictions),key = predictions.count)


    def accuracy_metrics(self,testdata):
        """计算建立的森林的精确度"""
        correct = 0  # 初始化精确度围为0
        for i in range(len(testdata)):
            # 计算随机森林中每棵树的预测结果
            predicted = self.bagging_predict(testdata[i])
            if testdata[i][-1] == predicted:
                # 判断预测结果是否正确
                correct +=1
        return correct / float(len(testdata))

    def __repr__(self):
        return (RandomForest())




