#进行预测结果的评估
import numpy as np
from math import sqrt
import matplotlib as mpl
import matplotlib.pyplot as plt
def accuracy_score(y_true,y_predict):
    """计算真实值与预测值之间的准确率"""
    assert len(y_true.shape[0]) == len(y_predict.shape[0]), \
        "预测值和真实值地大小必须相等"
    return sum(y_true == y_predict) / len(y_true)
def MSE(y_true,y_predict):
    """计算真实值和预测值之间的MSE"""
    assert len(y_true.shape[0]) == len(y_predict.shape[0]), \
        "预测值和真实值地大小必须相等"
    n= y_true.shape[0]
    mse = sum(np.square((y_true) - y_predict)) / n
    return mse
def RMSE(y_true,y_predict):
    """计算预测值和真实值之间的RMSE"""
    assert len(y_true.shape[0]) == len(y_predict.shape[0]), \
        "预测值和真实值地大小必须相等"
    score = sqrt(MSE(y_true,y_predict))
    return score

def MAE(y_true,y_predict):
    """计算预测值和真实值之间地MAE"""
    assert len(y_true.shape[0]) == len(y_predict.shape[0]), \
        "预测值和真实值地大小必须相等"
    score =  np.sum(np.absolute(y_true,y_predict)) / len(y_true)
    return score
def r2_score(y_true,y_predict):
    """使用R^2对回归结果进行评估"""
    score =  (1 - MSE(y_true,y_predict) / np.var(y_true))
    return score

#分类准确度的指标
def TN(y_true,y_predict):
    """计算混淆矩阵下的TN"""
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 0))

def FP(y_true,y_predict):
    """计算混淆矩阵下的FP"""
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 1))

def FN(y_true,y_predict):
    """计算混淆矩阵下的FN"""
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 0))

def TP(y_true,y_predict):
    """计算混淆矩阵下的FN"""
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))

def confusion_matrix(y_true,y_predict):
    """实现混淆矩阵"""
    return np.array([
        [TN(y_true,y_predict),  FP(y_true,y_predict)],
        [FN(y_true, y_predict), TP(y_true, y_predict)]
    ])

#precision=TP/(TP+FP)
def precision_score(y_true,y_predict):
    """实现精准率"""
    assert len(y_true) == len(y_predict)
    tp = TP(y_true,y_predict)
    fp = FP(y_true,y_predict)
    try:
        return tp / (tp+ fp)
    except:
        return 0.0

#recall=TP/(TP+FN)
def recall_score(y_true,y_predict):
    """实现召回率"""
    assert len(y_true) == len(y_predict)
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0

def f1_score(y_true,y_predict):
    precision = precision_score(y_true,y_predict)
    recall = recall_score(y_true,y_predict)
    try:
        return 2.*precision*recall / (precision +recall)
    except:
        return 0.

def TPR(y_true,y_predict):
    tp = TP(y_true,y_predict)
    fn = FN(y_true,y_predict)
    #检测异常
    try:
        return tp / (tp +fn)
    except:
        return 0.

def FPR(y_true,y_predict):
    fp = FP(y_true,y_predict)
    tn = TN(y_true,y_predict)
    try:
        return fp / (fp +tn)
    except:
        return 0.

#pr曲线的绘制
def PR_curve(y_true,y_predict):
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    y_pre_sort = np.sort(y_predict)[::-1]  #从小到大排序
    index = np.argsort(y_predict)[::-1]  #从大到小排序
    y_true_sort= y_true[index]

    precisions = []
    recalls = []
    for i ,item in enumerate(y_pre_sort):
        if i == 0:
            precisions.append(1)
            recalls.append(0)
        else:
            precisions.append(np.sum((y_true_sort[:i] ==1))/i)
            recalls.append(np.sum((y_true_sort[:i] == 1)) / pos)

    return precisions,recalls

def roc_curve(y_true,y_predict):
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    y_pre_sort = np.sort(y_predict)[::-1]  # 从小到大排序
    index = np.argsort(y_predict)[::-1]  # 从大到小排序
    y_true_sort = y_true[index]
    fprs = []
    tprs = []
    thrs = []   #thresholds
    for i, item in enumerate(y_pre_sort):
        tprs.append(np.sum((y_true_sort[:i] == 1))/ pos)
        fprs.append(np.sum((y_true_sort[:i] == 0)) / neg)
        thrs.append(item)
    return tprs,fprs,thrs

#计算roc曲线下面的面积
def roc_auc_score(y_true,y_predict):
    """用python实现auc值的求解"""
    #计算正负样本的索引，以便索引出之后的概率值
    pos = [i for i in range(len(y_true)) if y_true[i] == 1]
    neg = [i for i in range(len(y_true)) if y_true[i] == 0]

    #初始化auc为任意值，例如为0
    auc = 0
    for i in pos:
        for j in neg:
            if y_predict[i] > y_predict[j]:
                auc+= 1
            elif y_predict[i] == y_predict[j]:
                auc += 0.5
    auc = auc/(len(pos) *len(neg))
    return auc
