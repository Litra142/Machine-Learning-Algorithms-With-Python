#__train_test_spilt__
import numpy as np
def  train_test_spilt(X,y,test_ratio = 0.2,seed = None):
    """
    将已有的数据划分为训练数据集和测试数据集"""
    #但是呢，我们不能单纯地把数据地前一部分划分为训练数据集，后面的部分为测试数据集，因为我们的数据可能是有规律的划分
    #即，在未划分前，数据集内的数据原本就呈现一种有规律的分布，这样划分的话，不利于我们训练出来的数据集的泛化能力，所以在进行
    #划分前，必须对数据集进行sample的操作
    #首先先判断用户输入的数据是否符合要求
    assert X.shape[0] == y.shape[0],\
        "特征集（属性）的大小必须和标签集的大小相同"
    assert 0.0 <= test_ratio <= 1.0,\
        "测试数据集的比例（test_ratio)必须合法"
    if seed:
        #随机数的设置，可以确保多次打乱的结果是一致的
        np.random.seed(seed)
        #为了确保训练特征与标签的一一确认关系，所以我们可以对索引进行一个打乱的操作
    shuffled_indexes = np.random.permutation(len(X))        #permutation()生成一个随机序列
    #确定测试数据集的大小
    test_size = int(len(X) * test_ratio)
    #选择前test_size个数据作为测试数据集
    #前test_size的数据作为测试数据集
    test_indexes = shuffled_indexes[:test_size]
    train_indexes  = shuffled_indexes[test_size:]
    train_X = X[train_indexes]
    train_y = y[train_indexes]
    test_X = X[test_indexes]
    test_y = y[test_indexes]
    #返回划分好的数据集
    return train_X,train_y,test_X,test_y

def KFolds(X,y,folds = 10):
    #一般使用10_folds,所以将默认值设置为10
    #注意：传进来的样本必须先进行随机的划分，所以我们可以调用上面的方法
    #得到随机划分好的数据
    train_X, train_y, test_X, test_y =  train_test_spilt(X,y,666)
    #一般我们只需要使用train_X,和train_y
    X_folds = []
    y_folds = []
    #vsplit()用于将数据划分，具体的用法鉴于说明文档
    X_folds = np.vspilt(X,folds)
    y_folds = np.vspilt(y, folds)
    #spilt the train sets and validation sets
    for i in range(folds):
        X_train = np.vstack(X_folds[:i] + X_folds[i+1:])
        X_val = X_folds[i]
        y_train = np.vstack(y_folds[:i] + y_folds[i + 1:])
        y_val = y_folds[i]
        return X_train,X_val,y_train,y_val

