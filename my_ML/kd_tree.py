from operator import itemgetter
import sys
#kd-tree 每个节点中主要包含的数据结构如下
#构造树节点
class KdNode(object):
    """用python实现kd树的结点"""
    def __init__(self,don_elt,spilt,left,right):
        """初始化
        parameters
        -----
        don_elt:类数组形式，形状：[样本数量]
            k维向量节点（k维空间中的一个样本点）
        spilt:int
            进行分割维度的序号
        left:结构
            该结点分割超平面左子空间构成的kd-tree
        right:结构
            该结点分割超平面左子空间构成的kd-tree
        """
        self.dom_elt = don_elt
        self.spilt = spilt
        self.left = left
        self.right = right
class KdTree(object):
    """用python实现 kd-tree"""
    def __init(self,data):
        """初始化
        parameters
        ------
        data:类数组形式，形状:[样本数量]
        """
        k = len(data[0])
        def CreateNode(spilt,dataset):
            #按照spilt指定的维度划分数据集创建KdNode
            #如果数据集为空
            if not dataset:
                return None
            #key参数的值为一个函数，此函数只有一个参数并返回一个值来及逆行比较
            #operator模块中的itemgetter 含糊用于获取对象的那些维度的数据，参数为需要获取的数据子啊对象中的序号、
            #data_set.sort(key = itemgetter(spilt))#按照进行分割的哪一维度数据进行排序
            dataset.sort(key =lambda x:x[spilt])
            #按照python中的除法选择需要划分的维度
            spilt_pos = len(data_set) // 2
            #中位数分割法
            median = data[spilt_pos]
            spilt_next = (spilt+1) % k
            #递归地创建kd树
            return KdNode(median,spilt,
                          CreateNode(spilt_next,dataset[:spilt_pos]),  #创建左子树
                          CreatNode(spilt_next,dataset[spilt_pos+1 :]))
    self.root = CreateCode(0,data)   #从第0维分量开始构建kd树，并返回根节点









