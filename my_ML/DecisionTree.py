import numpy as np

# 用python实现决策树的分类问题
# 定义根节点
class TreeNode():
    """用python表示树的根节点"""
    def __init__(self,feature_idx = None,feature_val = None,feature_name = None,node_val= None,child = None):
        """"初始化
        Parameters
        -----
        feature_idx:特征索引，即以哪一维（即根据数据的哪一个特征）来进行划分
        feature_val:以该特征的哪一个值来进行划分
        feature_name:特征名
        node_val:叶结点值，（无子节点）用来会储存标签信息
        child:子节点，非叶节点储存划分信息
        """
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.feature_name = feature_name
        self.node_val = node_val
        self.child = child

class DecisionTree():
    """利用python实现决策树的构建"""
    def __init__(self,feature_name,etype = "gain",epsilon = 0.01):
        """初始化
        Parameters
        root:根节点
        feature_name:划分特征名称
        etype:str
            划分的标准，即用ID3还是选择用C4.5来对决策树进行划分,默认使用ID3
        epsilon:float,阈值
            当信息增益或者信息增益比小于该阈值时，不进行划分
        """
        self.root = None
        self.feature_name = feature_name
        self.etype = etype
        self.epsilon = epsilon

    def fit(self,X,y):
        """模型训练构建决策树"""
        self.root = self.build_tree(X,y)   # 根据训练集来构建决策树

    def predict(self,x):
        """根据传进来的参数预测"""
        return self.predict_help(x,self.root)

    def predict_help(self,x,tree = None):
        """预测函数的辅助函数
        Parameters
        -----
        x:样本数据
        tree:子树
        Returns
        -----
        返回输入数据地预测值（预测标签）
        """
        if tree is None:
            # 如果子树为空，则将其设置为根节点
            tree = self.root

        if tree.node_val is not None:
            # 判断是否为叶节点，判断的依据是判断其node_val是否不为空，因为在这里的实现中，只有叶节点才会设置标签信息，根节点值存储划分信息
            return tree.node_val
        # 否则，我们现在处于的是一个内部节点的位置，所以需要递归的遍历
        fea_idx = tree.feature_idx  # 先获取子树是根据哪一维度进行划分的
        for fea_val ,child_node in tree.child.items():
            # 遍历子树中的特征值和子节点
            if x[fea_idx] == fea_val:
                # 判断输入样本在划分结点的维度等于特征值，则说明我们找到了对应的结点
                if child_node.node_val is not None:
                    # 如果子结点时叶结点，直接返回叶结点的标签值（即将这个叶子结点地标签值作为我们输入地他预测数据地预测值）
                    return child_node.node_val
                # 否则递归地到子结点中去找这个叶子结点地标签值
                else:
                    return self.predict_help(x,child_node)

    def build_tree(self,X,y):
        """决策树的构建过程"""
        if X.shape[1] == 1:
            # 建树划分的过程中，每次建好一个结点时，需要将样本中的一个特征删掉，所以当只剩下1个特征时，直接将剩下的样本设置为叶子节点
            # 剩下的类别及进行投票，将出现次数最多的类别作为预测叶节点的类别
            node_val = self.vote_label(y)
            # 返回叶节点
            return TreeNode(node_val = node_val)

        if np.unique(y).shape[0] == 1:
            # 如果样本中没有特征（即y只剩下一个类别时）可以进行划分了，那么就将剩下的数据作为叶节点
            # 返回叶节点
            return TreeNode(node_val = y[0])

        # 否则计算信息增益或者时信息增益比来寻找最佳划分
        n_feature =X.shape[1]   # 特征的个数/特征的维度
        max_gain = -np.inf      # 存储最大信息增益挥着是最大信息增益比，将其初始化为负无穷
        max_fea_idx = 0         # 存储最大信息增益或者最大信息增益比的划分索引
        for i in range(n_feature):
            # 遍历每一个特征
            if self.etype == "gain":
                # 如果用户使用信息增益来划分的话，计算信息增益
                gain = self.calc_gain(X[:,i],y)
            else:
                # 计算信息增益比
                gain = self.calc_gain_ratio(X[:,i],y)

            if gain > max_gain:
                max_gain = gain
                max_fea_idx = i

        if max_gain < self.epsilon:
            # 如果计算的信息增益或者信息增益比小于给定的阈值，那么不对它进行划分，直接设置成叶子节点
            node_val = self.vote_label(y)  # 将剩下的样本的标签进行投票
            return TreeNode(node_val = node_val)
        # 否则，则意味找到最佳的划分
        feature_name = self.feature_name[max_fea_idx]
        # 初始子树对象为字典形式
        child_tree = dict()
        for fea_val in np.unique(X[:,max_fea_idx]):    # 遍历我们确定划分的那一列，确定这一列有哪些特征
            child_X = X[X[:,max_fea_idx] == fea_val]     # 将X的max_fea_idx这一列中属于fea_val的样本都单独拿出来，放到child_X中
            child_y = y[X[:,max_fea_idx] == fea_val]     # 同样将他们y的标签也拿出来
            # 将已经划分好的特征删除
            child_X = np.delete(child_X,max_fea_idx,1)  # delete()将矩阵child_X中max_fea_idx这一列删除
            child_tree[fea_val] = self.build_tree(child_X,child_y)  # 拿删除后的子集去建树

        # 这里构造的时内部结点，而内部节点没有标签值，所以只需要往其中传进划分信息，以及字子树信息
        return TreeNode(max_fea_idx,feature_name = feature_name,child = child_tree)

    def vote_label(self,y):
        """投票函数，统计剩余样本中类别出现次数最多的样本类别"""
        # 利用nuique()函数统计剩余样本标签个数以及每个标签出现的次数
        label,num_label = np.unique(y,return_counts = True)
        # 返回出现次数最多的标签，方便之后将它作为叶子结点来建树
        return label[np.argmax(num_label)]

    def calc_entropy(self,y):
        """计算经验熵（y包含了我们所有的类别信息）"""
        entropy = 0  # 初始化熵为0
        _,num_ck = np.unique(y,return_counts = True)   # 计算有多少个类别，rerturn_counts:统计每个类别有出现的次数，并将其返回
        for n in num_ck:
            # 计算每个类别的占比，n为每个类别出现的次数，y.shape[0]为总的数量
            p = n/ y.shape[0]
            entropy -= p * np.log2(p)
        return entropy

    def calc_condition_entropy(self,x,y):
        """计算条件熵
        Parameters
        -----
        x:X中的每一列
        y:样本的标签信息
        Returns
        ------
        con_entropy:返回条件熵
        """
        cond_entropy = 0 # 初始化条件熵为0
        xval,num_x = np.unique(x,return_counts = True)  # xval为这个特征可能有的取值，num_x为每个特征出现的次数
        for v, n in zip(xval,num_x):
            # 同时遍历两个数组
            # 根据公式计算各个部分的值
            y_sub = y[x==v]
            sub_entropy = self.calc_entropy(y_sub)
            p = v / y.shape[0]
            cond_entropy += p * sub_entropy
        return cond_entropy

    # 计算信息增益
    def calc_gain(self,x,y):
        """计算信息增益"""
        return self.calc_entropy(y) - self.calc_condition_entropy(x,y)

    def calc_gain_ratio(self,x,y):
        """计算信息增益比"""
        # 根据公式输入
        return self.calc_gain(x,y) / self.calc_entropy(x)

    def __repr__(self):
        return "DecisionTree()"





