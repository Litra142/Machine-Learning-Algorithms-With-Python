import numpy as np

class TreeNode():
    """树节点"""

    def __init__(self,feature_idx = None,feature_val = None,node_val= None,
                 left_child = None,right_child = None):
        """初始化
        Parameters
        -----
        feature_idx:该节点对应的划分特征索引
        feature_val:划分特征的值
        node_val:该节点储存的值，回归树存储结点的平均值，分类树存储出现次数最多的类别
        left_child:左子树
        right_val:右子树
        """
        self._feature_idx = feature_idx
        self._feature_val = feature_val
        self._node_val = node_val
        self._left_child = left_child
        self._right_child = right_child

class CART(object):
    """CART的python实现"""

    def __init__(self,min_sample = 2,min_gain = 1e-6,max_depth = np.inf):
        """初始化
        Parameters
        -----
        min_sample:当数据集样本数量少于min_sample时不再划分
        min_gain:如果划分后收益不能超过该值则不再进行划分（这样的机制有点像在id3和c4.5中小于阈值的时候不再划分）
            因为对于分类树来说，基尼指数需要有足够的下降
            对于回归树来说，平方误差要有足够的下降
        max_depth:树的最大深度
        """
        self._root = None
        self._min_sample = min_sample
        self._min_gain = min_gain
        self._max_depth = max_depth

    def fit(self,X,y):
        """模型训练
        Pararmeters
        -----
        X:训练集，每一行表示一个样本，每一列表示一个特征或属性
        y:训练集标签
        """
        self._root = self._build_tree(X,y)  # 根据训练集来构建决策树

    def predict(self,x):
        """给定输入样本，预测其类别或者输出"""
        return self._predict(x,self._root)

    def _build_tree(self,X,y,cur_depth = 0):
        """构建树
        Parameters
        -----
        X:用于构建子树的数据集
        y:X对应的标签
        cur_depth:当前树的深度
        """
        # 如果子树只剩下一个类别时则置为叶结点（即剩下的样本中标签全部一样）
        if np.unique(y).shape[0] == 1:
            # 那么把出现次数最多的标签作为叶节点的标签（对分类树来说），将剩下的值的平均值作为叶节点的值（对于回归树来说）
            node_val = self._calc_node_val(y)
            return TreeNode(node_val = node_val)

        # 划分前基尼指数或平方差
        before_divide = self._calc_evaluation(y)
        # 基尼指数或平方差最小的划分
        min_evaluation = np.inf  # 先初始化一个最小的基尼指数或平方差
        # 最佳划分的特征索引
        best_feature_idx = None  # 先初始化为None
        # 最佳划分特征的值
        best_feature_val = None  # 同样，也是先初始化为None

        n_sample,n_feature = X.shape    # n_sample:样本数量,n_feature：特征数量
        # 样本数至少大于给定的样本最大值，并且当前树的深度不能超过树的最深度才可以继续进行划分
        if n_sample >= self._min_sample and cur_depth <= self._max_depth:
            # 遍历每一个特征
            for i in range(n_feature):
                feature_value = np.unique(X[:,i])
                # 遍历该特征的每一个值
                for fea_val in feature_value:
                    # 向左划分
                    X_left = X[X[:,i] <= fea_val]
                    y_left = y[X[:,i] <= fea_val]

                    # 向右划分
                    X_right = X[X[:,i] > fea_val]
                    y_right = y[X[:,i] > fea_val]
                    if X_left.shape[0] >0 and y_left.shape[0]> 0 and X_right.shape[0] >0 and y_right.shape[0] > 0:
                        # 当左子树和右子树的元素个数都不为0的时候，再进行划分
                        # 划分后的基尼指数或平方差
                        after_divide = self._calc_division(y_left,y_right)
                        if after_divide < min_evaluation:
                            min_evaluation = after_divide
                            best_feature_idx = i
                            best_feature_val = fea_val

        # 如果划分前后基尼指数或平方差有足够的下降才继续划分
        if before_divide - min_evaluation > self._min_gain:
            # 向左划分
            X_left = X[X[:,best_feature_idx] <= best_feature_val]
            y_left = y[X[:,best_feature_idx] <= best_feature_val]
            # 向右划分
            X_right = X[X[:, best_feature_idx] > best_feature_val]
            y_right = y[X[:, best_feature_idx] > best_feature_val]

            # 构建左子树
            left_child = self._build_tree(X_left,y_left,cur_depth +1 )
            # 构建右子树
            right_child = self._build_tree(X_right, y_right, cur_depth + 1)

            # 返回结点
            return TreeNode(feature_idx=best_feature_idx, feature_val=best_feature_val,
                            left_child=left_child, right_child=right_child)
        # 样本数少于给定的阈值，或者树的高度超过阈值，或者未找到最合适的划分位置为叶结点
        node_val = self._calc_node_val(y)
        return TreeNode(node_val = node_val)
    
    def _predict(self,x,tree = None):
        """给定输入预测输出
        这个函数功能有点像之前决策树的构建中的predict_help"""
        # 根节点
        if tree is  None:  # 即该节点不是叶结点
            tree = self._root
        # 叶结点直接返回预测值
        if tree._node_val is not None:
            return tree._node_val
        # 用该节点对应的划分索引 获取输入样本在对应特征上的取值
        feature_val = x[tree._feature_idx]
        if feature_val <= tree._feature_val:
            # 值小于结点处的判别数，则向左走
            return self._predict(x,tree._left_child)
        # 反之：向右走
        return self._predict(x,tree._right_child)

    def _calc_division(self,y_left,y_right):
        """计算划分后的基尼指数或者平方差"""
        return NotImplementedError()

    def _calc_evaluation(self, y):
        """计算数据集基尼指数或平方差"""
        return NotImplementedError()

    def _calc_node_val(self, y):
        """计算叶结点的值，分类树和回归树分别实现"""
        return NotImplementedError()

class CARTClassification(CART):
    def _calc_division(self, y_left, y_right):
        """计算划分后的基尼指数"""
        # 计算左右子树基尼指数
        left_evaluation = self._calc_evaluation(y_left)
        right_evaluation = self._calc_evaluation(y_right)
        p_left = y_left.shape[0] / (y_left.shape[0] + y_right.shape[0])
        p_right = y_right.shape[0] / (y_left.shape[0] + y_right.shape[0])
        # 划分后的基尼指数
        after_divide = p_left * left_evaluation + p_right * right_evaluation
        # 返回划分后的基尼指数
        return after_divide

    def _calc_evaluation(self, y):
        """计算标签为y的数据集的基尼指数"""
        # 计算每个类别样本的个数
        _, num_ck = np.unique(y, return_counts=True)
        gini = 1
        for n in num_ck:
            gini -= (n / y.shape[0]) ** 2
        return gini

    def _calc_node_val(self, y):
        """分类树从标签中进行投票获取出现次数最多的值作为叶结点的类别"""
        # 计算每个标签及其出现次数
        label, num_label = np.unique(y, return_counts=True)
        # 返回出现次数最多的类别作为叶结点类别
        return label[np.argmax(num_label)]


class CARTRegression(CART):
    def _calc_division(self, y_left, y_right):
        """计算划分后的平方差"""
         # 计算左右子树平方差
        left_evaluation = self._calc_evaluation(y_left)
        right_evaluation = self._calc_evaluation(y_right)
        # 划分后的平方差
        after_divide = left_evaluation + right_evaluation
        return after_divide

    def _calc_evaluation(self, y):
        """计算平方差"""
        return np.sum(np.power(y - np.mean(y), 2))

    def _calc_node_val(self, y):
        """回归树返回标签的平均值作为叶结点的预测值"""
        return np.mean(y)







