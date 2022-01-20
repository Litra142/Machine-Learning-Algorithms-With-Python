import numpy as np
import random

# 这里的svm其实还是用于分类
class SVM():
    """利用python实现支持向量机"""

    def __init__(self,C = 1.0,kernel = "rbf",degree = 3,coef = 0.0,elsilon = 0.001,nepoch= 3000):
        """初始化
        Parameters
        -----
        C:float
            正则化项系数
        kernel:str
            核函数类型,可选类型有：1.linear:线性核函数 2.poly:多项式核函数 3.rbf:高斯核函数
        degree:int
            多项式核函数次数
        coef:float
            多项式核函数中的系数
        elsilon:float
            检验第一个变量对应的样本是否违反KKT条件的检验范围
        nepoch:int
            训练多少个epoch后结束训练
        """
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.coef = coef
        self.elsilon = elsilon
        self.nepoch = nepoch

    def fit(self,X,y):
        """训练模型
        Parameters
        -----
        X:类数组类型，形状：[样本数量，特征数量]
            待训练的样本,每一行表示一个同样，每一表示一个特征
        y:类数组类型，形状:[样本数量]
            待训练样本标签
        """
        self.init_params(X,y)   # 训练的时候首先需要初始化这些参数
        self.smo_outer(y)
        # 将求得的支持向量的索引保存下来
        # squeeze()选择数组中的某一维度移除,
        self.sv_idx = np.squeeze(np.argwhere(self.alpha > 0))
        # 通过将索引带入到原来的数据集中，就可以得到支持向量
        self.sv = X[self.sv_idx]
        # 得到支持向量所对应的标签y
        self.sv_y = y[self.sv_idx]
        # 得到那些支持向量所对应的alpha 注意：我们再预测的时候其实只需要支持向量的alpha，其他非支持向量的alpha不起作用
        self.sv_alpha = self.alpha[self.sv_idx]

    def predict(self,x):
        """给定输入样本，预测其类别，主要模型是由支持向量决定的
        Parameters
        -----
        x:类数组类型，形状：[样本数量，特征数量]
            待预测的样本
        Returns
        -----
        result：数组类型，形状：[样本数量]
            样本的类别
        """
        n_sv = self.sv.shape[0]  # 支持向量的个数
        # 由于在训练数据集上有对训练数据集进行映射到kernel特征空间当中,所以在预测过程中，也需要对预测数据集进行相同的操作
        x_kernel = np.zeros(n_sv)
        for i in range(n_sv):
            x_kernel[i] = self.kernel_transform(self.sv[i],x)
        # 根据公式计算预测值
        y_predict = np.dot(self.sv_alpha *self.sv_y,x_kernel) +self.b
        # 返回预测结果
        return 1 if y_predict >0 else -1

    def smo_outer(self,y):
        """smo外层循环
        X:训练集
        y:训练集标签
        """

        # 注意：统一用下标i表示第一个变量，下标j表示及第二个变量

        num_epoch = 0               # 记录当前迭代了多少次
        traverse_trainset = True    # 标识是否遍历整个训练集
        alpha_change = 0            # 表示α是否已经更新（α>0表示已经更新，反之，则还未进行更新）
        while alpha_change > 0 or traverse_trainset:
            # 只要α已经更新或者设置遍历整个训练集，便开始进行循环
            alpha_change = 0
            if traverse_trainset:
                for i in range(self.m):
                    # 第一种情况：遍历整个训练集去寻找α
                    alpha_change += self.smo_inner(i,y)
            else:
                # 第二种情况：如果不遍历整个训练集的话，那么选择遍历间隔边界的支持向量
                # 取到间隔边界的支持向量
                # 获取间隔边界上支持向量的索引，即 0 < alpha < C对应的支持向量索引
                bound_sv_idx = np.array(np.logical_and(self.alpha > 0,self.alpha < self.C))[0]
                # bound_sv_idx = np.nonzero(np.logical_and(self.alpha > 0,self.alpha < self.C))[0]
                # 将支持向量转换成int类型
                # bound_sv_idx = bound_sv_idx.astype('int')
                for i in range(bound_sv_idx):
                    # 之后来遍历间隔边界的支持向量
                    alpha_change += self.smo_inner(i,y)
            num_epoch += 1
            # 判断是否已经达到最大的迭代次数
            if num_epoch >= self.nepoch:
                break
            # 如果在上一轮遍历了全部的训练数据，那么下一轮迭代需要优先考虑间隔边界的支持向量所对应的样本，去寻找违反KTT的样本
            if traverse_trainset:
                traverse_trainset = False
            elif alpha_change ==0:
                # 如果在间隔边界上没有找到合适的α，那么需要去遍历整个数据集来寻找α
                traverse_trainset = True

    def smo_inner(self,i,y):
        """内部循环
        i:第一个变量的索引
        y:训练集标签
        """
        # 注意：统一用下标i表示第一个变量，下标j表示及第二个变量

        # 首先判断外层循环是违反ktt条件的，所以我们才进行内层循环去寻找这个变量
        if (self.violate_ktt(i,y)):
            # 求出第一个变量预测值和真实输出之差Ei
            Ei = self.g(i,y) - y[i]
            # 计算第二个E2
            j,Ej = self.select_j(i,y)
            # 先将α1和α2拷贝下来
            alpha_i_old = self.alpha[i].copy()
            alpha_j_old = self.alpha[j].copy()

            if y[i] != y[j]:
                # 当yi和yj异号时，分别计算L和H函数的最大值和最小值
                L = max(0,self.alpha[j] - self.alpha[i])
                H = min(self.C,self.C+self.alpha[j] - self.alpha[i])
            else:
                # 当两者同号时，计算
                L = max(0,self.alpha[j] + self.alpha[i]- self.C)
                H = min(self.C,self.alpha[j] + self.alpha[i])
            if L == H:
                # 如果最小值等于最大值（显然时不合适的），E2的变量不合适，退出
                return 0

            eta = self.K[i,i] + self.K[j,j] - 2 * self.K[i,j]   # k11 + k12 + 2k21(或者是2k12)
            if eta <= 0:
                # 根据α2的计算公式原理，eta必须是不等于0的数
                return 0

            # 更新α2（未剪辑时的解）
            self.alpha[j] += y[j] * (Ei - Ej) / eta
            # 对α2根据原理及逆行裁剪,clip()函数用于将α2裁剪到H和L之间
            self.alpha[j] = np.clip(self.alpha[j],L,H)
            # 每次进行一次变量更新时，我们需要将对应的误差也进行更新
            self.updata_E(j,y)
            # 判断α2更新的幅的大小，如果其更新的幅度太小的话，那么我们也可以判定这样的更新时没有意义的，因为它不满足条件：更新前后的变化要足够的大
            if abs(self.alpha[j] - alpha_j_old) < 0.00001:
                return 0
            # 现在再来更新α1，根据公式即可得出
            self.alpha[i] += y[i] *y[j] *(alpha_j_old - self.alpha[j])
            # 同样的，更新误差
            self.updata_E(i,y)

            # 更新b
            b1_new = self.b-Ei-y[i]*self.K[i,i]*(self.alpha[i]-alpha_i_old)-y[j]*self.K[i,j]*(self.alpha[j]-alpha_j_old)
            b2_new=self.b-Ej-y[i]*self.K[i, j]*(self.alpha[i]-alpha_i_old)-y[j]*self.K[j, j]*(self.alpha[j]-alpha_j_old)

            # 更新b的规则：当α1(2)>0 & α1(2)<C 的时候，取b1_new = b2_new = b
            # 否则，取两者的中点（平均值)来更新b
            if self.alpha[i] > 0 and self.alpha[i] < self.C:
                self.b = b1_new
            elif self.alpha[j] > 0 and self.alpha[j] < self.C:
                self.b = b2_new
            else:
                self.b = (b1_new + b2_new) / 2
            # 如果能够找到α1和α2来更新相应的参数时，循环返回1
            return 1

        else:
         # 如果找不到违反ktt条件的，就返回0，放弃本轮的alpha（i）,进行下一轮循环
            return 0

    def select_j(self,i,y):
        """给定第一个变量之后寻找第二个变量，我们选择的是使|Ei - Ej|最大的变量"""
        # 计算Ei
        # global max_Ej
        Ei = self.calc_E(i,y)
        # 更新缓冲
        self.ecache[i] = [1,Ei]
        max_diff = -np.inf    # 由于我们需要找到α使得最大，先将最大初始化为负无穷
        max_j = -1            # 第二个变量的索引
        max_Ej = -np.inf      # 初始化最大的E2(即Ej)为负无穷
        # nonzero:找出非0的索引，而ecache一开始我们将其第一列初始化全为0，后来ecache经过更新后，存在非0 的索引，所以这行代码的功能始，找出非0的索引
        ecache_idx = np.nonzero(self.ecache[:,0])[0]

        if len(ecache_idx) > 0:
            # 如果找到了（len(ecache_idx)大于0），就遍历这些索引
            for j in ecache_idx:
                if j ==i:
                    continue                # 在遍历的时候，如果找到了E1的索引，那么选择跳过E1的索引
                Ej = self.calc_E(j,y)       # 计算Ej相当于计算我们原理中的E2
                diff = abs(Ei - Ej)          # 计算E1 - E2的绝对值
                if diff > max_diff:         # 如果这个绝对值y大于我们初始化的最大值
                    max_diff = diff
                    max_j = j  # 最大的索引为j
                    max_Ej = Ej
            # 返回满足条件的索引,E2(Ej)
            return max_j, max_Ej
        else:  # 如果缓存中无Ej,那么只好随机的选择
            j =i
            while j ==i:
                j = random.randint(0,self.m - 1)
            Ej = self.calc_E(j,y)
            return j,Ej

    def violate_ktt(self,i,y):
        """判断是否违反KTT
        在选择第一个变量时我们选择违反KKT条件的样本所对应的变量，且在epsilon(阈值）范围内检验
        外层循环先遍历间隔边界上的支持向量，并检验这些间隔边界上的支持向量是否满足KKT条件，
        如果全都满足KKT条件，再去遍历整个训练集检验
        间隔边界上的支持向量对应于 0 < alpha < C"""
        # 首先判断α是否在间隔边界
        if self.alpha[i] > 0 and self.alpha[i]< self.C:
            return abs(y[i] * self.g(i,y) - 1) < self.elsilon  # 如果符合这里给出的关系式的话，即为违反了KTT，返回False
        # 反之，返回True
        return True

    def g(self,i,y):
        """计算第i个样本的预测值"""
        return np.dot(self.alpha *y,self.K[i])+ self.b

    def calc_E(self,i,y):
        """计算误差"""
        return self.g(i,y) - y[i]

    def updata_E(self,i,y):
        """更新缓存"""
        Ei = self.calc_E(i,y)
        self.ecache[i] = [1,Ei]

    # 在训练之前，我们应该先初始化一些变量
    def init_params(self,X,y):
        """初始化我们所需要的一些变量"""
        self.m = X.shape[0]            # 样本的数量
        self.n = X.shape[1]            # 数据特征数量
        self.alpha = np.zeros(self.m)  # 模型的α变量，有多少个样本，就有多少α变量
        self.b = 0                     # 初始化偏置b为0

        # 定义我们的误差缓存 ecache （即真实值与预测值之间的差距）
        self.ecache = np.zeros((self.m,2))  # ecache有m行2列，其中第一列用表示变量是否已经发生了更新，另外一列表示具体误差值的大小
        self.K = np.zeros((self.m,self.m))  # K表示的即为kernel ,其大小为m * m
        # 初始化 kernel
        for i in range(self.m):
            for j in range(self.m):
                # 我们实现的svm中kernel 有三种类型，1是线性核，2是多项式核，3是高斯核函数（所以在定义一个接口用来表示使用哪种kernel)
                # 将输入空间种的X转换成特征空间中的X矩阵，将输入空间的数据转换到特征空间中，再进行线性支持向量机
                self.K[i,j] = self.kernel_transform(X[i],X[j])

    def kernel_transform(self,x1,x2):
        """对原始数据做kernel映射
        x1,x2:训练样本"""
        # 通过一个非线性变换将输入空间中的数据映射到特征空间中，方便后续在特征空间中学习线性SVM模型。
        gamma = 1 / self.n  # 默认gamma 为1 / 特征数量（也叫做特征的维度）
        # 之后分为几种情况
        if self.kernel == "linear":
            # 如果是线性核的话，直接计算x1和x2的内积
            return np.dot(x1,x2)
        elif self.kernel == "poly":
            # 如果是多项式核，则根据公式计算即可
            return np.power(gamma * np.dot(x1,x2) + self.coef,self.degree)
        else:
            # 默认是高斯核函数，根据公式返回
            return np.exp(-gamma * np.dot(x1-x2,x1-x2))

    def __repr__(self):
        return(SVM())







