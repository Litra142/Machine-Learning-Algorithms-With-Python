# 隐马尔可夫模型
# 解决3类问题:
# 1概率问题 2.学习问题（参数估计） 3。预测问题（状态序列的预测）
import numpy as np
from itertools import accumulate

class GenData:
    """根据隐马尔科夫模型生成相应的观测序列
    算法思路：
    输入：隐马尔可夫模型 λ = (A,B,π),观测序列长度T
    输出：观测序列 O = (o1,o2,o3,...,oT)
    """

    def __init__(self,hmm,n_sample):
        """初始化
        Parameters
        -----
        hmm
        n_sample
        """
        self.hmm = hmm
        self.n_sample = n_sample

    def _locate(self,prob_arr):
        """给定概率向量，返回概率分布"""
        seed = np.random.rand(1)  # 生成一个随机数种子
        for state ,cdf in enumerate(accumulate(prob_arr)):
            if seed <= cdf:
                return state
        return

    def init_state(self):
        """根据初始状态概率向量，生成初始状态i1"""
        return self._locate(self.hmm.S)

    def state_trans(self,current_state):
        """状态转移概率分布{a(it)(it+1)"""
        return self._locate(self.hmm.A[current_state])

    def gen_obs(self,current_state):
        """生成观测概率分布b it(k)"""
        return self._locate(self.hmm.B[current_state])

    # 算法流程：
    # 1）按照初始状态分布π产生状态i1
    # 2)令t = 1
    # 3)按照状态it的观测概率分布b it(k)生成ot
    # 4)按照状态it的状态转移概率分布{a(it)(it+1)产生状态it+1,it+1 = 1,2,3,...,N
    def gen_data(self):
        """根据模型生成观测序列"""
        current_state = self.init_state()   # 流程1，生成初始状态i1
        start_obs = self.gen_obs(current_state)  # 流程3，按照状态it的观测概率分布b it(k)生成ot
        state = [current_state]  # 隐状态
        obs = [start_obs]  # 观测数据
        n = 0   # 初始化迭代次数
        while n < self.n_sample -1:
            n += 1
            current_state = self.state_trans(current_state)  # 根据转换矩阵，更新当前的隐状态
            state.append(current_state)  # 将更新后的值保存到之前初始化的列表中
            obs.append(self.gen_obs(current_state))   # 根据更新后的隐状态，更新观测数据，并同时将其保存到列表中
        return state,obs

class HMM:
    """用python实现隐马尔科夫模型"""

    def __init__(self,n_state,n_obs,S = None,A = None,B = None):
        """初始化
        Parameters
        -----
        n_state: int, 状态的个数
        n_obs:int, 观测的种类数
        S: 1*n的矩阵，表示的是：初始状态概率向量
        A: n*n的矩阵，状态转移概率矩阵
        B:n*m的矩阵，观测生成概率矩阵
        """
        self.n_state = n_state
        self.n_obs = n_obs
        self.S = S
        self.A = A
        self.B = B

    def _alpha(self,hmm,t):
        """计算时刻t各个状态的前向概率"""
        b = hmm.B[:,obs[0]]
        alpha = np.array([hmm.S*b])  # n*1，公式：alpha1(i) = π(i)b(i)(o1)  i=1,2,...N
        for i in range(1,t+1):
            alpha = (alpha @ hmm.A) * np.array([hmm.B[:,obs[i]]])
        return alpha[0]

    def forward_pro(self,hmm,obs):
        """向前算法计算最终生成观测序列的概率，即各个状态下概率之和"""
        # 公式：P(O|lambda) = sum(alphaT(i)) ,i = 1,2,3,...,N
        alpha = self._alpha(hmm,obs,len(obs) - 1)
        return np.sum(alpha)

    def _beta(self,hmm,obs,t):
        """计算时刻t各个状态的后向概率"""
        beta = np.ones(hmm.n_state)     # 初始化时刻T的各个隐藏状态后向概率 beta(i) = 1,i = 1,2,3,..N
        for i in reversed(range(t+1,len(obs))):
            # T-1,T-2,...1时刻的后向概率
            # 公式：beta(i)= sum(ai,j* bj(ot+1)*betat+1(j) ,i = 1,2,3,...N
            beta = np.sum(hmm.A*hmm.B[:,obs[i]] * beta,axis = 1)
        return beta

    def backward_prob(self,hmm,obs):
        """利用后向算法计算生成观测序列的概率"""
        deta = self._beta(hmm,obs,0)
        # 计算最终结果，P(O|lambda) = sum(π(i)bi(o1)beta1(i)
        return np.sum(hmm.S * hmm.B[:,obs[0]] * beta)

    def fb_prob(self,hmm,obs, t = None):
        """将前向和后向合并"""
        if t is None:
            t = 0
        # P(it = qi,O|lambda) = alphat(i)*betat(i)
        res = self._alpha(hmm,obs,t) * self._beta(hmm,obs,t)
        return res.sum()

    def _gamma(self,hmm,obs,t):
        """计算时刻t处于各个状态下的概率"""
        # 给定模型λ和观测序列O,在时刻t处于状态qi的概率记为:
        # γt(i)=P(it=qi|O,λ)=P(it=qi,O|λ) / P(O|λ)
        alpha =self._alpha(hmm,obs,t)
        beta = self._beta(hmm,obs,t)
        prob = alpha * beta     # P(it=qi,O|λ)= alphat(i) * betat(i)
        # P(O|λ) = sum(alphat(j)*betat(j))
        return prob / prob.sum()    # return γt(i)

    def point_prob(self,hmm,obs,t,i):
        """计算时刻t处于状态i的概率"""
        # 根据时刻t各个状态的概率即可求出处于状态i的概率
        prob = self._gamma(hmm,obs,t)
        return prob[i]

    def _xi(self,hmm,obs,t):
        """给定模型λ和观测序列O,在时刻t处于状态qi，计算时刻t+1处于状态qj的概率"""
        # 给定模型λ和观测序列O,在时刻t处于状态qi，则时刻t+1处于状态qj的概率记为:
        # ξt(i,j)=P(it=qi,it+1=qj|O,λ)=P(it=qi,it+1=qj,O|λ) / P(O|λ)
        # 而P(it=qi,it+1=qj,O|λ)可以由前向后向概率来表示为: P(it=qi,it+1=qj,O|λ)=alphat(i)aijbj(ot+1)betat+1(j)
        # 最终得到ξt(i,j)的表达式:ξt(i,j) = alphat(i)aijbj(ot+1)betat+1(j) / sum(sum(alphat(r)arsbs(ot+1)betat+1(s)))
        # 注意：第一个sum:r = 1,2,...,N    第二个sum:s = 1,2,3,...,N

        # 根据公式计算一系列参数
        alpha = np.mat(self._alpha(hmm,obs,t)) # 将计算得到的alpha转换成矩阵的形式
        beta_p = self._beta(hmm,obs,t+1)       # beta(t+1)(j):求出计算时刻t+1状态j的后向概率
        obs_prob = hmm.B[:,obs[t+1]]           # bj(此时的状态转移参数为bj)
        obs_beta = np.mat(obs_prob * beta_p)   # bj(ot+1)betat+1(j)
        alpha_obs_beta = np.asarray(alpha.T * obs_beta)     # alphat(i) * bj(ot+1)betat+1(j)
        xi = alpha_obs_beta *hmm.A             # alphat(i)aijbj(ot+1)betat+1(j)

        # return ξt(i,j) = alphat(i)a(ij)b(j)(ot+1)betat+1(j) / sum(sum(alphat(r)arsbs(ot+1)betat+1(s)))
        return xi / xi.sum()

    def fit(self,hmm,obs_data,maxstep = 100):
        """利用Baum-Welch算法学习"""
        # Baum-Welch算法流程：
        # 输入： D个观测序列样本{(O1),(O2),...(OD)}
        # 输出：HMM模型参数
        # 1)随机初始化所有的πi,aij,bj(k)
        # 2)对于每个样本d=1,2,...D，用前向后向算法计算γ(d)t(i)，ξ(d)t(i,j),t=1,2...T
        # 3)更新模型参数：
        #   πi
        #   aij
        #   bj(k)
        # 4) 如果πi,aij,bj(k)的值已经收敛，则算法结束，否则回到第2）步继续迭代。

        # 初始化各个参数
        hmm.A = np.ones((hmm.n_state,hmm.n_state)) / hmm.n_state
        hmm.B= np.ones((hmm.n_state, hmm.n_obs)) / hmm.n_obs
        # 初始化状态概率矩阵（向量），其初始化必须随机状态概率，否则容易陷入局部最优
        hmm.S = np.random.sample(hmm.n_sample)
        hmm.S = hmm.S / hmm.S.sum()
        step = 0
        # 进行迭代
        while step < maxstep:
            xi = np.zeros_like(hmm.A)
            gamma = np.zeros_like(hmm.S)
            B = np.zeros_like(hmm.B)
            # 计算出第一次迭代的S
            S = self._gamma(hmm,obs_data,0)
            for t in range(len(obs_data) - 1):
                tmp_gamma = self._gamma(hmm,obs_data,t)
                gamma += tmp_gamma
                xi += self._xi(hmm,obs_data,t)
                B[:,obs_data[t]] += tmp_gamma
            # 更新A
            for i in range(hmm.n_state):
                hmm.A[i] = xi[i] / gamma[i]
            # 更新B
            tmp_gamma_end = self._gamma(hmm,obs_data,len(obs_data) - 1)
            gamma += tmp_gamma_end
            B[:,obs_data[-1]] += tmp_gamma_end
            for i in range(hmm.n_state):
                hmm.B[i] = B[i] / gamma[i]
            # 更新S
            hmm.S = S
            step += 1
        return hmm

    def predict(self,hmm,obs):
        """采用Viterbi算法预测状态序列"""
        # Viterbi算法流程：
        # 输入：HMM模型λ=(A,B,Π)，观测序列O=(o1,o2,...oT)
        # 输出：最有可能的隐藏状态序列：I∗ = {i∗1,i∗2,...i∗T}
        # 1)初始化局部状态：
        #    δ1(i)=π(i)b(i)(o1),i=1,2...N
        #    Ψ1(i)=0,i=1,2...N

        # 2)进行动态规划递推时刻t=2,3,...T时刻的局部状态：
        #   δt(i)=max1≤j≤N [δt−1(j)aji]bi(0t),i=1,2...N
        #   Ψt(i)=arg max1≤j≤N [δt−1(j)aji],i=1,2...N

        # 3)计算时刻T最大的δT(i),即为最可能隐藏状态序列出现的概率。计算时刻T最大的Ψt(i),即为时刻T最可能的隐藏状态。
        # P∗=max1≤j≤NδT(i)
        # i∗T=arg max1≤j≤N[δT(i)]

        # 4)利用局部状态Ψ(i)开始回溯。对于t=T−1,T−2,...,1：
        #   i∗t=Ψt+1(i∗t+1)

        # 最终得到最有可能的隐藏状态序列I∗={i∗1,i∗2,...i∗T}

        N = len(obs)        # 观测序列的长度
        # 储存时刻t且状态为i时，前一个时刻t - 1的状态，用于构建最终的状态序列
        nodes_graph = np.zeros((hmm.n_state,N),dtype = int)
        # 储存到t时刻，且此刻状态为i的最大概率
        delta = hmm.S * hmm.B[:,obs[0]]
        nodes_graph[:,0] = range(hmm.n_state)

        for t in range(1,N):
            new_delta = []
            for i in range(hmm.n_state):
                # 遍历n个状态
                temp = [hmm.A[j,i] * d for j,d in enumerate(delta)]  # 当状态为i的时候
                max_d = max(temp)       # 得到状态为i的最大概率
                new_delta.append(max_d * hmm.B[i,obs[t]])
                nodes_graph[i,t] = temp.index(max_d)
            delta = new_delta

        current_state = np.argmax(nodes_graph[:,-1])
        path = []
        t = N
        while t > 0:
            path.append(current_state)
            current_state = nodes_graph[current_state,t - 1]
            t -= 1
        return list(reversed(path))

    def __repr__(self):
        return(HMM)
















