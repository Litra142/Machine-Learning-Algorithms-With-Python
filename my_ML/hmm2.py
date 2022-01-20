import numpy as np

class HMMScratch():
    """隐马尔可夫模型Scratch实现"""

    def __init__(self, Q, V, max_iter=10):
        """
        Q:
            状态集合
        V:
            观测集合
        max_iter:
            最大迭代次数
        """
        # 可能的状态个数
        self._N = Q.shape[0]
        # 可能的观测个数
        self._M = V.shape[0]
        self._max_iter = max_iter
        # 状态转移概率矩阵，初始时假设每个状态之间的转移概率均等
        self._A = np.ones((self._N, self._N)) / self._N
        # 观测概率矩阵，初始时假设每个状态生成每个观测的概率均等
        self._B = np.ones((self._N, self._M)) / self._M
        # 初始状态概率向量
        self._pi = np.ones(self._N) / self._N

    def fit(self, O):
        """模型训练"""
        # 观测序列长度
        self._T = O.shape[0]
        # 前向概率，self_alpha[t, i]表示t时刻状态为i且已经知道观测为(o1,o2,...,ot)
        self._alpha = np.zeros((self._T, self._N))
        # 后向概率，self._beta[t, i]表示t时刻状态为i且已经知道观测(ot+1,ot+2,...,oT)
        self._beta = np.zeros((self._T, self._N))

        for _ in range(self._max_iter):
            self._forward(O)
            self._backward(O)

            self._gamma = self._alpha * self._beta

            # 更新状态转移概率矩阵A
            for i in range(self._N):
                for j in range(self._N):
                    v = 0
                    s = 0
                    for t in range(self._T - 1):
                        v += self._calc_ksi(t, i, j, O)
                        s += self._gamma[t, i]
                    self._A[i][j] = v / s

            # 更新观测概率矩阵B
            for j in range(self._N):
                for k in range(self._M):
                    v = 0
                    s = 0
                    for t in range(self._T):
                        if O[t] == k:
                            v += self._gamma[t, j]
                        s += self._gamma[t, j]
                    self._B[j][k] = v / s

            # 更新状态概率
            self._pi = self._gamma[0, i]
            # for i in range(self._N):
            #     self._pi = self._calc_gamma(0, i)

    def predict_output(self, A, B, pi, O):
        """给定参数和观测序列，计算产生该观测的概率"""
        self._T = O.shape[0]
        # 使用前向概率作为输出概率
        alpha = np.zeros((self._T, self._N))
        # t = 0时刻
        alpha[0] = pi * B[:, O[0]]

        # t = 1,2,...,T-1时刻
        for t in range(1, self._T):
            alpha[t] = np.sum(alpha[t - 1] * A.T, axis=1) * B[:, O[t]]
        return np.sum(alpha[-1])

    def predict_output_old(self, A, B, pi, O):
        """给定参数和观测序列，计算产生该观测的概率"""
        self._T = O.shape[0]
        # 使用前向概率作为输出概率
        alpha = np.zeros((self._T, self._N))
        # t = 0时刻
        alpha[0] = pi * B[:, O[0]]

        # t = 1,2,...,T-1时刻
        for t in range(1, self._T):
            for i in range(self._N):
                s = 0
                for j in range(self._N):
                    s += alpha[t - 1][j] * A[j][i]
                alpha[t][i] = s * B[i][O[t]]
        return np.sum(alpha[-1])

    def predict_viterbi(self, A, B, pi, O):
        """给定参数和观测序列，预测概率最大的输入状态"""
        self._T = O.shape[0]
        hidden_state = np.zeros(self._T)
        self._delta = np.zeros((self._T, self._N))
        self._psi = np.zeros((self._T, self._N))
        # t = 0时刻
        self._delta[0] = pi * B[:, O[0]]

        # t = 1,2,..T-1时刻
        for t in range(1, self._T):
            self._delta[t] = np.amax(self._delta[t - 1] * A.T, axis=1) * B[:, O[t]]
            self._psi[t] = np.argmax(self._delta[t - 1] * A.T, axis=1)
        hidden_state[-1] = np.argmax(self._delta[-1])
        prob = np.amax(self._delta[-1])
        for t in range(self._T - 2, -1, -1):
            hidden_state[t] = np.argmax(self._delta[t] * A[:, int(hidden_state[t + 1])], axis=0)
        return hidden_state, prob

    def _forward(self, O):
        """计算前向概率"""
        # t = 0时刻
        self._alpha[0] = self._pi * self._B[:, O[0]]

        # t = 1,2,...,T-1时刻
        for t in range(1, self._T):
            self._alpha[t] = np.sum(self._alpha[t - 1] * self._A.T, axis=1) * self._B[:, O[t]]

    def _backward(self, O):
        """计算后向概率"""
        # t = T-1时刻
        self._beta[-1] = 1

        # t = T-2,T-3,...,0时刻
        for t in range(self._T - 2, -1, -1):
            self._beta[t] = np.sum(self._A * self._B[:, O[t + 1]] * self._beta[t + 1], axis=1)

    # def _calc_gamma(self, t, i):
    #     """计算t时刻处于状态i的概率"""
    #     v = self._alpha[t][i] * self._beta[t][i]
    #     s = 0
    #     for j in range(self._N):
    #         s += self._alpha[t][j] * self._beta[t][j]
    #     return v / s

    def _calc_ksi(self, t, i, j, O):
        """计算t时刻处于状态i，t+1时刻处于状态j的概率"""
        v = self._alpha[t][i] * self._A[i][j] * self._B[j][O[t + 1]] * self._beta[t + 1][j]
        s = 0
        for i in range(self._N):
            for j in range(self._N):
                s += self._alpha[t][i] * self._A[i][j] * self._B[j][O[t + 1]] * self._beta[t + 1][j]
        return v / s
