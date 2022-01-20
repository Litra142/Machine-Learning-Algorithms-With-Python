#用python实现多层感知器（多层前馈神经网络）
import numpy as np
import random
import pandas as pd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# sigmoid的导数
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

class MLP():
    """用python实现多层感知器"""
    def __init__(self,sizes):
        """初始化
        PArameters
        ------
        sizes:元组形式
            传进来的网络的规格（在本例中使用（784，30，10））
        """
        self.sizes = sizes
        #网络的层数，一般输入层不算进网络层数中，所以减一
        self.num_layers = len(sizes) - 1
        #初始化权重矩阵和偏置矩阵
        #sizes:(784,30,10)
        # w:[ch_out,ch_in]
        # b:[ch_out:1]
        self.weights = [np.random.randn(ch2,ch1) for ch1,ch2 in zip(sizes[:-1],sizes[1:])]  #[784,30],[30,10]
        self.biases = [np.random.randn(ch,1) for ch in sizes[1:]]  #[30,1]

    def forward(self,x):
        """前向传播算法"""
        for b,w in zip(self.biases,self.weights):
            #[30,784]@[781,1] => [30,1]+[30,1] = [30,1]
            z = np.dot(w,x)+b
            #[30,1]
            x= sigmoid(z)
        return x

    def backprop(self,x,y):
        """后向传播算法
        x:训练数据
        y:训练数据标签
        """
        #初始化导数列表
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        #1.前向
        #保存每一层网络的activation
        activations = [x]
        #保存每一层网络的z
        zs = []
        activations = x
        for b,w in zip(self.biases,self.weights):
            #将每一层网络的参数都求出来（前向传播）
            z = np.dot(w,activations)+b
            activations = sigmoid(z)

            zs.append(z)
            activations.append(activations)
        #计算损失值
        loss = np.power(activations[-1]-y,2).sum()

        #2.后向算法
        #2.1计算输出层的梯度
        delta = activations[-1] * (1 - activations[-1]) *(activations[-1] - y)
        #更新b和w的导数
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activations[-2].T)

        #2.2计算隐藏层的梯度
        for l in range(2,self.num_layers +1):
            #后向求梯度，l为正数，反着来，所以对l取导数
            l = -l

            z = zs[l]
            a = activations[l]

            #detla_j 计算并更新隐藏层的梯度
            delta = np.dot(self.weights[l+1].T,delta) * a * (1-a)
            nabla_b[l] = delta
            nabla_w[l] = np.dot(delta,activations[l-1].T)

        return nabla_w,nabla_b

    def train(self,train_data = None,epochs = 10,batch_size = 50,lr = 0.01,test_data = None):
        """训练
        Parameters
        -----
        train_data:训练数据
        epochs:跌打次数
        batch_size:每一个batch的大小
        lr:学习率
        test_data:测试数据
        """
        if test_data:
            #如果有测试数据传进来
            n_test = len(test_data)
        n = len(train_data)

        for j in range(epochs):
            #打乱数据集和顺序
            random.shuffle(train_data)
            mini_batches = [  #每一个小的batch的数据集
                train_data[k:k+batch_size]
                for k in range(0,n,batch_size)
            ]
            #遍历现有数据中的所有batch
            for mini_batch in mini_batches:
                loss = self.update_mini_batch(mini_batch,lr)  #更新损失数值

            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j,self.evaluate(test_data),n_test),loss)
            else:
                print("Epoch {0} compltete".format(j))

    def evaluate(self,test_data):
        """评估"""
        result = [(np.argmax(self.forward(x)),y)for x ,y in test_data]
        #正确的个数
        correct = sum(int(pred == y) for pred,y in result)


    def update_mini_batch(self,batch = None,lr = 0.01):
        """更新每次batch中的参数
        Parameters
        -----
        batch:每一个batch
            (x,y)
        lr:学习率，将默认值设置为0.01"""
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        #遍历每个batch中的数据
        for x,y in batch:
            #计算每个w,b的梯度
            #nabla_w = [w1,w2,w3]
            nabla_w_,nabla_b_ = self.backprop(x,y)
            #之后将所有的梯度累加起来求平均值
            #accu:之前累加的结果
            #现在的结果cur:
            nabla_w = [accu + cur for accu,cur in zip(nabla_w_,nabla_w_)]
            nabla_b = [accu + cur for accu, cur in zip(nabla_b_, nabla_b_)]

        #求平均值
        nabla_w = [w / len(batch) for w in nabla_w]
        nabla_b = [b / len(batch) for b in nabla_b]
        #更新w和b(利用类似梯度下降的那条公式）
        # w = w - lr * nabla_w  # b= b - lr * nabla_b
        self.weights = [w - lr * nabla for w ,nabla in zip(self.weights,self.nabla_w)]
        self.biases = [b - lr * nabla for b, nabla in zip(self.biases, self.nabla_b_)]


    # 定义加载数据的函数，data_folder为保存gz数据的文件夹，该文件夹下有4个文件
    # 'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
    # 't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'

#直接在这里使用main函数即可
def main():

    mnist_data = pd.read_csv(r"E:\360Downloads\python_work\Lib\site-packages\keras\datasets\mnist.npz")
    X = mnist_data['data']
    y = mnist_data['target']
    # 划分数据集
    # 由于mnist数据集的特点，因此，前60000即为训练样本，后10000为测试样本
    X_train = np.array(X[:60000], dtype=float)
    y_train = np.array(y[:60000], dtype=float)
    X_test = np.array(X[60000:], dtype=float)
    y_test = np.array(y[60000:], dtype=float)
    #将训练数据和训练标签重新组合在一起
    train_data=np.concatenate((X_train.copy(),y_train.copy().reshape((-1,1))),axis=1)
    test_data =np.concatenate((X_test.copy(),y_test.copy().reshape((-1,1))),axis=1)
    #模型训练
    mlp = MLP([784,30,10])
    #调用train方法
    mlp.train(train_data,1000,10,0.1,test_data = test_data)

if __name__ == "__main__":
    main()


