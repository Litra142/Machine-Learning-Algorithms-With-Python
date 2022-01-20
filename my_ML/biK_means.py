#为克服K-均值算法收敛于局部最小值的问题，有人提出了另一个称为二分K均值（bisectingK-means)的算法, 该算法首先将所有点作为一个簇，然后将该簇一分为二。之后选择其中一个簇继续进行划分，选择哪一个簇进行划分取决于对其划分是否可以最大程度降低SSE的值。上述基于SSE的划分过程不断重复，直到得到用户指定的簇数目为止。
import numpy as *
import pandas as pd

#计算距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

def biKmeans(dataSet,k,distMeas=distEclud):
    m = shape(dataSet)[0]
    # 将所有的点看成是一个簇
    # clusterAssment 存储 （所属的中心编号，距中心的距离）的列表
    clusterAssment = mat(zeros((m,2)))
    # centList 存储聚类中心
    centroid0 =mean(dataSet,axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j,:]) ** 2
    # 当簇小于数目k时
    while len(centList) < k:
        lowestSSE = inf
        for i in range(len(centList)):
            # 得到dataSet中行号与clusterAssment中所属的中心编号为i的行号对应的子集数据。
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]
            # 在给定的簇上进行K-均值聚类,k值为2
            centroidMat,splitClustAss = kMeans(ptsInCurrCluster,2,distMeas)
           # 计算将该簇划分成两个簇后总误差
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1])
           # 选择使得误差最小的那个簇进行划分
            if sseSplit + sseNotSplit < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat.copy()
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

       # 将需要分割的聚类中心下的点进行1划分
       # 新增的聚类中心编号为len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss

        # 更新被分割的聚类中心的坐标
        centList[bestCentToSplit] = bestNewCents[0,:]
        # 增加聚类中心
        centList.append(bestNewCents[1,:])

    return centList,clusterAssment