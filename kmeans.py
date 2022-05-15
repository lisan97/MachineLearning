import numpy as np
import random
from sklearn.datasets import load_iris
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans

def kmeans(X,k, max_iter=100):
    '''
    维护两个变量:
    簇中心矩阵(k,X.shape[1])
    簇分配矩阵(X.shape[0],1)
    '''
    centroids = initCentroids(X,k) #1.初始化簇中心
    for _ in range(max_iter):
        y_pre = findNearestCentroids(X,centroids)#2.将每个样本分配给最近的簇中心
        centroids = calculateCentroids(X,y_pre,k)#3.重新计算簇中心
    return y_pre

def initCentroids(X,k):
    n = len(X)
    index = random.sample(range(0,n),k)
    centroids = X[index,:]
    return centroids

def findNearestCentroids(X,centroids):
    n = len(X)
    y_pre = np.zeros(len(X),dtype=int)
    for i in range(n):
        difference = centroids - X[i,:] #1.求每个簇和该样本每个维度的差:(k,X.shape[1])
        square = np.power(difference,2) #2.求每个维度差的平方
        distance = np.sum(square,axis=1) #3.将平方相加 : (k,)
        y_pre[i] = np.argsort(distance)[0] #4.将样本分配给最近的那个簇
    return y_pre

def calculateCentroids(X,y_pre,k):
    n,m = X.shape
    centroids = np.zeros((k,m),dtype=float)
    # 分别求每个簇的新中心
    for i in range(k):
        index = np.where(y_pre == i)[0] # 一个簇一个簇的分开来计算
        tmp = X[index,:] # 每次先取出一个簇中的所有样本
        centroids[i,:] = np.sum(tmp,axis=0) / len(index)
    return centroids

def load_data():
    data = load_iris()
    x, y = data.data, data.target
    return x, y

if __name__ == '__main__':
    x, y = load_data()
    K = len(np.unique(y))
    y_pred = kmeans(x, K)
    nmi = normalized_mutual_info_score(y, y_pred)
    print("NMI by ours: ", nmi)

    model = KMeans(n_clusters=K)
    model.fit(x)
    y_pred = model.predict(x)
    nmi = normalized_mutual_info_score(y, y_pred)
    print("NMI by sklearn: ", nmi)
