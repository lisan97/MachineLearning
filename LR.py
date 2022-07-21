import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def sigmoid(inx):
    return 1 / (1 + np.exp(-inx))


class SGDLogisticRegression():
    def __init__(self, alpha=0.1, n_iterations=100):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.w = None
        self.b = None

    def initialize_weights(self, n_features):
        limit = 1 / np.sqrt(n_features)
        self.b = 0.
        self.w = np.random.uniform(-limit, limit, (n_features,))

    def batchfit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape
        self.initialize_weights(n)
        for _ in range(self.n_iterations):#梯度下降
            inx = np.dot(X,self.w) + self.b
            p = sigmoid(inx)
            loss_list = y-p
            self.w = self.w + self.alpha*np.dot(X.T,loss_list)/m
            self.b = self.b + self.alpha*np.sum(loss_list)/m

    def SGDfit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape
        self.initialize_weights(n)
        for _ in range(self.n_iterations):#随机梯度下降
            i = np.random.choice(m,1)
            inx = np.dot(X[i],self.w)+self.b
            p = sigmoid(inx)
            loss = y[i] - p
            self.w = self.w + self.alpha*X[i][0]*loss
            self.b = self.b + self.alpha*loss

    def minibatchfit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape
        self.initialize_weights(n)
        for _ in range(self.n_iterations):  # 小批量梯度下降
            idx = np.random.choice(m, 50)
            batch_X = X[idx]
            batch_y = y[idx]
            inx = np.dot(batch_X, self.w) + self.b
            p = sigmoid(inx)
            loss_list = batch_y - p
            self.w = self.w + self.alpha * np.dot(batch_X.T, loss_list) / 50  # 注意正负号
            self.b = self.b + self.alpha * np.sum(loss_list) / 50

    def predict(self, X):
        X = np.array(X)
        inx = np.dot(X, self.w) + self.b
        p_list = np.round(sigmoid(inx))
        return np.array(p_list).astype(int)

if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    y = y[y != 2]
    X = X[:len(y)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    SGD = SGDLogisticRegression(alpha=0.01,n_iterations=100)
    SGD.batchfit(X_train, y_train)
    y_pre = SGD.predict(X_test)
    acc = accuracy_score(y_pre,y_test)
    print('LR by ours', acc)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    acc = accuracy_score(clf.predict(X_test), y_test)
    print('LR by sklearn', acc)