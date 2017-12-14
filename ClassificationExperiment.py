# !python3
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.model_selection
import math


def loss_fuc(X, y, w, C):
    # print (np.shape(x),np.shape(y),np.shape(w))
    n_sample, n_feature = X.shape
    # w=w[:-2]
    b = w[-1]
    g = lambda _x, _y, _w: 1 - (_y * (np.dot(_x, _w)))[0, 0] if (_y * (np.dot(_x, _w))[0, 0]) < 1 else 0

    loss = [g(X[i].reshape(1, n_feature), y[i].reshape(1, 1), w.reshape(n_feature, 1)) for i in range(n_sample)]
    loss = np.array(loss)
    loss = loss.mean()
    loss = C * loss + (np.linalg.norm(w) ** 2 - b[0] ** 2) / 2.
    return loss


def SGD(X, y, w, C):
    X = X[:, :-1]
    n_sample, n_feature = X.shape
    # print('w',w.shape)
    w = w[:-1]
    # print('w',w.shape)
    b = w[-1]
    # print('b:',b)
    gw = lambda _x, _y, _w, _b: -_y * _x.T if (_y * (np.dot(_x, _w) + _b))[0, 0] <= 1 else np.zeros((_x.T).shape)
    gb = lambda _x, _y, _w, _b: -_y if (_y * (np.dot(_x, _w) + _b))[0, 0] <= 1 else np.zeros(shape=(1, 1))

    np.random.seed(0)
    rand = np.random.randint(n_sample, size=10)
    grad_w = [gw(X[i].reshape(1, n_feature), y[i].reshape(1, 1), w.reshape(n_feature, 1), b.reshape(1, 1)) for i in
              rand]
    grad_w = np.array(grad_w)
    # print('grad_w:',grad_w.shape,grad_w)
    grad_w = np.mean(grad_w, axis=0)
    # print("grad_w",grad_w.shape)
    grad_w = C * grad_w + w
    grad_b = [gb(X[i].reshape(1, n_feature), y[i].reshape(1, 1), w.reshape(n_feature, 1), b.reshape(1, 1)) for i in
              rand]
    grad_b = np.array(grad_b)
    grad_b = np.mean(grad_b, axis=0)
    grad_b = C * grad_b

    grad_w = np.append(grad_w, grad_b, axis=0)
    return grad_w


def NAG(X, y, X_test, y_test):
    n_sample, n_feature = X.shape
    lr = 0.01
    n_estimator = 300
    np.random.seed(0)
    w = np.random.normal(size=(n_feature))
    print(w.shape)
    w = w.reshape(1, n_feature)
    w = w.T
    print(w.shape)
    # grad_w0 = np.zeros(shape=(n_feature + 1, 1))
    v = np.zeros(shape=(n_feature, 1))
    mu = 0.001
    C = 1.
    loss = []

    # grad_w=grad_w0
    for t in range(1, n_estimator):
        loss_value = loss_fuc(X_test, y_test, w, C)
        print("[The %d iteration in test]: %f" % (t, loss_value))
        loss.append(loss_value)
        grad_w = SGD(X, y, w - mu * v, C)
        v = mu * v + lr * grad_w
        # print("w shape:%d,%d"%(w.shape[0],w.shape[1]))
        # print("v shape:%d,%d"%(v.shape[0],v.shape[1]))
        # print("mu shape:%f"%(mu))
        # print("v_pred shape:%d,%d"%(v_pred.shape[0],v_pred.shape[1]))
        w = w - v
        # print (np.shape(w_gradient))
    return w, loss


def RMSProp(X, y, X_test, y_test):
    n_sample, n_feature = X.shape
    lr = 0.1
    n_estimator = 300
    np.random.seed(0)
    w = np.random.normal(size=(n_feature))
    print(w.shape)
    w = w.reshape(1, n_feature)
    w = w.T
    print(w.shape)
    mu = 0.9
    C = 1.
    sigma = 1e-5
    G = 0.
    loss = []
    for t in range(1, n_estimator):
        loss_value = loss_fuc(X_test, y_test, w, C)
        print("[The %d iteration in test]: %f" % (t, loss_value))
        loss.append(loss_value)
        grad_w = SGD(X, y, w, C)
        G = mu * G + (1 - mu) * (np.dot(grad_w.T, grad_w))
        w = w - (lr / np.sqrt(G + sigma)) * grad_w
    return w, loss


def AdaDelta(X, y, X_test, y_test):
    n_sample, n_feature = X.shape
    n_estimator = 300
    np.random.seed(0)
    w = np.random.normal(size=(n_feature))
    w = w.reshape(1, n_feature)
    w = w.T
    mu = 0.01
    C = 1.
    sigma = 1e-5
    G = 0.
    loss = []
    dt = 0.
    for t in range(1, n_estimator):
        loss_value = loss_fuc(X_test, y_test, w, C)
        print("[The %d iteration in test]: %f" % (t, loss_value))
        loss.append(loss_value)
        grad_w = SGD(X, y, w, C)
        G = mu * G + (1 - mu) * (np.dot(grad_w.T, grad_w))
        dw = -(np.sqrt((dt + sigma) / (G + sigma)) * grad_w)
        w = w + dw
        dt = mu * dt + (1 - mu) * np.dot(dw.T, dw)
    return w, loss


def Adam(X, y, X_test, y_test):
    n_sample, n_feature = X.shape
    n_estimator = 300
    np.random.seed(0)
    w = np.random.normal(size=(n_feature))
    w = w.reshape(1, n_feature)
    w = w.T
    mu = 0.999
    C = 1.
    sigma = 1e-5
    belta = 0.9
    eta = 0.1
    m = 1.
    G = 0.
    loss = []
    for t in range(1, n_estimator):
        loss_value = loss_fuc(X_test, y_test, w, C)
        print("[The %d iteration in test]: %f" % (t, loss_value))
        loss.append(loss_value)
        grad_w = SGD(X, y, w, C)
        m = belta * m + (1 - belta) * grad_w
        G = mu * G + (1 - mu) * np.dot(grad_w.T, grad_w)
        w = w - (eta / (1 - belta ** t)) * (np.sqrt((1 - mu ** t) / (G + sigma))) * m

    return w, loss


def Linear_Classification(X, y, X_test, y_test):
    n_sample, n_feature = np.shape(X)
    n_test_sample, n_test_feature = np.shape(X_test)

    X = np.append(np.ones(shape=(n_sample, 1)), X, 1)
    X_test = np.append(np.zeros(shape=(n_test_sample, 1)), X_test, 1)
    X_test = np.append(np.ones(shape=(n_test_sample, 1)), X_test, 1)
    print (n_sample, n_feature)
    print (n_test_sample, n_test_feature)

    w_NAG, loss_NAG = NAG(X, y, X_test, y_test)
    w_RMSProp, loss_RMSProp = RMSProp(X, y, X_test, y_test)
    w_AdaDelta, loss_AdaDelta = AdaDelta(X, y, X_test, y_test)
    w_Adam, loss_Adam = Adam(X, y, X_test, y_test)

    plt.plot(loss_NAG, label='NAG')
    plt.plot(loss_RMSProp, label='RMSProp')
    plt.plot(loss_AdaDelta, label='AdaDelta')
    plt.plot(loss_Adam, label='Adam')
    plt.title('loss')
    plt.xlabel('iteration number')
    plt.ylabel('test datasets loss')
    plt.legend()
    plt.show()


X, y = sklearn.datasets.load_svmlight_file('../data/a9a')
print(y)
X_test, y_test = sklearn.datasets.load_svmlight_file('../data/a9a.t')
X = X.toarray()
print (X.shape)
X_test = X_test.toarray()
print (X_test.shape)
yl = len(y)
y = y.reshape(yl, 1)
ytestl = len(y_test)
y_test = y_test.reshape(ytestl, 1)
# print ('X',X)
# print ('y',y)
# print ('x_vali:',X_vali)
Linear_Classification(X, y, X_test, y_test)
