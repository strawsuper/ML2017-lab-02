# !python3
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.model_selection


def loss_fuc(X, y, w, alpha):
    # print (np.shape(x),np.shape(y),np.shape(w))
    loss=[np.log1p(np.exp(-(_y * np.dot(_x, w))[0])) for _x, _y in zip(X, y)]
    loss=np.array(loss)
    loss = loss.mean() + (alpha * np.linalg.norm(w) ** 2) / 2.
    # print loss
    return loss


def SGD(X, y, w, alpha):
    n_sample, n_feature = X.shape
    np.random.seed(0)
    rand = np.random.randint(n_sample, size=10)
    sigmoid = lambda _x: 1. / (1+np.exp(-_x))
    gw = lambda _x, _y, _w: -_y * _x.T * sigmoid(-_y * np.dot(_x, _w))
    grad_w = [gw(X[i].reshape(1, n_feature), y[i].reshape(1, 1), w) for i in rand]
    grad_w = np.array(grad_w)
    grad_w = np.mean(grad_w, axis=0)
    # print("grad_w", grad_w.shape)
    grad_w = grad_w + alpha * w
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
    alpha = 1.
    loss = []

    # grad_w=grad_w0
    for t in range(1, n_estimator):
        loss_value = loss_fuc(X_test, y_test, w, alpha)
        print("[The %d iteration in test]: %f" % (t, loss_value))
        loss.append(loss_value)
        grad_w = SGD(X, y, w - mu * v, alpha)
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
    alpha = 1.
    sigma = 1e-5
    G = 0.
    loss = []
    for t in range(1, n_estimator):
        loss_value = loss_fuc(X_test, y_test, w, alpha)
        print("[The %d iteration in test]: %f" % (t, loss_value))
        loss.append(loss_value)
        grad_w = SGD(X, y, w, alpha)
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
    alpha = 1.
    sigma = 1e-5
    G = 0.
    loss = []
    dt = 0.
    for t in range(1, n_estimator):
        loss_value = loss_fuc(X_test, y_test, w, alpha)
        print("[The %d iteration in test]: %f" % (t, loss_value))
        loss.append(loss_value)
        grad_w = SGD(X, y, w, alpha)
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
    alpha = 1.
    sigma = 1e-5
    belta = 0.9
    eta = 0.1
    m = 1.
    G = 0.
    loss = []
    for t in range(1, n_estimator):
        loss_value = loss_fuc(X_test, y_test, w, alpha)
        print("[The %d iteration in test]: %f" % (t, loss_value))
        loss.append(loss_value)
        grad_w = SGD(X, y, w, alpha)
        m = belta * m + (1 - belta) * grad_w
        G = mu * G + (1 - mu) * np.dot(grad_w.T, grad_w)
        w = w - (eta / (1 - belta ** t)) * (np.sqrt((1 - mu ** t) / (G + sigma))) * m

    return w, loss


def Linear_regression(X, y, X_test, y_test):
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
Linear_regression(X, y, X_test, y_test)
