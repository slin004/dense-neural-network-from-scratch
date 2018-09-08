import numpy as np


def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)


def cross_en(p, y):
    return -np.log(p[y])


def delta_cross_entropy(X, y):
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m), y] -= 1
    grad = grad / m
    return grad


def one_layer(x, y, w, b, lr):
    loss = 1
    while loss > 1e-1:
        fc_out = x.dot(w) + b
        # print("fc out", fc_out)
        sm_out = softmax(fc_out)
        # print("sm out", sm_out)
        loss = cross_en(sm_out, y)
        print("loss: ", loss)
        grad_wrt_fc_out = delta_cross_entropy([fc_out], y)[0]
        dw = np.array([x]).T.dot([grad_wrt_fc_out])
        db = grad_wrt_fc_out
        w -= lr * dw
        b -= lr * db


if __name__ == "__main__":
    x_dim = 5
    fc_out_dim = 4

    x = np.random.random(x_dim)
    y = np.random.randint(3, size=1)  # 0,1,2,3 four classes
    w = np.random.randn(x_dim, fc_out_dim) * np.sqrt(2.0 / fc_out_dim)
    b = np.zeros(fc_out_dim)
    one_layer(x, y, w, b, 0.1)
