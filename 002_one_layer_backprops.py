import numpy as np


def softmax(x):
    """batch ok"""
    exps = np.exp(x)
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_en(p, y):
    d = y.shape[0]
    return -np.log(p[range(d), y]).mean()


def delta_cross_entropy(x, y):
    """batch ok"""
    m = y.shape[0]
    grad = softmax(x)
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

        # grad_wrt_fc_out.shape ( n_sample, n_classes )
        grad_wrt_fc_out = delta_cross_entropy(fc_out, y)
        # dw.shape ( x_dim, fc_out_dim )
        dw = x.T.dot(grad_wrt_fc_out)
        db = grad_wrt_fc_out.sum(axis=0)
        w -= lr * dw
        b -= lr * db


if __name__ == "__main__":
    x_dim = 50
    fc_out_dim = 100
    n_sample = 3
    n_classes = 3
    learning_rate = 0.1

    x_train = np.random.random((n_sample, x_dim))
    y_train = np.random.randint(n_classes, size=n_sample)
    w_init = np.random.randn(x_dim, fc_out_dim) * np.sqrt(2.0 / fc_out_dim)
    b_init = np.zeros(fc_out_dim)
    one_layer(x_train, y_train, w_init, b_init, learning_rate)
