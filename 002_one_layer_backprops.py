import numpy as np


def softmax_ND(X):
    """ X => (N sample, D dimensions) """
    exps = np.exp(X)
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_en_ND(P, Y):
    D = y.shape[0]
    return -np.log(P[range(D), Y]).mean()


def delta_cross_entropy(X, y):
    m = y.shape[0]
    grad = softmax_ND(X)
    grad[range(m), y] -= 1
    grad = grad / m
    return grad


def one_layer(x, y, w, b, lr):
    loss = 1
    while loss > 1e-1:
        fc_out = x.dot(w) + b
        # print("fc out", fc_out)
        sm_out = softmax_ND(fc_out)
        # print("sm out", sm_out)
        loss = cross_en_ND(sm_out, y)
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

    x = np.random.random((n_sample, x_dim))
    y = np.random.randint(3, size=n_sample)  # 0,1,2,3 four classes
    w = np.random.randn(x_dim, fc_out_dim) * np.sqrt(2.0 / fc_out_dim)
    b = np.zeros(fc_out_dim)
    one_layer(x, y, w, b, 0.1)
