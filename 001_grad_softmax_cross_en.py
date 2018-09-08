import numpy as np


def one_layer_grad_desc_ND(logits, y, lr):
    x = logits.copy()
    loss = 1
    while loss > 1e-1:
        loss = cross_en_ND(softmax_ND(x), y)
        print("loss: ", loss)
        x = x - lr * delta_cross_entropy(x, y)


def cross_en_ND(P, Y):
    D = y.shape[0]
    return -np.log(P[range(D),Y]).mean()


def softmax_ND(X):
    """ X => (N sample, D dimensions) """
    exps = np.exp(X)
    return exps/np.sum(exps, axis=1, keepdims=True)


def delta_cross_entropy(X, y):
    m = y.shape[0]
    grad = softmax_ND(X)
    grad[range(m), y] -= 1
    grad = grad / m
    return grad


if __name__ == "__main__":
    fc_out_dim = 4
    n_sample = 2

    logits = np.random.random((n_sample, fc_out_dim))
    y = np.random.randint(3, size=n_sample)  # 0,1,2,3 four classes
    one_layer_grad_desc_ND(logits, y, 0.1)
