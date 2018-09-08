import numpy as np


def two_layer(x, y, w01, b1, w12, b2, lr):
    loss = 1
    while loss > 1e-1:
        fc1_out = x.dot(w01) + b1
        rl_out = relu(fc1_out)
        fc2_out = rl_out.dot(w12) + b2
        sm_out = softmax(fc2_out)
        loss = cross_en(sm_out, y)
        print("loss: ", loss)

        grad_wrt_fc2_out = delta_cross_entropy([fc2_out], y)[0]
        dw12 = np.array([rl_out]).T.dot([grad_wrt_fc2_out])
        db2 = grad_wrt_fc2_out

        grad_wrt_relu_out = grad_wrt_fc2_out.dot(w12.T)
        grad_wrt_fc1_out = relu_grad_filter(grad_wrt_relu_out)
        dw01 = np.array([x]).T.dot([grad_wrt_fc1_out])
        db1 = grad_wrt_fc1_out

        w12 -= lr * dw12
        b2 -= lr * db2
        w01 -= lr * dw01
        b1 -= lr * db1


def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)


def relu_grad_filter(grad):
    dx = grad.copy()
    dx[grad <= 0] = 0
    return dx


def cross_en(p, y):
    return -np.log(p[y])


def relu(x):
    return np.maximum(0, x)


def delta_cross_entropy(X, y):
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m), y] -= 1
    grad = grad / m
    return grad


if __name__ == "__main__":
    y = np.random.randint(3, size=1)  # 0,1,2,3 four classes
    x3 = np.random.random(3)
    w35 = np.random.randn(3, 5) * np.sqrt(2.0 / 5)
    b5 = np.zeros(5)
    w54 = np.random.randn(5, 4) * np.sqrt(2.0 / 4)
    b4 = np.zeros(4)
    two_layer(x3, y, w35, b5, w54, b4, 0.1)
