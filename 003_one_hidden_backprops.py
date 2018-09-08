import numpy as np
from output import Output
from hidden import Hidden
from copy import deepcopy

def two_layer(x, y, w01, b1, w12, b2, lr, epoch):
    loss = 1
    for i in range(epoch):
        print("===epoch %d ====" % i)
        fc1_out = x.dot(w01) + b1
        print("x", x, "w", w01, "b", b1, "fc1out", fc1_out)
        rl_out = relu(fc1_out)
        fc2_out = rl_out.dot(w12) + b2
        print("fc2_out: ", fc2_out)
        sm_out = softmax(fc2_out)
        print("sm_out: ", sm_out)
        loss = cross_en(sm_out, y)
        print("loss: ", loss)

        grad_wrt_fc2_out = delta_cross_entropy(fc2_out, y)
        print("fc2 grad", grad_wrt_fc2_out)
        dw12 = rl_out.T.dot(grad_wrt_fc2_out)
        db2 = grad_wrt_fc2_out.sum(axis=0)

        grad_wrt_relu_out = grad_wrt_fc2_out.dot(w12.T)
        print("grad relu", grad_wrt_relu_out)
        grad_wrt_fc1_out = relu_grad_filter(grad_wrt_relu_out)
        print("fc1 grad", grad_wrt_fc1_out)
        dw01 = x.T.dot(grad_wrt_fc1_out)
        db1 = grad_wrt_fc1_out.sum(axis=0)

        w12 -= lr * dw12
        b2 -= lr * db2
        w01 -= lr * dw01
        b1 -= lr * db1


def softmax(x):
    """batch ok"""
    exps = np.exp(x)
    return exps / np.sum(exps, axis=1, keepdims=True)


def relu_grad_filter(grad):
    """batch ok"""
    dx = grad.copy()
    dx[grad <= 0] = 0
    return dx


def cross_en(p, y):
    """batch ok"""
    d = y.shape[0]
    return -np.log(p[range(d), y]).mean()


def relu(x):
    """batch ok"""
    return np.maximum(0, x)


def delta_cross_entropy(X, y):
    """batch ok"""
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m), y] -= 1
    grad = grad / m
    return grad


""""""


def softmax_no_batch(X):
    exps = np.exp(X)
    return exps / np.sum(exps)


def cross_en_no_batch(p, y):
    return -np.log(p[y])


def delta_cross_entropy_no_batch(X, y):
    m = y.shape[0]
    grad = softmax_no_batch(X)
    grad[range(m), y] -= 1
    grad = grad / m
    return grad


def relu_no_batch(x):
    return np.maximum(0, x)


def relu_grad_filter_no_batch(grad):
    dx = grad.copy()
    dx[grad <= 0] = 0
    return dx


def two_layer_no_batch(x, y, w01, b1, w12, b2, lr):
    loss = 1
    # while loss > 1e-1:
    print("****should be all 00000", b1)
    fc1_out = x.dot(w01) + b1
    print("x", x, "w", w01, "b", b1, "fc1out", fc1_out)
    rl_out = relu_no_batch(fc1_out)
    fc2_out = rl_out.dot(w12) + b2
    print("fc2_out: ", fc2_out)
    sm_out = softmax_no_batch(fc2_out)
    print("sm_out: ", sm_out)
    loss = cross_en_no_batch(sm_out, y)
    print("loss: ", loss)

    grad_wrt_fc2_out = delta_cross_entropy_no_batch([fc2_out], y)[0]
    print("fc2 grad", grad_wrt_fc2_out)
    dw12 = np.array([rl_out]).T.dot([grad_wrt_fc2_out])
    db2 = grad_wrt_fc2_out

    grad_wrt_relu_out = grad_wrt_fc2_out.dot(w12.T)
    print("grad relu", grad_wrt_relu_out)
    grad_wrt_fc1_out = relu_grad_filter_no_batch(grad_wrt_relu_out)
    print("fc1 grad", grad_wrt_fc1_out)
    dw01 = np.array([x]).T.dot([grad_wrt_fc1_out])
    db1 = grad_wrt_fc1_out

    w12 -= lr * dw12
    b2 -= lr * db2
    w01 -= lr * dw01
    b1 -= lr * db1


""""""


def one_hidden_model(m_x_train, m_y_train, y_classes, model_lr, h_w, h_b, o_w, o_b, epoch):
    print("===model===")
    model_n_hidden = 5

    assert (max(m_y_train) <= (y_classes - 1))

    layer_h = Hidden(m_x_train.shape[1], model_n_hidden)
    layer_o = Output(model_n_hidden, y_classes)

    layer_h.load_params(h_w, h_b)
    layer_o.load_params(o_w, o_b)

    for i in range(epoch):
        hidden_out = layer_h.fwd(m_x_train)
        layer_o.fwd(hidden_out)
        print("*******loss", layer_o.cross_en(m_y_train))

        grad_o = layer_o.bwd(m_y_train)
        grad_h = layer_h.bwd(grad_o)
        # print("grad layer out", grad_o)
        # print("grad layer hidden", grad_h)

        layer_o.upd(model_lr)
        layer_h.upd(model_lr)


if __name__ == "__main__":
    np.random.seed(0)

    x_dim = 3
    n_sample = 1
    n_classes = 4
    n_hidden = 5
    learning_rate = 0.1
    epoch = 100

    x_train = np.random.random((n_sample, x_dim))
    y_train = np.random.randint(n_classes - 1, size=n_sample)  # 0,1,2,3 four classes

    assert (max(y_train) <= (n_classes - 1))

    w1 = np.array([[-0.89955785, 0.96137636, -0.18429373, -0.08417369, -0.10945882],
                   [-1.11416634, -0.05544932, 0.86449042, 0.71171113, -0.22704872],
                   [0.77198034, -0.84717137, 0.27092711, -0.07808495, 0.89453068]], dtype='float64')
    b1 = np.array([0., 0., 0., 0., 0., ], dtype='float64')
    w2 = np.array([[-0.08771707, 1.4199815, 0.16255433, 0.42772446],
                   [1.15057575, 1.12752457, 0.16294156, -0.04589854],
                   [-0.68517251, 0.4180718, -0.55350587, -0.31412004],
                   [-0.24408347, -0.62352715, -0.31300311, -0.38248558],
                   [-0.93566305, -0.07976088, 0.64159047, 0.57648288]], dtype='float64')
    b2 = np.array([0., 0., 0., 0.], dtype='float64')

    print("x", x_train, "y", y_train)
    # print("=====batch=====")
    # two_layer(
    #     deepcopy(x_train),
    #     deepcopy(y_train),
    #     deepcopy(w1), deepcopy(b1),
    #     deepcopy(w2), deepcopy(b2), deepcopy(learning_rate),
    #     deepcopy(epoch))

    # print("=====no batch=====")
    # two_layer_no_batch(
    #     deepcopy(x_train[0]),
    #     deepcopy(y_train),
    #     deepcopy(w1), deepcopy(b1),
    #     deepcopy(w2), deepcopy(b2), deepcopy(learning_rate))
    #
    print("=====model=====")
    one_hidden_model(
        deepcopy(x_train),
        deepcopy(y_train),
        deepcopy(n_classes),
        deepcopy(learning_rate),
        deepcopy(w1),
        deepcopy(b1),
        deepcopy(w2),
        deepcopy(b2),
        deepcopy(epoch))
