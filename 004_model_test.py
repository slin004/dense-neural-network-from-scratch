import numpy as np
from output import Output
from hidden import Hidden
from copy import deepcopy

np.random.seed(0)


def one_hidden_model(m_x_train, m_y_train, y_classes, model_lr, epoch, given_w=[], given_b=[]):
    model_n_hidden = 5
    assert (max(m_y_train) <= (y_classes - 1))

    # define layers
    hidden_layers = [
        Hidden(m_x_train.shape[1], model_n_hidden)
    ]
    layer_o = Output(model_n_hidden, y_classes)

    # optional load w b
    if given_w and given_b:
        for i, layer in enumerate(hidden_layers):
            layer.load_params(given_w[i], given_b[i])
        layer_o.load_params(given_w[-1], given_b[-1])

    for i in range(epoch):

        data_for_hidden = m_x_train

        for layer in hidden_layers:
            data_for_hidden = layer.fwd(data_for_hidden)

        layer_o.fwd(data_for_hidden)
        print("Loss: ", layer_o.cross_en(m_y_train))

        # backward
        grad_for_hidden = layer_o.bwd(m_y_train)
        layer_o.upd(model_lr)

        for layer in reversed(hidden_layers):
            grad_for_hidden = layer.bwd(grad_for_hidden)
            layer.upd(model_lr)


if __name__ == "__main__":
    np.random.seed(0)

    x_dim = 3
    n_sample = 1
    n_classes = 4
    learning_rate = 0.1
    epoch = 10

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

    x_train = np.random.random((n_sample, x_dim))
    y_train = np.random.randint(n_classes - 1, size=n_sample)  # 0,1,2,3 four classes
    one_hidden_model(
        deepcopy(x_train),
        deepcopy(y_train),
        deepcopy(n_classes),
        deepcopy(learning_rate),
        deepcopy(epoch),
        given_w=[w1,w2], given_b=[b1,b2])
