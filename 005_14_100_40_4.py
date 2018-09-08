import numpy as np
from layer import Output, Hidden
from copy import deepcopy
from helper import load_datasets, load_100_40_4_params

np.random.seed(0)


def model(m_x_train, m_y_train, y_classes, model_lr, epochs,
          batch=1,
          given_w=[], given_b=[], save_params=False):
    # assert (max(m_y_train) <= (y_classes - 1))

    # define layers
    hidden_layers = [
        Hidden(m_x_train.shape[1], 100, name="14x100"),
        Hidden(100, 40, "100x40")
    ]
    layer_o = Output(40, y_classes, "40x4")

    # optional load w b
    if given_w and given_b:
        for i, layer in enumerate(hidden_layers):
            layer.load_params(given_w[i], given_b[i])
        layer_o.load_params(given_w[-1], given_b[-1])

    loss_count = 0
    total_loss = 0
    infer_correct = 0

    for epoch in range(epochs):
        batch_p = 0
        while batch_p < len(m_x_train):
            x_batch = m_x_train[batch_p:batch_p + batch]
            y_batch = m_y_train[batch_p:batch_p + batch]
            batch_p += batch

            #forward
            data_for_hidden = x_batch
            for layer in hidden_layers:
                data_for_hidden = layer.fwd(data_for_hidden)
            sm_out = layer_o.fwd(data_for_hidden)

            # metrics
            y_hat = np.argmax(sm_out, axis=1)
            infer_correct += (y_hat == y_batch).sum()
            total_loss += layer_o.cross_en(y_batch)
            loss_count +=1
            # print("loss - ",layer_o.cross_en(y_batch))

            # backward
            grad_for_hidden = layer_o.bwd(y_batch)
            # print(grad_for_hidden.shape)
            # print(grad_for_hidden)

            layer_o.upd(model_lr)
            for layer in reversed(hidden_layers):
                grad_for_hidden = layer.bwd(grad_for_hidden)
                layer.upd(model_lr)


        # metrics per epoch
        print(epoch, " - epoch train: loss - ", (total_loss/loss_count), " accuracy - ", (infer_correct/len(m_x_train)))
        loss_count = 0
        total_loss = 0
        infer_correct = 0

        if save_params:
            for i, layer in enumerate(hidden_layers):
                layer.save_params()
            layer_o.save_params()



if __name__ == "__main__":
    np.random.seed(0)

    x_train_real, x_test, y_train_real, y_test = load_datasets()
    w, b = load_100_40_4_params()

    # X = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]])
    # label = np.array([3])
    # label = np.array([[0,0,0,1]])
    x_dim = 14
    n_sample = 5
    n_classes = 4

    x_train = np.random.random((n_sample, x_dim))
    # y_train = np.random.randint(n_classes - 1, size=n_sample)  # 0,1,2,3 four classes

    model(x_train_real, y_train_real, n_classes, model_lr=0.01, epochs=100, batch=16
    # print(label.shape)
    # model(X, label, n_classes, model_lr=0.01, epochs=1, batch=1
        ,given_w=w, given_b=b, save_params=True)
