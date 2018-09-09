import numpy as np
from layer import Output, Hidden
from copy import deepcopy
from helper import load_datasets
# from helper import load_100_40_4_params
from helper import load_28_6_params
from helper import load_14_28_params
from helper import load_100_40_4_params_new
from helper import save_params_to_csv
from helper import save_plt

np.random.seed(0)


class Model:
    def __init__(self, layers_config, model_name="", beta=0.):
        self.hidden_layers = list()
        for layer_config in layers_config[:-1]:
            self.hidden_layers.append(Hidden(layer_config['in_dim'],
                                             layer_config['out_dim'],
                                             name=layer_config['name'],
                                             beta=beta))
        self.layer_o = Output(layers_config[-1]['in_dim'],
                              layers_config[-1]['out_dim'],
                              name=layers_config[-1]['name'],
                              beta=beta)
        self.model_name = model_name
        self.beta = beta

    def load_given_w_b(self, given_w, given_b):
        # optional load w b
        if given_w and given_b:
            for i, layer in enumerate(self.hidden_layers):
                layer.load_params(given_w[i], given_b[i])
            self.layer_o.load_params(given_w[-1], given_b[-1])

    def train(self, m_x_train, m_y_train, m_x_test=[], m_y_test=[],
              batch=1, epochs=1, model_lr=0.1, verification=False):
        total_loss = 0
        infer_correct = 0
        num_sample = len(m_x_train)


        # var for plot
        plt_iter = list()
        plt_train_loss = list()
        plt_test_loss = list()
        plt_train_acc = list()
        plt_test_acc = list()
        plt_title = 'Network_%s_Batch_%d' % (self.model_name, batch)
        if self.beta != 0:
            plt_title += 'Beta_%.1f' % self.beta

        for epoch in range(epochs):
            batch_p = 0
            while batch_p < len(m_x_train):
                x_batch = m_x_train[batch_p:batch_p + batch]
                y_batch = m_y_train[batch_p:batch_p + batch]
                batch_p += batch

                # forward
                data_for_hidden = x_batch
                for layer in self.hidden_layers:
                    data_for_hidden = layer.fwd(data_for_hidden)
                sm_out = self.layer_o.fwd(data_for_hidden)

                # metrics
                y_hat = np.argmax(sm_out, axis=1)
                infer_correct += (y_hat == y_batch).sum()
                # print("***debug y_hat ", y_hat, " y ", y_batch, " number of correct", (y_hat == y_batch).sum())
                total_loss += self.layer_o.cross_en(y_batch)
                # print("loss - ",layer_o.cross_en(y_batch))

                # backward
                grad_for_hidden = self.layer_o.bwd(y_batch)
                # print(grad_for_hidden.shape)
                # print(grad_for_hidden)

                self.layer_o.upd(model_lr)
                for layer in reversed(self.hidden_layers):
                    grad_for_hidden = layer.bwd(grad_for_hidden)
                    layer.upd(model_lr)

            # metrics per epoch/iter
            loss = total_loss/num_sample
            accuracy = infer_correct/num_sample
            print("%d train: loss: %f accuracy: %f" % (epoch, loss, accuracy))
            plt_iter.append(epoch)
            plt_train_loss.append(loss)
            plt_train_acc.append(accuracy)

            if len(m_x_test) > 0:
                test_loss, test_acc = self.test(m_x_test, m_y_test)
                plt_test_loss.append(test_loss)
                plt_test_acc.append(test_acc)
                print("%d test: loss: %f accuracy: %f" % (epoch, test_loss, test_acc))

            # stats
            total_loss = 0
            infer_correct = 0

        save_plt(plt_iter, plt_title,
                 plt_train_loss, plt_train_acc,
                 plt_test_loss, plt_test_acc)


    def test(self, m_x_test, m_y_test):
        total_loss = 0
        infer_correct = 0
        batch = 1
        batch_p = 0
        num_sample = len(m_x_test)
        while batch_p < num_sample:
            x_batch = m_x_test[batch_p:batch_p + batch]
            y_batch = m_y_test[batch_p:batch_p + batch]
            batch_p += batch

            # forward
            data_for_hidden = x_batch
            for layer in self.hidden_layers:
                data_for_hidden = layer.fwd(data_for_hidden)
            sm_out = self.layer_o.fwd(data_for_hidden)

            # metrics
            y_hat = np.argmax(sm_out, axis=1)
            infer_correct += (y_hat == y_batch).sum()
            total_loss += self.layer_o.cross_en(y_batch)

        # metrics per epoch
        loss = total_loss/num_sample
        accuracy = infer_correct/num_sample
        return loss, accuracy

    def do_save_params(self, dir):
        dw_to_save = list()
        db_to_save = list()
        for i, layer in enumerate(self.hidden_layers):
            dw_to_save.append(layer.dw)
            db_to_save.append(layer.db)
        dw_to_save.append(self.layer_o.dw)
        db_to_save.append(self.layer_o.db)
        save_params_to_csv(dw_to_save, db_to_save, self.model_name, dir)


def do_net1(verification=False, lr=0.01, batch=1, epoch=1, beta=0.):
    net1_layer_config = [
        {"name": "14x100", "in_dim": 14, "out_dim": 100},
        {"name": "100x40", "in_dim": 100, "out_dim": 40},
        {"name": "40x4", "in_dim": 40, "out_dim": 4},
    ]
    net1 = Model(net1_layer_config, model_name="100-40-4",beta=0.)

    if verification:
        X = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]])
        label = np.array([3])

        w, b = load_100_40_4_params_new('given_params')
        net1.load_given_w_b(w, b)
        net1.train(X, label, batch=1, epochs=1, model_lr=0.1)
        net1.do_save_params('given_params')

        w, b = load_100_40_4_params_new('to_submit_params')
        net1.load_given_w_b(w, b)
        net1.train(X, label, batch=1, epochs=1, model_lr=0.1)
        net1.do_save_params('to_submit_params')
    else:
        x_train, x_test, y_train, y_test = load_datasets()
        net1.train(x_train, y_train, x_test, y_test, batch=batch, epochs=epoch, model_lr=lr)


def do_net2(verification=False, lr=0.01, batch=1, epoch=1, beta=0.):
    layer_config = [
        {"name": "14x28", "in_dim": 14, "out_dim": 28},
        {"name": "28x28", "in_dim": 28, "out_dim": 28},
        {"name": "28x28", "in_dim": 28, "out_dim": 28},
        {"name": "28x28", "in_dim": 28, "out_dim": 28},
        {"name": "28x28", "in_dim": 28, "out_dim": 28},
        {"name": "28x28", "in_dim": 28, "out_dim": 28},
        {"name": "40x4", "in_dim": 28, "out_dim": 4},
    ]
    net = Model(layer_config, model_name="28-6-4", beta=0.9)

    if verification:
        X = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]])
        label = np.array([3])
        w, b = load_28_6_params('given_params')
        net.load_given_w_b(w, b)
        net.train(X, label, batch=1, epochs=1, model_lr=0.1, verification=True)
        net.do_save_params('given_params')

        w, b = load_28_6_params('to_submit_params')
        net.load_given_w_b(w, b)
        net.train(X, label, batch=1, epochs=1, model_lr=0.1, verification=True)
        net.do_save_params('to_submit_params')
    else:
        x_train, x_test, y_train, y_test = load_datasets()
        net.train(x_train, y_train, x_test, y_test, batch=batch, epochs=epoch, model_lr=lr)


def do_net3(verification=False, lr=0.01, batch=1, epoch=1, beta=0.):
    layer_config = list()
    for i in range(28):
        layer_config.append({"name": "14x14", "in_dim": 14, "out_dim": 14})

    layer_config.append({"name": "14x4", "in_dim": 14, "out_dim": 4})

    net = Model(layer_config, model_name="14-28-4", beta=0.9)

    if verification:
        X = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]])
        label = np.array([3])

        w, b = load_14_28_params('given_params')
        net.load_given_w_b(w, b)
        net.train(X, label, batch=1, epochs=1, model_lr=0.1, verification=True)
        net.do_save_params('given_params')

        w, b = load_14_28_params('to_submit_params')
        net.load_given_w_b(w, b)
        net.train(X, label, batch=1, epochs=1, model_lr=0.1, verification=True)
        net.do_save_params('to_submit_params')
    else:
        x_train, x_test, y_train, y_test = load_datasets()
        net.train(x_train, y_train, x_test, y_test, batch=batch, epochs=epoch, model_lr=lr)


if __name__ == "__main__":
    np.random.seed(0)
    # do_net1(verification=True)
    # do_net2(verification=True)
    # do_net3(verification=True)
    do_net1(epoch=1, batch=16)
    # do_net2()
    # do_net3()
