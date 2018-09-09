import numpy as np
from layer import Output, Hidden
from copy import deepcopy
from helper import load_datasets, load_100_40_4_params, save_params_to_csv

np.random.seed(0)


class Model:
    def __init__(self, layers_config, model_name=""):
        self.hidden_layers = list()
        for layer_config in layers_config[:-1]:
            self.hidden_layers.append(Hidden(layer_config['in_dim'],
                                             layer_config['out_dim'],
                                             name=layer_config['name']))
        self.layer_o = Output(layers_config[-1]['in_dim'],
                              layers_config[-1]['out_dim'],
                              layers_config[-1]['name'])
        self.model_name = model_name

    def load_given_w_b(self, given_w, given_b):
        # optional load w b
        if given_w and given_b:
            for i, layer in enumerate(self.hidden_layers):
                layer.load_params(given_w[i], given_b[i])
            self.layer_o.load_params(given_w[-1], given_b[-1])

    def train(self, m_x_train, m_y_train, batch=1, epochs=1, model_lr=0.1):
        loss_count = 0
        total_loss = 0
        infer_correct = 0

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
                total_loss += self.layer_o.cross_en(y_batch)
                loss_count += 1
                # print("loss - ",layer_o.cross_en(y_batch))

                # backward
                grad_for_hidden = self.layer_o.bwd(y_batch)
                # print(grad_for_hidden.shape)
                # print(grad_for_hidden)

                self.layer_o.upd(model_lr)
                for layer in reversed(self.hidden_layers):
                    grad_for_hidden = layer.bwd(grad_for_hidden)
                    layer.upd(model_lr)

            # metrics per epoch
            print(epoch, " - epoch train: loss - ", (total_loss / loss_count), " accuracy - ",
                  (infer_correct / len(m_x_train)))
            loss_count = 0
            total_loss = 0
            infer_correct = 0

    def do_save_params(self):
        dw_to_save = list()
        db_to_save = list()
        for i, layer in enumerate(self.hidden_layers):
            dw_to_save.append(layer.dw)
            db_to_save.append(layer.db)
        dw_to_save.append(self.layer_o.dw)
        db_to_save.append(self.layer_o.db)
        save_params_to_csv(dw_to_save, db_to_save, self.model_name)


def do_net1(verification=False):
    net1_layer_config = [
        {"name": "14x100", "in_dim": 14, "out_dim": 100},
        {"name": "100x40", "in_dim": 100, "out_dim": 40},
        {"name": "40x4", "in_dim": 40, "out_dim": 4},
    ]
    net1 = Model(net1_layer_config, model_name="100-40-4")
    w, b = load_100_40_4_params()
    net1.load_given_w_b(w, b)

    if verification:
        X = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]])
        label = np.array([3])
        net1.train(X, label, batch=1, epochs=1, model_lr=0.1)
        net1.do_save_params()
    else:
        x_train_real, x_test, y_train_real, y_test = load_datasets()
        net1.train(x_train_real, y_train_real, batch=32, epochs=300, model_lr=0.01)


if __name__ == "__main__":
    np.random.seed(0)
    do_net1()
