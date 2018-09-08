import numpy as np
from copy import deepcopy


class Layer:
    def __init__(self, in_dim, out_dim, name):
        self.w = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / out_dim)
        self.b = np.zeros(out_dim)
        self.dw = np.zeros((in_dim, out_dim))
        self.db = np.zeros(out_dim)
        self.name = name

    def upd(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def save_params(self):
        np.savetxt("./save_params/" + self.name + "_w.csv", self.dw, delimiter=',')
        np.savetxt("./save_params/" + self.name + "_b.csv", [self.db], delimiter=',')

    def load_params(self, w, b):
        assert self.w.shape[0] == w.shape[0]
        assert self.w.shape[1] == w.shape[1]
        assert self.b.shape[0] == b.shape[0]
        self.w = w
        self.b = b

    def fc_pass_grad(self, grad_wrt_fc_out):
        # print("delta cross/output fc out grad", grad_wrt_fc_out)
        self.dw = np.dot(self.inputs_a.T, grad_wrt_fc_out)
        # print("********debug", self.dw)
        self.db = grad_wrt_fc_out.sum(axis=0)
        # print("delta b", self.db)
        return grad_wrt_fc_out.dot(self.w.T)

    def fwd(self, x):
        self.inputs_a = deepcopy(x)
        self.fc_out = x.dot(self.w) + self.b
        self.act_out = self.activation(self.fc_out)
        return self.act_out


class Output(Layer):

    def activation(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def _delta_cross_entropy(self, X, y):
        m = y.shape[0]
        grad = self.activation(X)
        grad[range(m), y] -= 1
        grad = grad / m
        return grad

    def _delta_cross_entropy2(self, X, y):
        return X - y

    def cross_en(self, y):
        d = y.shape[0]
        return -np.log(self.act_out[range(d), y]).mean()

    def cross_en2(self, y):
        m = y.shape[1]
        loss = y * np.log(self.act_out)
        cost = -np.sum(loss) / m
        cost = np.squeeze(cost)
        assert (cost.shape == ())
        return cost

    def bwd(self, y):
        # print("ouput fc out", self.fc_out)
        grad_wrt_z = self._delta_cross_entropy(self.fc_out, y) # no one hot
        # grad_wrt_z = self._delta_cross_entropy(self.act_out, y) # one hot
        grad_wrt_a = self.fc_pass_grad(grad_wrt_z)
        return grad_wrt_a


class Hidden(Layer):
    def activation(self, x):
        return np.maximum(0, x)

    def activation_pass_grad(self, grad_wrt_act_output):
        grad_wrt_act_inputs = deepcopy(grad_wrt_act_output)
        grad_wrt_act_inputs[self.fc_out <= 0] = 0
        return grad_wrt_act_inputs

    def bwd(self, grad_wrt_h):
        grad_wrt_z = self.activation_pass_grad(grad_wrt_h)
        grad_wrt_a = self.fc_pass_grad(grad_wrt_z)
        return grad_wrt_a
