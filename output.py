import numpy as np


class Output:
    def __init__(self, in_dim, out_dim, name):
        self.w = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / out_dim)
        self.b = np.zeros(out_dim)
        self.dw = np.zeros((in_dim, out_dim))
        self.db = np.zeros(out_dim)
        self.name = name

    def _softmax(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def _delta_cross_entropy(self, X, y):
        m = y.shape[0]
        grad = self._softmax(X)
        grad[range(m), y] -= 1
        grad = grad / m
        return grad

    def _fc_dx(self, dx_wrt_output):
        return dx_wrt_output.dot(self.w.T)

    def load_params(self, w, b):
        assert self.w.shape[0] == w.shape[0]
        assert self.w.shape[1] == w.shape[1]
        assert self.b.shape[0] == b.shape[0]
        self.w = w
        self.b = b

    def cross_en(self, y):
        d = y.shape[0]
        return -np.log(self.out_props[range(d), y]).mean()

    def fwd(self, x):
        self.in_data = x.copy()
        self.fc_out = x.dot(self.w) + self.b
        # print("out fc out", self.fc_out)
        self.out_props = self._softmax(self.fc_out)
        # print("output props", self.out_props)
        return self.out_props

    def bwd(self, y):
        dx_wrt_fc_out = self._delta_cross_entropy(self.fc_out, y)
        print("delta cross/output fc out grad", dx_wrt_fc_out)
        self.dw = self.in_data.T.dot(dx_wrt_fc_out)
        self.db = dx_wrt_fc_out.sum(axis=0)
        return self._fc_dx(dx_wrt_fc_out)

    def upd(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def save_params(self):
        print("dw dtype", self.dw.dtype)
        np.savetxt("./save_params/"+self.name+"_w.csv", self.dw, delimiter=',')
        np.savetxt("./save_params/"+self.name+"_b.csv", [self.db], delimiter=',')
