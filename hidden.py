import numpy as np
class Hidden:
    def __init__(self, in_dim, out_dim):
        self.w = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / out_dim)
        self.b = np.zeros(out_dim)
        self.dw = np.zeros((in_dim, out_dim))
        self.db = np.zeros(out_dim)

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_dx(self, dx_wrt_output):
        dx_wrt_input = dx_wrt_output.copy()
        dx_wrt_input[dx_wrt_output <= 0] = 0
        return dx_wrt_input

    def _fc_dx(self, dx_wrt_output):
        return dx_wrt_output.dot(self.w.T)

    def load_params(self, w, b):
        self.w = w
        self.b = b

    def fwd(self, x):
        self.in_data = x.copy()
        fc_out = x.dot(self.w) + self.b
        # print("x", x, "w", self.w, "b", self.b, "fc1out", fc_out)
        return self._relu(fc_out)

    def bwd(self, dx_wrt_output):
        dx_wrt_fc_out = self._relu_dx(dx_wrt_output)
        # print("hidden fc output grad", dx_wrt_fc_out)
        self.dw = self.in_data.T.dot(dx_wrt_fc_out)
        self.db = dx_wrt_fc_out.sum(axis=0)
        return self._fc_dx(dx_wrt_fc_out)

    def upd(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db
