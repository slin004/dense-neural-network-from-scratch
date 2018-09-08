import numpy as np


def load_datasets():
    x_train = np.genfromtxt("datasets/x_train.csv", delimiter=',')
    x_test = np.genfromtxt("datasets/x_test.csv", delimiter=',')
    y_train = np.genfromtxt("datasets/y_train.csv", delimiter=',', dtype='int')
    y_test = np.genfromtxt("datasets/y_test.csv", delimiter=',', dtype='int')
    return x_train, x_test, y_train, y_test


def load_100_40_4_params():
    b = list()
    b.append(np.genfromtxt("given_params/b-100-40-4.csv", delimiter=',', max_rows=1)[1:])
    b.append(np.genfromtxt("given_params/b-100-40-4.csv", delimiter=',', skip_header=1, max_rows=1)[1:])
    b.append(np.genfromtxt("given_params/b-100-40-4.csv", delimiter=',', skip_header=2, max_rows=1)[1:])
    w = list()
    w.append(
        np.delete(
            np.genfromtxt("given_params/w-100-40-4.csv", delimiter=',', max_rows=14),
            0, 1))
    w.append(
        np.delete(
            np.genfromtxt("given_params/w-100-40-4.csv", delimiter=',', skip_header=14, max_rows=100),
            0, 1))
    w.append(
        np.delete(
            np.genfromtxt("given_params/w-100-40-4.csv", delimiter=',', skip_header=114, max_rows=40),
            0, 1))
    return w, b


if __name__ == "__main__":
    # x_train, x_test, y_train, y_test = load_datasets()
    # print(x_train[:10])
    # print(y_train[:10])

    w, b = load_100_40_4_params()
    print(b)
    print(w)
    # w, b = load_14_28_4_params()
    # w, b = load_28_6_4_params()
