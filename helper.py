import numpy as np


def load_datasets():
    x_train = np.genfromtxt("datasets/x_train.csv", delimiter=',')
    x_test = np.genfromtxt("datasets/x_test.csv", delimiter=',')
    y_train = np.genfromtxt("datasets/y_train.csv", delimiter=',', dtype='int')
    y_test = np.genfromtxt("datasets/y_test.csv", delimiter=',', dtype='int')
    return x_train, x_test, y_train, y_test


def load_14_28_params():
    b = list()
    w = list()
    network_design = [14, ] # input
    for i in range(28):
        network_design.append(14) # 28 ge 14 layers
    network_design.append(4)
    i = 1
    row_loaded = 0
    while i < len(network_design):
        b.append(np.genfromtxt("given_params/b-14-28-4.csv", delimiter=',', skip_header=i - 1, max_rows=1)[1:])
        w.append(
            np.delete(
                np.genfromtxt("given_params/w-14-28-4.csv", delimiter=',',
                              skip_header=row_loaded,
                              max_rows=network_design[i - 1]), 0, 1))
        row_loaded += network_design[i - 1]
        i += 1

    for i in range(len(b)):
        print(i, w[i].shape, b[i].shape)

    return w, b


def load_28_6_params():
    b = list()
    w = list()
    network_design = [14, 28, 28, 28, 28, 28, 28, 4]
    i = 1
    row_loaded = 0
    while i < len(network_design):
        b.append(np.genfromtxt("given_params/b-28-6-4.csv", delimiter=',', skip_header=i - 1, max_rows=1)[1:])
        w.append(
            np.delete(
                np.genfromtxt("given_params/w-28-6-4.csv", delimiter=',',
                              skip_header=row_loaded,
                              max_rows=network_design[i - 1]), 0, 1))
        row_loaded += network_design[i - 1]
        i += 1

    for i in range(len(b)):
        print(w[i].shape, b[i].shape)

    return w, b


def load_100_40_4_params_new():
    b = list()
    w = list()
    network_design = [14, 100, 40, 4]
    i = 1
    row_loaded = 0
    while i < len(network_design):
        b.append(np.genfromtxt("given_params/b-100-40-4.csv", delimiter=',', skip_header=i - 1, max_rows=1)[1:])
        w.append(
            np.delete(
                np.genfromtxt("given_params/w-100-40-4.csv", delimiter=',',
                              skip_header=row_loaded,
                              max_rows=network_design[i - 1]), 0, 1))
        row_loaded += network_design[i - 1]
        i += 1

    for i in range(len(b)):
        print(w[i].shape, b[i].shape)

    return w, b


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


def save_params_to_csv(dw, db, model_name):
    with open("./save_params/dw-" + model_name + ".csv", 'ab') as csv_file:
        for param in dw:
            np.savetxt(csv_file, param, delimiter=',')
    with open("./save_params/db-" + model_name + ".csv", 'ab') as csv_file:
        for param in db:
            np.savetxt(csv_file, [param], delimiter=',')


if __name__ == "__main__":
    # x_train, x_test, y_train, y_test = load_datasets()
    # print(x_train[:10])
    # print(y_train[:10])

    # w, b = load_100_40_4_params_new()
    # w, b = load_100_40_4_params()
    # w, b = load_28_6_params()
    w, b = load_14_28_params()
    # print(b)
    # print(w)
    # w, b = load_14_28_4_params()
    # w, b = load_28_6_4_params()
