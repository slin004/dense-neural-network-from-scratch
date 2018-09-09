import numpy as np
import matplotlib.pyplot as plt


def load_datasets():
    x_train = np.genfromtxt("datasets/x_train.csv", delimiter=',')
    x_test = np.genfromtxt("datasets/x_test.csv", delimiter=',')
    y_train = np.genfromtxt("datasets/y_train.csv", delimiter=',', dtype='int')
    y_test = np.genfromtxt("datasets/y_test.csv", delimiter=',', dtype='int')
    return x_train, x_test, y_train, y_test


def load_14_28_params(dir):
    b = list()
    w = list()
    network_design = [14, ]  # input
    for i in range(28):
        network_design.append(14)  # 28 ge 14 layers
    network_design.append(4)
    i = 1
    row_loaded = 0
    while i < len(network_design):
        b.append(np.genfromtxt(dir + "/b-14-28-4.csv", delimiter=',', skip_header=i - 1, max_rows=1)[1:])
        w.append(
            np.delete(
                np.genfromtxt(dir + "/w-14-28-4.csv", delimiter=',',
                              skip_header=row_loaded,
                              max_rows=network_design[i - 1]), 0, 1))
        row_loaded += network_design[i - 1]
        i += 1

    for i in range(len(b)):
        print(i, w[i].shape, b[i].shape)

    return w, b


def load_28_6_params(dir):
    b = list()
    w = list()
    network_design = [14, 28, 28, 28, 28, 28, 28, 4]
    i = 1
    row_loaded = 0
    while i < len(network_design):
        b.append(np.genfromtxt(dir + "/b-28-6-4.csv", delimiter=',', skip_header=i - 1, max_rows=1)[1:])
        w.append(
            np.delete(
                np.genfromtxt(dir + "/w-28-6-4.csv", delimiter=',',
                              skip_header=row_loaded,
                              max_rows=network_design[i - 1]), 0, 1))
        row_loaded += network_design[i - 1]
        i += 1

    for i in range(len(b)):
        print(w[i].shape, b[i].shape)

    return w, b


def load_100_40_4_params_new(dir):
    b = list()
    w = list()
    network_design = [14, 100, 40, 4]
    i = 1
    row_loaded = 0
    while i < len(network_design):
        b.append(np.genfromtxt(dir + "/b-100-40-4.csv", delimiter=',', skip_header=i - 1, max_rows=1)[1:])
        w.append(
            np.delete(
                np.genfromtxt(dir + "/w-100-40-4.csv", delimiter=',',
                              skip_header=row_loaded,
                              max_rows=network_design[i - 1]), 0, 1))
        row_loaded += network_design[i - 1]
        i += 1

    for i in range(len(b)):
        print(w[i].shape, b[i].shape)

    return w, b


def save_params_to_csv(dw, db, model_name, dir):
    with open(dir + "/dw-" + model_name + ".csv", 'w+') as csv_file:
        for param in dw:
            np.savetxt(csv_file, param, delimiter=',')
    with open(dir + "/db-" + model_name + ".csv", 'w+') as csv_file:
        for param in db:
            np.savetxt(csv_file, [param], delimiter=',')


def save_plt(plt_iter, plt_title,
             plt_train_loss, plt_train_acc,
             plt_test_loss, plt_test_acc):
    plt.plot(plt_iter, plt_train_loss, label="train")
    plt.legend()
    plt.ylabel('Cost')
    plt.xlabel('Iteration')
    plt.title(plt_title)
    plt.savefig('save_plots/%s_train_cost.png' % plt_title)
    plt.clf()

    plt.plot(plt_iter, plt_test_loss, label="test")
    plt.legend()
    plt.ylabel('Cost')
    plt.xlabel('Iteration')
    plt.title(plt_title)
    plt.savefig('save_plots/%s_test_cost.png' % plt_title)
    plt.clf()

    plt.plot(plt_iter, plt_train_acc, label="train")
    plt.plot(plt_iter, plt_test_acc, label="test")
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.title(plt_title)
    plt.savefig('save_plots/%s_train_test_accuracy.png' % plt_title)
    plt.clf()


if __name__ == "__main__":
    # x_train, x_test, y_train, y_test = load_datasets()
    # print(x_train[:10])
    # print(y_train[:10])

    # w, b = load_100_40_4_params_new()
    # w, b = load_100_40_4_params()
    # w, b = load_28_6_params()
    # w, b = load_14_28_params()
    # print(b)
    # print(w)
    # w, b = load_14_28_4_params()
    # w, b = load_28_6_4_params()
    # save_plt([1,4,8,9], "a test",
    #     [2,1,4,2], [0.2,0.4,0.2,0.8],
    #     [3,1,9,1], [0.2,0.3,0.3,0.1])
    print("end")