import numpy as np
from read_dataset import read_data

# define input symbols
X, y_array = read_data('./dataset/dataset.csv')

# define variable symbols
W = [np.random.rand(), np.random.rand()]
b = np.random.rand()

lose_rate = 0.01
min_error_threshold = 0.5
train = int(3 / 4 * len(X))
test = len(X) - train


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x, w, b):
    wx_plus_b = w * x + b
    return x * sigmoid(wx_plus_b) * (1 - sigmoid(wx_plus_b))


def calculate_y(W, X, b):
    w2 = np.array(W)
    x2 = np.array(X)
    w2.reshape([2, 1])
    np.matmul(x2, w2) + b
    return np.matmul(X, W) + b


def calculate_cost(y, y_prim):
    return 1 / 2 * ((y - y_prim) ** 2)


def one_layer_nn_training(n_epoch):
    for i in range(n_epoch):
        grad = np.zeros([len(W)])
        for r in range(len(W)):
            for j in range(train):
                y = calculate_y(W, X[j], b)
                cost = calculate_cost(y_array[j], y)
                # print('cost for each y[i] :')
                # print(cost)
                # calculate d_cost/dw using derivative function
                d_cost_dw = (y - y_array[j]) * derivative_sigmoid(W[r], X[j][r], b)
                grad[r] += d_cost_dw
            W[r] = W[r] - lose_rate * grad[r]
    positive = 0
    lost = 0
    all_best_y = []
    for i in range(train, train + test):
        y = calculate_y(W, X[i], b)
        all_best_y.append(y)
        if np.abs(y - y_array[i]) > min_error_threshold:
            lost += 1
        else:
            positive += 1
    # rate = lose_rate / (lose_rate + positive) * 100
    return lost, all_best_y
