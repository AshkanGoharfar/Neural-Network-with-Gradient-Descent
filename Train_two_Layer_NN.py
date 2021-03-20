import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from read_dataset import read_data
from operator import itemgetter


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


# define input symbols
X, y_array = read_data('./dataset/dataset.csv')

# define variable symbols
W = [np.random.rand(), np.random.rand()]
b = np.random.rand()

n_epoch = 20
# lr = 0.01
train = int(3 / 4 * len(X))
test = len(X) - train


def calculate_y0(X, W, V, U, B):
    W_prime = np.array(W)
    U_prime = np.array(U)
    V_prime = np.array(V)
    W_prime.reshape([2, 1])
    V_prime.reshape([2, 1])
    Z = np.array([sigmoid(np.matmul(X, W_prime) + B[0]), sigmoid(np.matmul(X, U_prime) + B[1])])
    U_prime.reshape([2, 1])
    return sigmoid(np.matmul(Z, U_prime) + B[2])


W = [np.random.rand(), np.random.rand()]
V = [np.random.rand(), np.random.rand()]
U = [np.random.rand(), np.random.rand()]
B = [np.random.rand(), np.random.rand(), np.random.rand()]


def two_layer_nn_training(lr):
    for i in range(n_epoch):
        grad_w = np.zeros([len(W)])
        grad_v = np.zeros([len(V)])
        grad_u = np.zeros([len(U)])
        for r1 in range(len(U)):
            for r2 in range(len(W)):
                for r3 in range(len(V)):
                    for j in range(train):
                        y0 = calculate_y0(X[j], W, V, U, B)
                        cost = calculate_cost(y_array[j], y0)
                        # calculate_y is also used to calculate WX+b0
                        dcost_du = (y0 - y_array[j]) * derivative_sigmoid(calculate_y(W, X[j], B[0]), U[r1], B[0])

                        grad_u[r1] += dcost_du / 3
                        dcost_dw = (y0 - y_array[j]) * derivative_sigmoid(calculate_y(W, X[j], B[0]), U[r1],
                                                                          B[0]) * derivative_sigmoid(X[j][r2], W[r2],
                                                                                                     B[0])
                        grad_w[r2] += dcost_dw / 3
                        dcost_dv = (y0 - y_array[j]) * derivative_sigmoid(calculate_y(V, X[j], B[0]), U[r1],
                                                                          B[0]) * derivative_sigmoid(X[j][r3], V[r3],
                                                                                                     B[1])
                        grad_v[r3] += dcost_dv / 3
        for r1 in range(len(U)):
            U[r1] = U[r1] - lr * grad_u[r1]
        for r1 in range(len(V)):
            V[r1] = V[r1] - lr * grad_v[r1]
        for r1 in range(len(W)):
            W[r1] = W[r1] - lr * grad_w[r1]
    lost = 0
    y_res = []
    for i in range(train, train + test):
        y = calculate_y0(X[i], W, V, U, B)
        y_res.append(y)
        lost += 1 if np.abs(y - y_array[i]) > 0.5 else 0
    return lost, y_res


epoch_array = [i * 0.001 for i in range(50)]
all_lost = []
all_answer = []
for item in epoch_array:
    lost, y_result = two_layer_nn_training(item)
    all_lost.append(lost)
    all_answer.append([y_result, lost])

best_y = sorted(all_answer, key=itemgetter(1), reverse=False)[0]
input_y = list(y_array[0:train]) + best_y[0]
all_clusters = []
each_cluster = []
for i in range(len(input_y)):
    if input_y[i] not in each_cluster:
        if input_y[i] > 0:
            input_y[i] = 1
        elif np.absolute(input_y[i] - 1) == 0:
            input_y[i] = 0
        else:
            input_y[i] = 0
        if input_y[i] not in each_cluster:
            each_cluster.append(input_y[i])
for i in range(len(each_cluster)):
    cluster = []
    for j in range(len(X)):
        if input_y[j] == each_cluster[i]:
            cluster.append(X[j])
    all_clusters.append(cluster)
fig, ax = plt.subplots()
rgb_colors = ['brown', 'red', 'orange', 'purple', 'gray', 'yellow', 'black', 'green', 'blue']
counter = 0
for cluster in all_clusters:
    for node in cluster:
        plt.scatter(node[0], node[1], color=rgb_colors[counter])
    counter += 1
ax.set(xlabel='X', ylabel='Y', title='Two Layer NN Trained data with different categories')
ax.grid()
fig.savefig('Two Layer NN Trained data categories')
plt.show()

# result = [two_layer_nn_training(item) for item in epoch_array]
fig, ax = plt.subplots()
ax.plot(epoch_array, all_lost)
ax.set(xlabel='X', ylabel='Y', title='Two Layer NN Trained data with different categories')
ax.grid()
fig.savefig('Two Layer NN Trained data categories')
plt.show()
print(np.mean(all_lost))
