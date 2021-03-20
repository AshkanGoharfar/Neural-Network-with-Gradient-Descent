import matplotlib.pyplot as plt
import numpy as np
from Train_one_Layer_NN import one_layer_nn_training
from read_dataset import *
from operator import itemgetter

if __name__ == '__main__':
    plot_inputs('./dataset/dataset.csv')

    X, y_array = read_data('./dataset/dataset.csv')

    train_len = int(3 / 4 * len(X))

    epoch_array = [i * 3 for i in range(10)]
    best_lose_rate = 0.1
    lose_rate = []
    flag_best_answer = 0
    counter = 0
    y_best = []
    rate_best = 0
    sequential_lose = 0
    best_y = []
    while flag_best_answer == 0 and counter < len(epoch_array):
        rate, y_test = one_layer_nn_training(epoch_array[counter])
        lose_rate.append(rate)
        best_y.append([y_test, rate])
        if rate < best_lose_rate:
            # print('yeesss : ')
            # print(best_lose_rate)
            sequential_lose += 1
            y_best = y_test
            rate_best = rate
        counter += 1
        # if sequential_lose > 2:
        #     flag_best_answer = 1

    # result = [one_layer_nn_training(item) for item in epoch_array]
    # print(result)
    fig, ax = plt.subplots()
    ax.plot(epoch_array[0: counter], lose_rate)
    ax.set(xlabel='epoch size', ylabel='lost', title='One layer NN result of different epochs with different lose rate')
    ax.grid()
    fig.savefig('One layer NN with epoch(3-300)')
    plt.show()

    best_y = sorted(best_y, key=itemgetter(1), reverse=False)[0]
    input_y = list(y_array[0:train_len]) + best_y[0]
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
            print('Node : ')
            print(node)
            print(rgb_colors[counter])
            plt.scatter(node[0], node[1], color=rgb_colors[counter])
        counter += 1
    ax.set(xlabel='X', ylabel='Y', title='One Layer NN Trained data with different categories')
    ax.grid()
    fig.savefig('One Layer NN Trained data categories')
    plt.show()

    ###################### Two Layer #############33
