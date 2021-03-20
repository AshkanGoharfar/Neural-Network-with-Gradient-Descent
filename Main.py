import matplotlib.pyplot as plt
import numpy as np
from Train_one_Layer_NN import one_layer_nn_training
from read_dataset import plot_inputs

if __name__ == '__main__':
    plot_inputs('./dataset/dataset.csv')
    epoch_array = [i * 3 for i in range(100)]
    best_lose_rate = 0.1
    lose_rate = []
    flag_best_answer = 0
    counter = 0
    y_best = []
    rate_best = 0
    sequential_lose = 0
    while flag_best_answer == 0 and counter < len(epoch_array):
        rate, y_test = one_layer_nn_training(epoch_array[counter])
        lose_rate.append(rate)
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
    print(np.mean(lose_rate))
