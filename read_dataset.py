import random
import matplotlib.pyplot as plt


def read_data(path):
    f = open(path)
    X = []
    y_arr = []
    counter = 0
    for l in f:
        if counter != 0:
            l = l.split('\n')
            del l[-1]
            l = l[0].split(',')
            l[2] = float(l[2])
            X.append([float(l[0]), float(l[1])])
            y_arr.append(int(l[2]))
        counter += 1
    z = list(zip(X, y_arr))
    random.shuffle(z)
    X, y_arr = zip(*z)
    return X, y_arr


def plot_inputs(path):
    input_x, input_y = read_data(path)
    all_clusters = []
    each_cluster = []
    for i in range(len(input_y)):
        if input_y[i] not in each_cluster:
            each_cluster.append(input_y[i])
    for i in range(len(each_cluster)):
        cluster = []
        for j in range(len(input_x)):
            if input_y[j] == each_cluster[i]:
                cluster.append(input_x[j])
        all_clusters.append(cluster)
    fig, ax = plt.subplots()
    rgb_colors = ['brown', 'red', 'orange', 'purple', 'gray', 'yellow', 'black', 'green', 'blue']
    counter = 0
    for cluster in all_clusters:
        for node in cluster:
            plt.scatter(node[0], node[1], color=rgb_colors[counter])
        counter += 1
    ax.set(xlabel='X', ylabel='Y', title='Input data with different categories')
    ax.grid()
    fig.savefig('Input data categories')
    plt.show()
    return 0
