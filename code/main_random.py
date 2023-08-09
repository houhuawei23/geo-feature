import numpy as np
import tensorflow as tf
import bone_dataset_remaug as dra
import dataset_visual as dv
import data_load as dl
import network as fnn
from multiprocessing.pool import Pool
import os
from PIL import Image
import imageio
from sklearn.model_selection import train_test_split
from multiprocessing import Manager
#import RNNnet_mnist as rnn


# dv.make_mnist(True)

def label_parallel(info_list, unit_hop, ratio, percentage, l):
    dataset = info_list[0]
    ave_egr = info_list[1]
    path_dict = info_list[2]
    gdist = info_list[4]
    path_index = info_list[5]
    weight = info_list[6]
    print(l)
    rsub_data, rem_data = dra.whole_remove(dataset, ave_egr, path_dict, gdist, path_index, weight, unit_hop, ratio,
                                           percentage)
    # new_data_dict [l] = rsub_data
    print('l-shape', l, np.shape(rsub_data))
    dv.class_write(rsub_data, 'mnist_rem', l, 28, 28)


def floatrange(start, stop, steps):
    return [start + float(i) * (stop - start) / (float(steps) - 1) for i in range(steps)]


def hyper_computation_parl(k, shared_dict):
    p = Pool(10)
    for l in range(10):
        print(l)
        p.apply_async(hp.hyper_computation, args=(k, l, shared_dict))
    # print("p.close")
    p.close()
    p.join()
    # print("pppp")
    print(len(shared_dict))
    # return shared_dict


def class_load_without_write(ndata_dict):
    X = ndata_dict[0]
    X = X.reshape(-1, 28, 28)
    Y = [0 for i in range(len(ndata_dict[0]))]
    # print("0,", np.shape(X))
    # print(np.shape(X), np.shape(Y))
    for l in range(1, 10):
        X_1 = ndata_dict[l]
        X_1 = X_1.reshape(-1, 28, 28)
        # print(l, np.shape(X_1))
        Y_1 = [l for k in range(len(ndata_dict[l]))]
        X = np.vstack((X, X_1))
        Y = np.append(Y, Y_1)

    # Y = Y.reshape(-1,1)
    print(np.shape(X), np.shape(Y))
    x_train, X_test, y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=42)
    x_train = np.concatenate((x_train, X_test), axis=0)
    y_train = np.concatenate((y_train, Y_test), axis=0)
    return x_train, y_train

def load_mnist_random(path = '', percentage = 1):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if os.path.exists(path + "image_list.txt"):
        os.remove(path + "image_list.txt")
    labellist = os.listdir(path)
    labellist = [int(x) for x in labellist]
    #print(labellist)
    X = []
    Y = []
    for label in labellist:
        X1, Y1 = dl.img_load(path, label)
        x_train, X_test, y_train, Y_test = train_test_split(X1, Y1, test_size=1-percentage, random_state=42)
        X.append(x_train)
        Y.append(y_train)

    x_train = X[0]
    y_train = Y[0]
    for i in range(len(X)-1):
        x_train = np.concatenate((x_train, X[i + 1]), axis = 0)
        y_train = np.concatenate((y_train, Y[i + 1]), axis = 0)
    x_train, X_test, y_train, Y_test = train_test_split(x_train, y_train, test_size = 0.01, random_state = 42)
    x_train = np.concatenate((x_train, X_test), axis = 0)
    y_train = np.concatenate((y_train, Y_test), axis = 0)

    return (x_train, y_train), (x_test, y_test)

def load_fashion_mnist_random(path = '', percentage = 1):
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    if os.path.exists(path + "image_list.txt"):
        os.remove(path + "image_list.txt")
    labellist = os.listdir(path)
    labellist = [int(x) for x in labellist]
    #print(labellist)
    X = []
    Y = []
    for label in labellist:
        X1, Y1 = dl.img_load(path, label)
        x_train, X_test, y_train, Y_test = train_test_split(X1, Y1, test_size=1-percentage, random_state=42)
        X.append(x_train)
        Y.append(y_train)

    x_train = X[0]
    y_train = Y[0]
    for i in range(len(X)-1):
        x_train = np.concatenate((x_train, X[i + 1]), axis = 0)
        y_train = np.concatenate((y_train, Y[i + 1]), axis = 0)
    x_train, X_test, y_train, Y_test = train_test_split(x_train, y_train, test_size = 0.01, random_state = 42)
    x_train = np.concatenate((x_train, X_test), axis = 0)
    y_train = np.concatenate((y_train, Y_test), axis = 0)

    return (x_train, y_train), (x_test, y_test)

def main():
    #dv.make_mnist(True)
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist_random(path='../datas/fashion_mnist/train/', percentage = 0.7)
    print(np.shape(x_train))
    print(np.shape(y_train))
    eval_value_n = fnn.mnist_fnn(x_train, y_train, x_test, y_test)
    print("Rhyperparmater:knn-unit_hop-ratio-percentage-result-result(modified)", eval_value_n)
    eval_value_n = fnn.mnist_rnn(x_train, y_train, x_test, y_test)
    print("Rhyperparmater:knn-unit_hop-ratio-percentage-result-result(modified)", eval_value_n)
    eval_value_n = fnn.mnist_cnn(x_train, y_train, x_test, y_test)
    print("Rhyperparmater:knn-unit_hop-ratio-percentage-result-result(modified)", eval_value_n)




if __name__ == "__main__":
    main()