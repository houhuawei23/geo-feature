import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import datetime
import tensorflow as tf
import os
from PIL import Image
import imageio
from sklearn.model_selection import train_test_split


def img_load(path, label):
    X = []
    Y = []
    imgfiles = os.listdir(path + "/"+ str(label))
    for file in imgfiles:
        img = imageio.imread(path + "/"+ str(label) + '/' + file)
        X.append(img)
        Y.append(label)

    X = np.array(X, dtype=int)
    Y = np.array(Y, dtype=int)
    return X, Y

# flag == False : original data
# flag == True : local data
def load_mnist(flag = False, path = ''):
    if (flag == False):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    else:
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
            X1, Y1 = img_load(path, label)
            X.append(X1)
            Y.append(Y1)

        x_train = X[0]
        y_train = Y[0]
        for i in range(len(X)-1):
            x_train = np.concatenate((x_train, X[i + 1]), axis = 0)
            y_train = np.concatenate((y_train, Y[i + 1]), axis = 0)
        x_train, X_test, y_train, Y_test = train_test_split(x_train, y_train, test_size = 0.01, random_state = 42)
        x_train = np.concatenate((x_train, X_test), axis = 0)
        y_train = np.concatenate((y_train, Y_test), axis = 0)

    return (x_train, y_train), (x_test, y_test)


#path = '../datas/mnist/train/'
#(a,b), (c,d) = load_mnist(True, path)
#print(np.shape(a), np.shape(b))


def load_fashion_mnist(flag = False, path = ''):
    if (flag == False):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        if os.path.exists(path + "image_list.txt"):
            os.remove(path + "image_list.txt")
        labellist = os.listdir(path)
        labellist = [int(x) for x in labellist]
        print(labellist)
        X = []
        Y = []
        for label in labellist:
            X1, Y1 = img_load(path, label)
            X.append(X1)
            Y.append(Y1)

        x_train = X[0]
        y_train = Y[0]
        for i in range(len(X)-1):
            x_train = np.concatenate((x_train, X[i + 1]), axis = 0)
            y_train = np.concatenate((y_train, Y[i + 1]), axis = 0)
        x_train, X_test, y_train, Y_test = train_test_split(x_train, y_train, test_size=0.01, random_state=42)
        x_train = np.concatenate((x_train, X_test), axis=0)
        y_train = np.concatenate((y_train, Y_test), axis=0)

    return (x_train, y_train), (x_test, y_test)