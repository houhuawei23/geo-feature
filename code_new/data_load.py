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

curfile_dir = os.path.dirname(os.path.abspath(__file__))
datas_dir = os.path.join(curfile_dir, '..', 'dataset')
def img_load(path, label):
    X = []
    Y = []
    imgfiles = os.listdir(os.path.join(path, str(label)))
    for file in imgfiles:
        img = imageio.imread(os.path.join(path, str(label), file))
        X.append(img)
        Y.append(label)

    X = np.array(X, dtype=int)
    Y = np.array(Y, dtype=int)
    return X, Y

def load_dataset(dataset_name, local = True, path = ''):
    if (local == False):
        try:
            if (dataset_name == 'mnist'):
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            elif (dataset_name == 'fashion_mnist'):
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            else:
                print("Not in online dataset list")
                return False
        except:
            print("Error in loading dataset", dataset_name)
            return False
        return (x_train, y_train), (x_test, y_test)
    
    # else:
    try:
        dataset_dir = os.path.join(datas_dir, dataset_name, 'train')
        classes = os.listdir(dataset_dir)        
        # Initialize empty lists to store the image data and labels
        data = []
        labels = []

        # Loop over each class directory
        for class_dir in classes:
            class_dir_path = os.path.join(dataset_dir, class_dir)
            
            # Get a list of all the image filenames
            image_filenames = os.listdir(class_dir_path)

            # Loop over the images
            for filename in image_filenames:
                # Load the image and convert it to grayscale
                image_path = os.path.join(class_dir_path, filename)
                image = Image.open(image_path).convert('L')

                # Convert the image data to a numpy array and normalize it
                image_data = np.array(image) / 255.0

                # Append the image data and label to the lists
                data.append(image_data)
                labels.append(int(class_dir))

        # Convert the lists to numpy arrays
        data = np.array(data)
        labels = np.array(labels)

        # Use train_test_split to split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)
    except:
        print("Error in loading local dataset", dataset_name)
        return False
    return (x_train, y_train), (x_test, y_test)



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