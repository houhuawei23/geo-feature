import dataset_visual as dv
import numpy as np
import tensorflow as tf
import bone_dataset_remaug as dra
import dataset_visual as dv
import data_load as dl
from multiprocessing.pool import Pool
import os
from PIL import Image
import imageio
from sklearn.model_selection import train_test_split
from multiprocessing import Manager
import shutil

#dv.make_mnist(True)
#dv.make_fashion_mnist(True)


def make_rem_mnist(percentage):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x, X_test, y, Y_test = train_test_split(x_train, y_train, test_size=1-percentage, random_state=42)
    print(np.shape(x))
    label = np.unique(y)
    flag = True
    num = np.zeros(10)
    if(flag == True):
        for i in label:
            if os.path.exists(f"../datas/mnist/train/{i}"):  # 如果路径存在则删除
                shutil.rmtree(f"../datas/mnist/train/{i}")
            os.makedirs(f"../datas/mnist/train/{i}")

        with open("../datas/mnist/train/image_list.txt", 'w') as img_list:
            i = 0
            for img, label in zip(x, y):
                img = Image.fromarray(img) # 将array转化成图片
                img_save_path = f"../datas/mnist/train/{label}/{i}.jpg" # 图片保存路径
                img.save(img_save_path) # 保存图片
                img_list.write(img_save_path + "\t" + str(label) + "\n")
                i += 1
                num[label] = num[label] + 1
        #os.remove("../datas/mnist/train/image_list.txt")
        print(num)
    return True


def make_rem_fashion_mnist(percentage):
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x, X_test, y, Y_test = train_test_split(x_train, y_train, test_size=1-percentage, random_state=42)
    label = np.unique(y)
    num = np.zeros(10)
    flag = True
    if(flag == True):
        for i in label:
            if os.path.exists(f"../datas/fashion_mnist/train/{i}"):  # 如果路径存在则删除
                shutil.rmtree(f"../datas/fashion_mnist/train/{i}")
            os.makedirs(f"../datas/fashion_mnist/train/{i}")

        with open("../datas/fashion_mnist/train/image_list.txt", 'w') as img_list:
            i = 0
            for img, label in zip(x, y):
                img = Image.fromarray(img) # 将array转化成图片
                img_save_path = f"../datas/fashion_mnist/train/{label}/{i}.jpg" # 图片保存路径
                img.save(img_save_path) # 保存图片
                img_list.write(img_save_path + "\t" + str(label) + "\n")
                i += 1
        #os.remove("../datas/fashion_mnist/train/image_list.txt")
    return True


#make_rem_mnist(0.6)
make_rem_fashion_mnist(0.6)
#dv.make_mnist(True)
