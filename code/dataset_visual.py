import tensorflow as tf
import tensorflow.keras as keras
import os
from PIL import Image
import imageio
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn import manifold
import shutil
import data_load as dl
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import axes3d


def make_mnist(flag = False):
    # load_data
    (x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
    label = np.unique(y)
    num = np.zeros(10)
    if(flag == True):
        for i in label:
            if os.path.exists(f"../dataset/mnist/train/{i}"):#如果路径存在则删除
                shutil.rmtree(f"../dataset/mnist/train/{i}")
            os.makedirs(f"../dataset/mnist/train/{i}")

        with open("../dataset/mnist/train/image_list.txt", 'w') as img_list:
            i = 0
            for img, label in zip(x, y):
                if (num[label] >= 10000):
                    continue
                else:
                    img = Image.fromarray(img) # 将array转化成图片
                    img_save_path = f"../dataset/mnist/train/{label}/{i}.jpg" # 图片保存路径
                    img.save(img_save_path) # 保存图片
                    img_list.write(img_save_path + "\t" + str(label) + "\n")
                    i += 1
                    num[label] = num[label] + 1

        #os.remove("../dataset/mnist/train/image_list.txt")
    print("finished!")
    return True



def make_fashion_mnist(flag = False):
    # load_data
    (x, y), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    label = np.unique(y)
    num = np.zeros(10)
    print(label)
    print("fasion_mnist")
    if(flag == True):
        for i in label:
            if os.path.exists(f"../dataset/fashion_mnist/train/{i}"):  # 如果路径存在则删除
                shutil.rmtree(f"../dataset/fashion_mnist/train/{i}")
            os.makedirs(f"../dataset/fashion_mnist/train/{i}")

        with open("../dataset/fashion_mnist/train/image_list.txt", 'w') as img_list:
            i = 0
            for img, label in zip(x, y):
                if (num[label] >= 1000):
                    continue
                else:
                    img = Image.fromarray(img) # 将array转化成图片
                    img_save_path = f"../dataset/fashion_mnist/train/{label}/{i}.jpg" # 图片保存路径
                    img.save(img_save_path) # 保存图片
                    img_list.write(img_save_path + "\t" + str(label) + "\n")
                    i += 1
                    num[label] = num[label] + 1
        #os.remove("../dataset/fashion_mnist/train/image_list.txt")
    return True
def class_read(name, k):
    cur_dir = os.getcwd() # /home/hhw/geo_feature
    # print(cur_dir)
    imgpath = os.path.join(cur_dir, f"dataset/{name}/train/"+ str(k))
    # imgpath = f"../dataset/{name}/train/"+ str(k)
    imgfiles = os.listdir(imgpath)
    
    imgs1=[]
    for file in imgfiles:
            img = imageio.imread(imgpath + "/" + file)
            img = img.flatten()
            imgs1.append(img / 255)
    
    imgs1 = np.array(imgs1,dtype=float)
    return imgs1

#print(class_read(0).shape)

def class_write(dataset, name, label, size1, size2):
    if os.path.exists(f"../datas_modified/{name}/train/{label}"):#如果路径存在则删除
        shutil.rmtree(f"../datas_modified/{name}/train/{label}")
    os.makedirs(f"../datas_modified/{name}/train/{label}")

    for i in range(np.shape(dataset)[0]):
        img = dataset[i]
        img = img * 255
        img = img.reshape(size1, size2)
        img = Image.fromarray(img)
        #print(img)
        path = f"../datas_modified/{name}/train/{label}/{i}.png" # 图片保存路径
        img.convert('P').save(path)
    return True

"""
#Test
dataset = np.array([[0.5,0.6, 0.1, 0.4, 0.5, 0.9, 0.2, 0.1, 0.8],[0.5,0.6, 0.1, 0.4, 0.5, 0.9, 0.2, 0.1, 0.8]])
size1 = 3
size2 = 3
class_write(dataset, 0, size1, size2)
"""


def umap_visual(k):
    data = class_read(k)
    num, _, _ = np.shape(data) 
    data = data.reshape(num,784)
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(data)
    print(embedding.shape)
    plt.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    #plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('UMAP projection of the Digits dataset')
    plt.show()
    return True

#print(umap_visual(0))

def t_sne_visual(k):
    data = class_read(k)
    num, _, _ = np.shape(data) 
    data = data.reshape(num,784)
    reducer = manifold.TSNE(n_components=3)
    embedding = reducer.fit_transform(data)
    print(embedding.shape)

    plt.figure("3D Scatter", facecolor="lightgray")
    ax3d = plt.subplot(projection="3d")  # 创建三维坐标
    plt.title('3D Scatter', fontsize=20)
    ax3d.set_xlabel('x', fontsize=14)
    ax3d.set_ylabel('y', fontsize=14)
    ax3d.set_zlabel('z', fontsize=14)
    plt.tick_params(labelsize=10)
    ax3d.scatter(embedding[:,0],embedding[:,1], embedding[:,2], s=3, cmap="jet", marker="o")
    plt.show()
    return True

#print(t_sne_visual(0))

#make_mnist(flag=True)