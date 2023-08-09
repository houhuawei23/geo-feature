import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import bone_dataset_remaug as dra
import sklearn.datasets as skl
import dataset_visual as dv
from mpl_toolkits.mplot3d import Axes3D



def generate_Swissroll(n):
    t = (3 * np.pi) / 2 * (1 + 2 * tf.random.uniform([1, n], minval=0, maxval=1, dtype=tf.float32))
    h = 20 * tf.random.uniform([1, n], minval=0, maxval=1, dtype=tf.float32)
    a1 = tf.constant(t * tf.cos(t))  ##映射第一个轴
    a3 = tf.constant(t * tf.sin(t))  ##映射第三个轴  ，第二个轴是h
    X = tf.concat([a1, h, a3], axis=0)  ##组成数据样本
    return X.numpy().T

def whole_remove(dataset, knn, unit_hop, ratio, percentage):
    print("reading")
    edist = dra.eucli_distance_all(dataset)
    print("edist")
    knn_edist = dra.knn_eucli_distance(dataset, knn)
    print("knn-edist")
    gdist, predecessors = dra.gdist_appro(knn_edist)
    print("gdist")
    path_dict, ave_egr, _ = dra.path_aveegr(edist, gdist, predecessors)
    print("dict, ave_egr")
    path_index = dra.bone_path(path_dict, gdist)
    weight = dra.bone_weight(path_dict, path_index)
    #print(weight)
    remove_tag = dra.dataset_compression_index(ave_egr, path_dict, gdist, unit_hop, ratio, path_index, weight)
    print("remove_tag")
    #print(remove_tag)
    rsub_data, rem_data = dra.dataset_compress(dataset, remove_tag, percentage)
    print("data")
    print(rsub_data.shape)
    return rsub_data, rem_data

def whole_augment(dataset, knn, unit_hop, ratio, percentage):
    print("reading")
    edist = dra.eucli_distance_all(dataset)
    print("edist")
    knn_edist = dra.knn_eucli_distance(dataset, knn)
    print("knn-edist")
    gdist, predecessors = dra.gdist_appro(knn_edist)
    print("gdist")
    path_dict, ave_egr, _ = dra.path_aveegr(edist, gdist, predecessors)
    print("dict, ave_egr")
    path_index = dra.bone_path(path_dict, gdist)
    weight = dra.bone_weight(path_dict, path_index)
    add_tag = dra.dataset_augment_index(ave_egr, path_dict, gdist, unit_hop, ratio, path_index, weight)
    print("add_tag")
    asub_data, add_data = dra.dataset_augment(dataset, add_tag, percentage, edist, path_dict)
    print(np.shape(asub_data))
    return asub_data, add_data

def polt_swissroll(data, change):
    plt.figure()
    x, y, z = list(data.T[0]), list(data.T[1]), list(data.T[2])
    x1, y1, z1 = list(change.T[0]), list(change.T[1]), list(change.T[2])
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x, y, z, s=10, alpha=0.3, c='r')
    ax.scatter(x1, y1, z1,s=10, alpha=0.8, c='b')
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

#dataset = generate_Swissroll(500)
dataset, t = skl.make_swiss_roll(n_samples = 1000, noise = 0.1)
x,y,z = list(dataset.T[0]), list(dataset.T[1]), list(dataset.T[2])
ax = plt.subplot(111, projection='3d')
ax.scatter(x, y, z, s=10, alpha=0.3, c='r')
ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()
#polt_swissroll(dataset)

dataset_cafter, sub_data = whole_remove(dataset, 5, 0.3, 0.9, 0.1)
print(np.shape(sub_data))
polt_swissroll(dataset_cafter, np.array(sub_data))

dataset_aafter, add_data = whole_augment(dataset,5, 0.9, 0.9, 0.1)
polt_swissroll(dataset, np.array(add_data))

np.savetxt("../swiss_roll/dataset.txt", dataset, fmt='%f',delimiter=',')
np.savetxt("../swiss_roll/dataset_rem.txt", dataset_cafter, fmt='%f',delimiter=',')
np.savetxt("../swiss_roll/dataset_add.txt", dataset_aafter, fmt='%f',delimiter=',')