import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import bone_dataset_remaug as dra
import sklearn.datasets as skl
import dataset_visual as dv
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_moons

x, y = make_moons(n_samples=800, shuffle=True,
                  noise=0.1, random_state=None)
x = 10*x
plt.scatter(x[:, 0], x[:, 1],s=10, alpha = 0.3, c="r")
plt.show()


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
    print(weight)
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
    path_dict, ave_egr,_ = dra.path_aveegr(edist, gdist, predecessors)
    print("dict, ave_egr")
    path_index = dra.bone_path(path_dict, gdist)
    weight = dra.bone_weight(path_dict, path_index)
    add_tag = dra.dataset_augment_index(ave_egr, path_dict, gdist, unit_hop, ratio, path_index, weight)
    print("add_tag", add_tag)
    asub_data, add_data = dra.dataset_augment(dataset, add_tag, percentage, edist, path_dict)
    print(np.shape(asub_data))
    return asub_data, add_data

def polt_swissroll(data, change):
    plt.figure()
    x, y = list(data.T[0]), list(data.T[1])
    x1, y1 = list(change.T[0]), list(change.T[1])
    plt.scatter(x, y, s=10, alpha=0.3, c='r')
    plt.scatter(x1, y1,s=10, alpha=0.8, c='b')
    plt.show()


dataset_cafter, sub_data = whole_remove(x, 5, 1, 0.9, 0.1)
print(np.shape(sub_data))
polt_swissroll(dataset_cafter, np.array(sub_data))


dataset_aafter, add_data = whole_augment(x, 5, 10, 0.9, 0.2)
polt_swissroll(x, np.array(add_data))
