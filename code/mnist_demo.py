import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import bone_dataset_remaug as dra
import sklearn.datasets as skl
import dataset_visual as dv

from sklearn.datasets import make_moons





label= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#label= [0]
for i in label:
    data = dv.class_read('mnist', i)
    data = data[0:1000]
    #dataset_aafter, add_data = dra.whole_augment(data, 5, 0.9, 0.9, 0.05)
    dataset_aafter, remove_data = dra.whole_remove(data, 5, 0.9, 0.9, 0.1)
    dv.class_write(dataset_aafter, 'mnist_c', i, 28, 28)

