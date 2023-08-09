import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import bone_dataset_remaug as dra
import sklearn.datasets as skl
import dataset_visual as dv
import data_load as dl

from sklearn.datasets import make_moons


#dv.make_fashion_mnist(True)
label= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#label= [0]
for i in label:
    data = dv.class_read('fashion_mnist', i)
    data = data[0:1000]
    dataset_aafter, add_data = dra.whole_augment(data, 5, 0.9, 0.9, 0.05)
    dv.class_write(add_data, 'fashion_mnist', i, 28, 28)