import numpy as np
import bone_dataset_remaug as dra
import dataset_visual as dv
import data_load as dl
import network as fnn
from multiprocessing.pool import Pool
from multiprocessing import Manager
from sklearn.model_selection import train_test_split


dv.make_mnist(True)


def label_parallel(info_list, unit_hop, ratio, percentage, l, new_data_dict):
    dataset = info_list[0]
    ave_egr = info_list[1]
    path_dict = info_list[2]
    gdist = info_list[4]
    path_index = info_list[5]
    weight = info_list[6]
    rsub_data, rem_data = dra.whole_remove(dataset, ave_egr, path_dict, gdist, path_index, weight, unit_hop, ratio,
                                           percentage)
    new_data_dict[l] = rsub_data


def floatrange(start, stop, steps):
    return [start + float(i) * (stop - start) / (float(steps) - 1) for i in range(steps)]


def hhw_hyper_computation(k, l, dataset_name):
    hyper_uhop = []
    hyper_ratio = []
    dataset = dv.class_read(dataset_name, l)
    # print("finish read dataset")
    edist = dra.eucli_distance_all(dataset) # O(D * N^2)
    # print("finish edist")
    knn_edist = dra.knn_eucli_distance(dataset, k) # O(D * N^2)
    print("finish knn_edist")
    gdist, predecessors = dra.gdist_appro(knn_edist) # O(N^2 * logN)
    print("finish gdist")
    path_dict, ave_egr, path_hop = dra.path_aveegr(edist, gdist, predecessors) # time consuming O(N^3)
    # print("finish path_dict")
    path_index = dra.bone_path(path_dict, gdist)    # O(N^3) ~ O(N^4) ?
    # print("finish path_index")
    weight = dra.bone_weight(path_dict, path_index) # O(N^3) worst
    # print("finish weight")
    bave_egr = np.multiply(path_index, ave_egr) # O(N^2)
    hyper_ratio.append([bave_egr.min(), bave_egr.mean(), bave_egr.max()])
    hop = gdist / path_hop
    bhop = np.multiply(path_index, hop)
    hyper_uhop.append([bhop.min(), bhop.mean(), bhop.max()])
    return [dataset, ave_egr, path_dict, edist, gdist, path_index, weight, hyper_ratio, hyper_uhop]


"""
dataset: N * (28 * 28)

edist: N * N, euclidean distance

knn_edist: N * N, k nearest neighbor
    if (i, j) is neighbor, knn_edist[i][j] = edist[i][j]
    else, knn_edist[i][j] = 0

gdist: N * N, geodesic distance

predecessors: N * N, predecessors, 前一个节点

path_dict: (N * N / 2), path_dict['i-j'] = [node_list, length]
    node_list: [i, ..., j]
    length: length of path

ave_egr: N * N, average edge ratio

path_hop: N * N, path hop, number of nodes in path

path_index: N * N, path index
    if (i, j) is bone path, path_index[i][j] = 1
    else, path_index[i][j] = 0

weight: N, bone weight for each node

bave_egr: N * N, bone average edge ratio
    if (i, j) is bone path, bave_egr[i][j] = ave_egr[i][j]
    else, bave_egr[i][j] = 0

hyper_ratio: 1 * 1 * 3, [[min, mean, max]]  of bave_egr

hop = gdist / path_hop: N * N, average hop for each node

bhop: N * N, bone hop

hyper_uhop: 1 * 1 * 3, [[min, mean, max]] of bhop

return: pdict[i]
    0: dataset
    1: ave_egr
    2: path_dict
    3: edist
    4: gdist
    5: path_index
    6: weight
    7: hyper_ratio
    8: hyper_uhop

"""


def hyper_computation_parl(k, dataset_name):
    # need to download dataset
    """
    manager = Manager()
    shared_dict = manager.dict()
    p = Pool(10)
    for l in range(10):
        p.apply_async(dra.hyper_computation, args=(k, l, shared_dict))
    p.close()
    p.join()
    pdict = dict(shared_dict)
    """
    pdict = dict()
    for l in range(10):
        pdict[l] = hhw_hyper_computation(k, l, dataset_name)
    Hyper_ratio = []
    Hyper_uhop = []
    for l in range(10):
        Hyper_ratio.append(pdict[l][7])
        Hyper_uhop.append(pdict[l][8])

    Hyper_ratio = np.array(Hyper_ratio)
    Hyper_uhop = np.array(Hyper_uhop)
    hyper_ratio = np.min(Hyper_ratio, axis=0)
    hyper_uhop = np.min(Hyper_uhop, axis=0)

    return hyper_ratio, hyper_uhop, pdict


"""
Hyper_ratio: 10 * 3, [min, mean, max] of bave_egr for each class
Hyper_uhop: 10 * 3, [min, mean, max] of bhop for each class
hyper_ratio: 3 * 1, [min, mean, max] of bave_egr for all classes
hyper_uhop: 3 * 1, [min, mean, max] of bhop for all classes
pdict: 10 * 9, see hhw_hyper_computation
"""


def class_load_without_write(ndata_dict):
    X = ndata_dict[0]
    Y = [0 for i in range(len(ndata_dict[0]))]
    print(np.shape(X), np.shape(Y))
    for l in range(1, 10):
        X_l = ndata_dict[l]
        Y_l = [l for i in range(len(ndata_dict[l]))]
        X = np.vstack((X, X_l))
        Y = np.append(Y, Y_l)
    X = X.reshape(-1, 28, 28)
    Y = Y.reshape(-1, 1)
    print(np.shape(X), np.shape(Y))
    x_train, X_test, y_train, Y_test = train_test_split(
        X, Y, test_size=0.01, random_state=42)
    x_train = np.concatenate((x_train, X_test), axis=0)
    y_train = np.concatenate((y_train, Y_test), axis=0)
    return x_train, y_train
import os
from PIL import Image
def read(dataset_name):
    cur_dir = os.getcwd()
    data_dir = os.path.join(cur_dir, "datas", dataset_name, 'train')
    classes = os.listdir(data_dir)
    
    # Initialize empty lists to store the image data and labels
    data = []
    labels = []

    # Loop over each class directory
    for class_dir in classes:
        class_dir_path = os.path.join(data_dir, class_dir)
        
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

    return (x_train, y_train), (x_test, y_test)

def main(dataset_name):
    print("processing")
    # knn = [8, 9, 10, 11, 12, 13, 14, 15]
    knn = [8, 9]
    result = []
    for k in knn:  # iteration for knn
        hyper_ratio, hyper_uhop, pdict = hyper_computation_parl(k, dataset_name)
        print("hyper-parameter computation finish!")
        # print(f'hyper_ratio:{hyper_ratio}')
        # print(f'hyper_uhop:{hyper_uhop}')
        # print("pdict:", pdict)
        # (x_train, y_train), (x_test, y_test) = dl.load_mnist(flag=False, path='')
        (x_train, y_train), (x_test, y_test) = read(dataset_name)
        eval_value_o = fnn.mnist_fnn(
            x_train, y_train, x_test, y_test)  # 移到循环外？
        # [loss, accuracy]
        # 对于每个单位跳数（unit_hop），比例（ratio）和百分比（percentage），调用label_parallel函数来对数据进行预处理
        for unit_hop in range(round(hyper_uhop[0][1]), round(hyper_uhop[0][2]), 3):
            for ratio in floatrange(hyper_ratio[0][1], hyper_ratio[0][2], 3):
                for percentage in floatrange(0.1, 0.5, 5):
                    # 搜参？？
                    print("----------------------------------")
                    print("begin processing ieration")
                    print("unit_hop:", unit_hop)
                    print("ratio:", ratio)
                    print("percentage:", percentage)
                    manager = Manager()
                    new_data_dict = manager.dict()
                    p = Pool(10)
                    for i in range(10):
                        p.apply_async(label_parallel,
                                      args=(pdict[i], unit_hop, ratio, percentage, i, new_data_dict))
                    # label_parallel: O(N^3) worst
                    p.close()
                    p.join()
                    print("begin test and comparison")

                    ndata_dict = dict(new_data_dict)
                    x_train, y_train = class_load_without_write(ndata_dict)
                    # print(x_train.shape)
                    # 再次调用fnn函数，对处理后的数据进行测试
                    eval_value_n = fnn.mnist_fnn(
                        x_train, y_train, x_test, y_test)
                    print("Rhyperparmater:knn-unit_hop-ratio-percentage-result-result(modified)",
                          k, unit_hop, ratio, percentage, eval_value_o, eval_value_n)
                    result.append(
                        [k, unit_hop, ratio, percentage, eval_value_o, eval_value_n])
                    print("----------------------------------")


    for i in range(len(result)):
        print(result[i])
    # np.savetxt("./datas_modified/mnist_rem/result_remove.txt",
    #            result, fmt='%f', delimiter=',')

"""
result[i]:
0: knn
1: unit_hop
2: ratio
3: percentage
4: eval_value_o
5: eval_value_n
"""
if __name__ == "__main__":
    dataset_name = "mnist_50"
    main(dataset_name)
