import numpy as np
import bone_dataset_remaug as dra
import dataset_visual as dv
import data_load as dl
import network as fnn
from multiprocessing.pool import Pool
from multiprocessing import Manager
from sklearn.model_selection import train_test_split


# dv.make_mnist(True)

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


def hyper_computation_parl(k):
    manager = Manager()
    shared_dict = manager.dict()
    p = Pool(10)
    for l in range(10):
        p.apply_async(dra.hyper_computation, args=(k, l, shared_dict))
    p.close()
    p.join()
    pdict = dict(shared_dict)
    print(len(pdict))
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
    x_train, X_test, y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=42)
    x_train = np.concatenate((x_train, X_test), axis=0)
    y_train = np.concatenate((y_train, Y_test), axis=0)
    return x_train, y_train


def main():
    print("processing")
    knn = [8, 9, 10, 11, 12, 13, 14, 15]

    result = []
    for k in knn:  # iteration for knn
        hyper_ratio, hyper_uhop, pdict = hyper_computation_parl(k)
        print("hyper-parameter computation finish!")

        (x_train, y_train), (x_test, y_test) = dl.load_mnist(flag=False, path='')
        eval_value_o = fnn.mnist_fnn(x_train, y_train, x_test, y_test)

        for unit_hop in range(round(hyper_uhop[0][1]), round(hyper_uhop[0][2]), 3):
            for ratio in floatrange(hyper_ratio[0][1], hyper_ratio[0][2], 9):
                for percentage in floatrange(0.1, 0.9, 9):
                    print("begin processing ieration")

                    manager = Manager()
                    new_data_dict = manager.dict()
                    p = Pool(10)
                    for i in range(10):
                        p.apply_async(label_parallel, args=(pdict[i], unit_hop, ratio, percentage, i, new_data_dict))
                    p.close()
                    p.join()
                    print("begin test and comparison")

                    ndata_dict = dict(new_data_dict)
                    x_train, y_train = class_load_without_write(ndata_dict)
                    print(x_train.shape)

                    eval_value_n = fnn.mnist_fnn(x_train, y_train, x_test, y_test)
                    print("Rhyperparmater:knn-unit_hop-ratio-percentage-result-result(modified)", k, unit_hop, ratio,
                          percentage, eval_value_o, eval_value_n)
                    result.append([k, unit_hop, ratio, percentage, eval_value_o, eval_value_n])

    # print(result)
    np.savetxt("../datas_modified/mnist_rem/result_remove.txt", result, fmt='%f', delimiter=',')


if __name__ == "__main__":
    main()









