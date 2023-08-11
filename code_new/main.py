import numpy as np
import bone_dataset_remaug as dra
import dataset_visual as dv
import data_load as dl
import network as net
from multiprocessing.pool import Pool
from multiprocessing import Manager
from sklearn.model_selection import train_test_split
import os

curfile_dir = os.path.dirname(os.path.abspath(__file__))


def label_parallel(info_list, 
                   unit_hop, 
                   ratio, 
                   percentage, 
                   l, 
                   new_data_dict):
    try:
        dataset = info_list[0]
        ave_egr = info_list[1]
        path_dict = info_list[2]
        gdist = info_list[4]
        path_index = info_list[5]
        weight = info_list[6]
        rsub_data, rem_data = dra.whole_remove(dataset, ave_egr, path_dict, 
                                               gdist, path_index, weight, 
                                               unit_hop, ratio, percentage)
        new_data_dict[l] = rsub_data
    except:
        print("Error in label_parallel", l)
        return False


def floatrange(start, stop, num):
    return [start + float(i) * (stop - start) / (float(num) - 1) for i in range(num)]


def hyper_computation_parl(dataset_name, k):
    print("hyper_computation_parl")
    
    
# parallel version
    manager = Manager()
    shared_dict = manager.dict()
    p = Pool(10)
    for l in range(10):
        p.apply_async(dra.hyper_computation, args=(
            dataset_name, k, l, shared_dict))
    p.close()
    p.join()
    pdict = dict(shared_dict)
# parallel version end
    
    
# serial version
    pdict = dict()
    for l in range(10):
        pdict[l] = hhw_hyper_computation(k, l, dataset_name)
# serial version end
        
    Hyper_ratio = []
    Hyper_uhop = []
    for l in range(10):
        print("label", l)
        Hyper_ratio.append(pdict[l][7])
        Hyper_uhop.append(pdict[l][8])

    print("Hyper_ratio")
    for i in Hyper_ratio:
        print(i)
    
    print("Hyper_uhop")
    for i in Hyper_uhop:
        print(i)
        
    Hyper_ratio = np.array(Hyper_ratio)
    Hyper_uhop = np.array(Hyper_uhop)
    
    print("Hyper_ratio 2")
    for i in Hyper_ratio:
        print(i)
    
    print("Hyper_uhop 2")
    for i in Hyper_uhop:
        print(i)
        
    hyper_ratio = np.min(Hyper_ratio, axis=0)
    hyper_uhop = np.min(Hyper_uhop, axis=0)

    print("hyper_ratio\n", hyper_ratio)
    print("hyper_uhop\n", hyper_uhop)
    
    return hyper_ratio, hyper_uhop, pdict

def hhw_hyper_computation(k, l, dataset_name):
    hyper_uhop = []
    hyper_ratio = []
    dataset = dv.class_read(dataset_name, l)
    
    edist = dra.eucli_distance_all(dataset) # O(D * N^2)
    
    knn_edist = dra.knn_eucli_distance(dataset, k) # O(D * N^2)
    
    gdist, predecessors = dra.gdist_appro(knn_edist) # O(N^2 * logN)
    
    path_dict, ave_egr, path_hop = dra.path_aveegr(edist, gdist, predecessors) # time consuming O(N^3)
    # print(path_hop)
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
    
    # print("edist:")
    # print(edist)
    
    # print("knn_edist:")
    # print(knn_edist)
    
    # print("gdist:")
    # print(gdist)
    
    # print("predecessors:")
    # print(predecessors)
    
    # print("path_dict:")
    # print(path_dict)
    
    # print("ave_egr:")
    # print(ave_egr)
    
    # print("path_hop:")
    # print(path_hop)
    
    # print("path_index:")
    # print(path_index)
    
    # print("weight:")
    # print(weight)
    
    # print("bave_egr:")
    # print(bave_egr)

    # print("hop:")
    # print(hop)
    
    # print("bhop:")
    # print(bhop)
    
    
    return [dataset, ave_egr, path_dict, edist, gdist, path_index, weight, hyper_ratio, hyper_uhop]

def class_load_without_write(ndata_dict):
    X = ndata_dict[0]
    Y = [0 for i in range(len(ndata_dict[0]))]
    # print(np.shape(X), np.shape(Y))
    for l in range(1, 10):
        X_l = ndata_dict[l]
        Y_l = [l for i in range(len(ndata_dict[l]))]
        X = np.vstack((X, X_l))
        Y = np.append(Y, Y_l)
    X = X.reshape(-1, 28, 28)
    Y = Y.reshape(-1, 1)
    # print(np.shape(X), np.shape(Y))
    x_train, X_test, y_train, Y_test = train_test_split(
        X, Y, test_size=0.01, random_state=42)
    x_train = np.concatenate((x_train, X_test), axis=0)
    y_train = np.concatenate((y_train, Y_test), axis=0)
    return x_train, y_train


def main(dataset_name):
    print("processing")
    # knn = [8, 9, 10, 11, 12, 13, 14, 15]
    knn = [3]
    num_knn = len(knn)
    ki = 0
    result = []
    for k in knn:  # iteration for knn
        ki += 1
        hyper_ratio, hyper_uhop, pdict = hyper_computation_parl(dataset_name, k)
        print("hyper-parameter computation finish!")

        (x_train, y_train), (x_test, y_test) = dl.load_dataset(dataset_name, local=True)
        eval_value_o = net.mnist_cnn(x_train, y_train, x_test, y_test)

        num_unit_hop = 2
        num_ratio = 2
        num_percentage = 2
        num_all = num_unit_hop * num_ratio * num_percentage
        num_cur = 0
        range_unit_hop = np.linspace(
            round(hyper_uhop[0][1]), round(hyper_uhop[0][2]), num_unit_hop)
        range_ratio = floatrange(
            hyper_ratio[0][1], hyper_ratio[0][2], num_ratio)
        range_percentage = floatrange(0.1, 0.9, num_percentage)

        for unit_hop in range_unit_hop:
            for ratio in range_ratio:
                for percentage in range_percentage:
                    num_cur += 1
                    print(f"=== processing k: {ki} / {num_knn}, num_cur: {num_cur} / {num_all} ===")
                    print("begin processing ieration")

                    manager = Manager()
                    new_data_dict = manager.dict()
                    p = Pool(10)
                    for i in range(10):
                        p.apply_async(label_parallel,
                                      args=(pdict[i], unit_hop, 
                                            ratio, percentage,
                                            i, new_data_dict))
                    # info_list, unit_hop, ratio, percentage, label, new_data_dict
                    p.close()
                    p.join()
                    print("begin test and comparison")

                    ndata_dict = dict(new_data_dict)
                    x_train, y_train = class_load_without_write(ndata_dict)
                    # print(x_train.shape)

                    eval_value_n = net.mnist_cnn(
                        x_train, y_train, x_test, y_test)
                    print(
                        "hyperparmater:knn-unit_hop-ratio-percentage-result-result(modified)")
                    print("{knn}-{unit_hop}-{ratio:.2f}-{percentage:.2f}".format(
                        knn=k, unit_hop=unit_hop, ratio=ratio, percentage=percentage), end="-")
                    print("[{eval_value_o_0:.2f}, {eval_value_o_1: .2f}]".format(
                        eval_value_o_0=eval_value_o[0], eval_value_o_1=eval_value_o[1]), end="-")
                    print("[{eval_value_n_0:.2f}, {eval_value_n_1: .2f}]".format(
                        eval_value_n_0=eval_value_n[0], eval_value_n_1=eval_value_n[1])) 
                    # print(k, unit_hop, ratio, percentage,
                    #       eval_value_o, eval_value_n)
                    print("=== processing end ===")
                    result.append(
                        [k, unit_hop, ratio, percentage, eval_value_o, eval_value_n])

    # print(result)
    res_folder = os.path.join(
        curfile_dir, "..", "datas_modified", dataset_name)
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    res_file_path = os.path.join(res_folder, "result_remove_cnn.txt")
    with open(res_file_path, 'w') as f:
        for item in result:
            f.write("%s\n" % item)


if __name__ == "__main__":
    # dv.make_fashion_mnist(True)
    # dv.make_mnist(True)
    dataset_name = 'mnist_10'
    k = 5
    # hyper_ratio, hyper_uhop, pdict = hyper_computation_parl(dataset_name, k)
    # hhw_hyper_computation(k, 0, dataset_name)
    hyper_ratio, hyper_uhop, pdict = hyper_computation_parl(dataset_name, k)
    unit_hop = 8.0
    ratio = 0.5
    percentage = 0.1
    new_data_dict = {}
    # label_parallel(pdict[0], unit_hop, ratio, percentage, 0, new_data_dict)
    
    # print("new_data_dict[0]\n", new_data_dict[0])
    # print(len(new_data_dict[0]))
    # print(len(new_data_dict[0][0]))
    
    info_list = pdict[0]
    
    dataset = info_list[0]
    ave_egr = info_list[1]
    path_dict = info_list[2]
    gdist = info_list[4]
    path_index = info_list[5]
    weight = info_list[6]
    rsub_data, rem_data = dra.whole_remove(dataset, ave_egr, path_dict, 
                                            gdist, path_index, weight, 
                                            unit_hop, ratio, percentage)
    
    print("rsub_data\n", rsub_data)
    print("rem_data\n", rem_data)
    print(len(rem_data))
    
    