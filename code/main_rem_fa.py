import numpy as np

import bone_dataset_remaug as dra
import dataset_visual as dv
import data_load as dl
import network as fnn
from multiprocessing.pool import Pool
from multiprocessing import Manager
from sklearn.model_selection import train_test_split
from multiprocessing import Process

#dv.make_mnist(True)

def label_parallel(info_list, unit_hop, ratio, percentage, l, new_data_dict):
    print("data_process")
    dataset = info_list[0]
    ave_egr = info_list[1]
    path_dict = info_list[2]
    gdist = info_list[4]
    path_index = info_list[5]
    weight = info_list[6]
    rsub_data, rem_data = dra.whole_remove(dataset,ave_egr, path_dict, gdist,path_index, weight, unit_hop, ratio, percentage)
    new_data_dict [l] = rsub_data


def floatrange(start,stop,steps):
    return [start+float(i)*(stop-start)/(float(steps)-1) for i in range(steps)]

def hyper_computation_parl(k, hdict):
   
    p = Pool(10)
    p.starmap_async(dra.hyper_computation, ((k, 0, hdict), (k, 1, hdict),(k, 2, hdict),(k, 3, hdict),(k, 4, hdict),(k, 5, hdict),(k, 6,hdict),(k, 7, hdict),(k, 8,hdict),(k, 9, hdict)))
    p.close()
    p.join()

    Hyper_ratio = []
    Hyper_uhop = []
    for l in range(10):
        Hyper_ratio.append(hdict[l][7])
        Hyper_uhop.append(hdict[l][8])
    
    Hyper_ratio = np.array(Hyper_ratio)
    Hyper_uhop = np.array(Hyper_uhop)
    hyper_ratio = np.min(Hyper_ratio, axis=0)
    hyper_uhop = np.min(Hyper_uhop, axis=0)
    return hyper_ratio, hyper_uhop, hdict 


def class_load_without_write(dict):
    X = dict[0]
    Y = [0 for i in range(len(dict[0]))]
    print(np.shape(X), np.shape(Y))
    for l in range(1,10):
        X_l = dict[l]
        Y_l = [l for i in range(len(dict[l]))]
        X = np.vstack((X, X_l))
        Y = np.append(Y,Y_l)
    X = X.reshape(-1,28,28)
    X = X * 255
    Y = Y.reshape(-1,1)
    print(np.shape(X), np.shape(Y))
    x_train, X_test, y_train, Y_test = train_test_split(X, Y, test_size = 0.01)
    x_train = np.concatenate((x_train, X_test), axis = 0)
    y_train = np.concatenate((y_train, Y_test), axis = 0)
    #print(x_train[0])
    return x_train, y_train


   
if __name__ == "__main__":
    print("processing")
    knn = [6, 8]
    per = [0.3]
    manager = Manager()
    new_data_dict = manager.dict()
    hdict = manager.dict()
    for k in knn:        #iteration for knn
        print("begin KNN")
        hyper_ratio, hyper_uhop, dict = hyper_computation_parl(k,hdict)
        print("hyper-parameter computation finish!")
        #print(hyper_ratio)
        #print(hyper_uhop)


        (x_train, y_train), (x_test, y_test) = dl.load_fashion_mnist(flag = False, path='')
        #eval_value_o = fnn.mnist_fnn(x_train, y_train, x_test, y_test)

        for unit_hop in floatrange(hyper_uhop[0][1], hyper_uhop[0][2], 7)[1:-2]:
            for ratio in floatrange(hyper_ratio[0][1], hyper_ratio[0][2], 7)[1:-2]:
                for percentage in per:
                    print("begin processing ieration")

                    
                    p0 = Process(target = label_parallel, args = (dict[0], unit_hop, ratio, percentage, 0, new_data_dict, ))
                    p1 = Process(target = label_parallel, args = (dict[1], unit_hop, ratio, percentage, 1, new_data_dict, ))
                    p2 = Process(target = label_parallel, args = (dict[2], unit_hop, ratio, percentage, 2, new_data_dict, ))
                    p3 = Process(target = label_parallel, args = (dict[3], unit_hop, ratio, percentage, 3, new_data_dict, ))
                    p4 = Process(target = label_parallel, args = (dict[4], unit_hop, ratio, percentage, 4, new_data_dict, ))
                    p5 = Process(target = label_parallel, args = (dict[5], unit_hop, ratio, percentage, 5, new_data_dict, ))
                    p6 = Process(target = label_parallel, args = (dict[6], unit_hop, ratio, percentage, 6, new_data_dict, ))
                    p7 = Process(target = label_parallel, args = (dict[7], unit_hop, ratio, percentage, 7, new_data_dict, ))
                    p8 = Process(target = label_parallel, args = (dict[8], unit_hop, ratio, percentage, 8, new_data_dict, ))
                    p9 = Process(target = label_parallel, args = (dict[9], unit_hop, ratio, percentage, 9, new_data_dict, ))

                    p0.start()
                    p1.start()
                    p2.start()
                    p3.start()
                    p4.start()
                    p5.start()
                    p6.start()
                    p7.start()
                    p8.start()
                    p9.start()

                    p0.join()
                    p1.join()
                    p2.join()
                    p3.join()
                    p4.join()
                    p5.join()
                    p6.join()
                    p7.join()
                    p8.join()
                    p9.join()

                    print("begin test and comparison")
                   
                    x_train, y_train = class_load_without_write(new_data_dict)
                    #print(np.shape(x_train))
                    acc_f = []
                    loss_f = []
                    acc_r = []
                    loss_r = []
                    acc_c = []
                    loss_c = []
                    for iter in range(10):
                        eval_value_n = fnn.mnist_fnn(x_train, y_train, x_test, y_test)
                        loss_f.append(eval_value_n[0])
                        acc_f.append(eval_value_n[1])

                        eval_value_n = fnn.mnist_rnn(x_train, y_train, x_test, y_test)
                        loss_r.append(eval_value_n[0])
                        acc_r.append(eval_value_n[1])

                        eval_value_n = fnn.mnist_cnn(x_train, y_train, x_test, y_test)
                        loss_c.append(eval_value_n[0])
                        acc_c.append(eval_value_n[1])

                    print("fnn", sum(loss_f)/10, sum(acc_f)/10)
                    print("fnn", loss_f, acc_f)
                    print("rnn", sum(loss_r)/10, sum(acc_r)/10)
                    print("rnn", loss_r, acc_r)
                    print("cnn", sum(loss_c)/10, sum(acc_c)/10)
                    print("cnn", loss_c, acc_c)
                    print("Rhyperparmater:knn-unit_hop-ratio-percentage-result(modified)", k, unit_hop, ratio, percentage)

   




        


    
