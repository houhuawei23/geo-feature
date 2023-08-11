

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

