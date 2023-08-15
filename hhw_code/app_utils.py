import numpy as np

from multiprocessing.pool import Pool
from multiprocessing import Manager

from geo import get_shortest_path


def dataset_compress_beta(
    classified_dataset: dict,
    each_class_feature: dict,
    rate: float = 0.1,
    egr_threshold: float = 0.5,
    udist_threshold: float = 10.0,
):
    labels = list(classified_dataset.keys())
    n_classes = len(labels)
    # serial
    features = []
    for label in labels:
        ifeature = class_compress_beta()
        features.append(ifeature)
    removed_dataset = {}
    results = {}
    for ifeature in features:
        label = ifeature["label"]
        removed_dataset[label] = ifeature["removed_dataset"]
        results[label] = ifeature["res_dataset"]
    return results, removed_dataset


def dataset_compress(
    classified_dataset,
    each_class_feature,
    rate=0.1,
    egr_threshold=0.5,
    uhop_threshold=8.0,
):
    """
    input:
        classified_dataset: dict, the dataset classified by label
            type: dict(label: Tensor(size, features))

        each_class_feature: dict(feature), the geo feature of each label
            each_class_feature[label] = feature

    output:
        - s_res_dataset: dict, the dataset after compress
        - s_removed_dataset: dict, the removed data
    """
    # s_res_dataset = Manager().dict()
    # s_removed_dataset = Manager().dict()

    # parallel
    num_classes = len(classified_dataset.keys())
    process_pool = Pool(num_classes)
    res_dataset = {}
    removed_dataset = {}
    results = {}
    for label in classified_dataset.keys():
        result = process_pool.apply_async(
            class_compress,
            args=(
                classified_dataset[label],
                each_class_feature[label],
                rate,
                egr_threshold,
                uhop_threshold,
            ),
        )
        results[label] = result
    process_pool.close()
    process_pool.join()
    for label, result in results.items():
        # print("")
        res_X, removed_data = result.get()

        res_dataset[label] = res_X
        removed_dataset[label] = removed_data

    # parallel end

    # serial

    # serial end
    return res_dataset, removed_dataset


def class_compress_beta(
    X: np.ndarray,
    feature: dict,
    rate: float = 0.1,
    aegr_thres: float = 0.5,
    udist_thres: float = 10.0,
):
    try:
        num = X.shape[0]
        remove_tag = np.zeros(num, dtype=np.float32)
        for row in range(num):
            for col in range(row + 1, num):
                is_bone = feature["bone_path_mask"][row][col]
                aegr = feature["path_aegr"][row][col]
                if is_bone and (aegr != 0) and (aegr > aegr_thres):
                    
                    path_len = feature["path_len"][row][col]
                    nodes_arr = get_shortest_path(feature["pre_matrix"], row, col)
                    
                    for i in range(1, path_len - 1):
                        remove_tag[nodes_arr[i]] += 1
        remove_num = int(num * rate)
        remove_tag = remove_tag / feature["bone_weight"]
        removed_data = []
        # 选出remove_tag最大的remove_num个点
        remove_index = np.argsort(remove_tag)[-remove_num:]
        # res_X = np.delete(X, remove_index, axis=0)
        # removed_data = X[remove_index]
    except Exception as e:
        print(e)
    
    return remove_index
def class_compress(X, feature, rate=0.1, egr_threshold=0.5, uhop_threshold=8.0):
    """
    input:
        - X: Tensor(size, features)
        - feature: dict, the geo feature of the class
            - 'label', 'path_avg_egr', 'path_dict', 'euclidean_dist', 'geodesic_dist'
            - 'bone_path_index', 'bone_weight', 'class_uhop', 'class_ratio'
        - rate: delete rate (float)
        - egr_threshold: the threshold of egr (float)
        - uhop_threshold: the threshold of uhop (float)

    variables:
        - bone_weight: 越大, bone 权重越高, 越不应删除？
        - remove_tag: 越大，票数越多，越应被删除
        - subpath_egrs: the 1-hop subpath egr list of p_ij
    """
    num = X.shape[0]
    remove_tag = np.zeros(num)

    try:
        for row in range(num):
            for col in range(row + 1, num):
                # print(str(row)+'-'+str(col))
                is_bone_path = feature["bone_path_index"][row][col]
                egr = feature["path_avg_egr"][row][col]
                node_arr = feature["path_dict"][str(row) + "-" + str(col)]["node_arr"]
                node_hops = feature["path_dict"][str(row) + "-" + str(col)]["len"] - 2
                hops_threshold = int(
                    uhop_threshold * feature["geodesic_dist"][row][col]
                )
                if (
                    is_bone_path
                    and (egr != 0)
                    and (egr > egr_threshold)
                    and (node_hops > hops_threshold)
                ):
                    # print(f"egr: {egr}, node_hops: {node_hops}, hops_threshold: {hops_threshold}")
                    subpath_egrs = []
                    remove_candidate = []

                    for k in range(node_hops):
                        # (k) - (k+2) egr
                        # start node id:            node_arr[k]
                        # end node id:              node_arr[k+2]
                        # mid node of (start, end): node_arr[k+1]
                        subpath_egrs.append(
                            feature["path_avg_egr"][node_arr[k]][node_arr[k + 2]]
                        )  # 隔1个节点
                        remove_candidate.append(node_arr[k + 1])

                    # 选取最小的 subpath_egrs
                    subpath_egrs = np.array(subpath_egrs)
                    min_index_list = np.argsort(
                        subpath_egrs
                    )  # return id of sorted subpath_egrs

                    for id in range(node_hops - hops_threshold):
                        index = min_index_list[id]
                        remove_tag[remove_candidate[index]] += 1

        # for id in range(num):
        #     remove_tag[id] = remove_tag[id] / feature["bone_weight"][id]
        remove_tag = remove_tag / feature["bone_weight"]
        # res_data, remove_data
        removed_data = []
        topk = int(num * rate)
        import heapq
        from torch import stack as torch_stack

        max_num_index = heapq.nlargest(topk, range(len(remove_tag)), remove_tag.take)
        max_num_index.sort(reverse=True)

        res_X = np.delete(X, max_num_index, axis=0)
        removed_data = X[max_num_index]
    except Exception as e:
        print(e)
        res_X = X
        removed_data = []

    return res_X, removed_data


def dataset_augment(
    classified_dataset,
    each_class_feature,
    rate=0.1,
    egr_threshold=0.5,
    uhop_threshold=8.0,
):
    """
    classified_dataset: dict, the dataset classified by label
        type: dict(label: Tensor(size, features))

    each_class_feature: dict(feature), the geo feature of each label
        each_class_feature[label] = feature

    """
    s_res_dataset = Manager().dict()
    s_agumented_dataset = Manager().dict()


# parallel


def class_augment(
    X,
    feature,
    rate=0.1,
    egr_threshold=0.5,
    uhop_threshold=8.0,
    s_res_dataset=None,
    s_removed_dataset=None,
):
    """ """
    num = X.shape[0]
    remove_tag = np.zeros(num)
    # for i in feature['path_dict'].items():
    #     print(i)
    # print(len(feature['path_dict']))
    # print(f"num: {num}")
    # exit()
    for row in range(num):
        for col in range(row + 1, num):
            # print(str(row)+'-'+str(col))
            is_bone_path = feature["bone_path_index"][row][col]
            egr = feature["path_avg_egr"][row][col]
            node_arr = feature["path_dict"][str(row) + "-" + str(col)]["node_arr"]
            node_hops = feature["path_dict"][str(row) + "-" + str(col)]["len"] - 2
            hops_threshold = int(uhop_threshold * feature["geodesic_dist"][row][col])
            if (
                is_bone_path
                and (egr != 0)
                and (egr > egr_threshold)
                and (node_hops > hops_threshold)
            ):
                # print(f"egr: {egr}, node_hops: {node_hops}, hops_threshold: {hops_threshold}")
                subpath_egrs = []
                remove_candidate = []

                for k in range(node_hops):
                    # (k) - (k+2) egr
                    # start node id:            node_arr[k]
                    # end node id:              node_arr[k+2]
                    # mid node of (start, end): node_arr[k+1]
                    subpath_egrs.append(
                        feature["path_avg_egr"][node_arr[k]][node_arr[k + 2]]
                    )  # 隔1个节点
                    remove_candidate.append(node_arr[k + 1])

                # 选取最小的 subpath_egrs
                subpath_egrs = np.array(subpath_egrs)
                min_index_list = np.argsort(
                    subpath_egrs
                )  # return id of sorted subpath_egrs

                for id in range(node_hops - hops_threshold):
                    index = min_index_list[id]
                    remove_tag[remove_candidate[index]] += 1

    for id in range(num):
        remove_tag[id] = remove_tag[id] / feature["bone_weight"][id]

    # res_data, remove_data
    removed_data = []
    topk = int(num * rate)
    import heapq
    from torch import stack as torch_stack

    max_num_index = heapq.nlargest(topk, range(len(remove_tag)), remove_tag.take)
    max_num_index.sort(reverse=True)

    res_X = np.delete(X, max_num_index, axis=0)
    removed_data = X[max_num_index]

    # return
    s_res_dataset[feature["label"]] = res_X
    s_removed_dataset[feature["label"]] = removed_data
    # return res_X, removed_data
