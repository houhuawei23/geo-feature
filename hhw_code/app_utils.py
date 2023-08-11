


import numpy as np

from multiprocessing.pool import Pool
from multiprocessing import Manager


def dataset_compress(classified_dataset, each_class_feature, 
                     rate=0.1,
                     egr_threshold=0.5, uhop_threshold=8.0):
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
    s_res_dataset = Manager().dict()
    s_removed_dataset = Manager().dict()
    
# parallel
    num_classes = len(classified_dataset.keys())
    process_pool = Pool(num_classes)

    for label in classified_dataset.keys():
        process_pool.apply_async(class_compress, 
                                 args=(classified_dataset[label], 
                                       each_class_feature[label],
                                       rate,
                                       egr_threshold,
                                       uhop_threshold,
                                       s_res_dataset,
                                       s_removed_dataset))
    process_pool.close()
    process_pool.join()

# parallel end
    
# serial 
    # for label in classified_dataset.keys():
    #     # label = 1
    #     print(f"dataset_compress, label: {label}, shape: {classified_dataset[label].shape}")
    #     class_compress(X = classified_dataset[label], 
    #                    feature = each_class_feature[label],
    #                    rate = rate,
    #                    egr_threshold = egr_threshold,
    #                    uhop_threshold = uhop_threshold,
    #                    s_res_dataset = s_res_dataset,
    #                    s_removed_dataset = s_removed_dataset)
# serial end
    return dict(s_res_dataset), dict(s_removed_dataset)
        
        

def class_compress(X, feature, rate=0.1, 
                   egr_threshold=0.5, uhop_threshold=8.0, 
                   s_res_dataset=None, s_removed_dataset=None):
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
    # for i in feature['path_dict'].items():
    #     print(i)
    # print(len(feature['path_dict']))
    # print(f"num: {num}")
    # exit()
    for row in range(num):
        for col in range(row+1, num):
            # print(str(row)+'-'+str(col))
            is_bone_path = feature['bone_path_index'][row][col]
            egr = feature['path_avg_egr'][row][col]
            node_list = feature['path_dict'][str(row)+'-'+str(col)]['node_list']
            node_hops = feature['path_dict'][str(row)+'-'+str(col)]['len'] - 2
            hops_threshold = int(uhop_threshold * feature['geodesic_dist'][row][col])
            if is_bone_path and (egr != 0) and \
            (egr > egr_threshold) and (node_hops > hops_threshold):
                # print(f"egr: {egr}, node_hops: {node_hops}, hops_threshold: {hops_threshold}")
                subpath_egrs = []
                remove_candidate = []
                
                for k in range(node_hops):
                    # (k) - (k+2) egr
                    # start node id:            node_list[k]
                    # end node id:              node_list[k+2]
                    # mid node of (start, end): node_list[k+1]
                    subpath_egrs.append(feature['path_avg_egr'][node_list[k]][node_list[k+2]]) # 隔1个节点
                    remove_candidate.append(node_list[k+1])
                
                # 选取最小的 subpath_egrs
                subpath_egrs = np.array(subpath_egrs)
                min_index_list = np.argsort(subpath_egrs) # return id of sorted subpath_egrs
                
                for id in range(node_hops - hops_threshold):
                    index = min_index_list[id]
                    remove_tag[remove_candidate[index]] += 1
    
    for id in range(num):
        remove_tag[id] = remove_tag[id] / feature['bone_weight'][id]
        
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
    s_res_dataset[feature['label']] = res_X
    s_removed_dataset[feature['label']] = removed_data
    # return res_X, removed_data

def dataset_augment(classified_dataset, each_class_feature, 
                    rate=0.1,
                    egr_threshold=0.5, uhop_threshold=8.0):
    """
    classified_dataset: dict, the dataset classified by label
        type: dict(label: Tensor(size, features))
    
    each_class_feature: dict(feature), the geo feature of each label
        each_class_feature[label] = feature
    
    """
    s_res_dataset = Manager().dict()
    s_agumented_dataset = Manager().dict()
    
# parallel


def class_augment(X, feature, rate=0.1, 
                 egr_threshold=0.5, uhop_threshold=8.0, 
                 s_res_dataset=None, s_removed_dataset=None):
    """
    
    """
    num = X.shape[0]
    remove_tag = np.zeros(num)
    # for i in feature['path_dict'].items():
    #     print(i)
    # print(len(feature['path_dict']))
    # print(f"num: {num}")
    # exit()
    for row in range(num):
        for col in range(row+1, num):
            # print(str(row)+'-'+str(col))
            is_bone_path = feature['bone_path_index'][row][col]
            egr = feature['path_avg_egr'][row][col]
            node_list = feature['path_dict'][str(row)+'-'+str(col)]['node_list']
            node_hops = feature['path_dict'][str(row)+'-'+str(col)]['len'] - 2
            hops_threshold = int(uhop_threshold * feature['geodesic_dist'][row][col])
            if is_bone_path and (egr != 0) and \
            (egr > egr_threshold) and (node_hops > hops_threshold):
                # print(f"egr: {egr}, node_hops: {node_hops}, hops_threshold: {hops_threshold}")
                subpath_egrs = []
                remove_candidate = []
                
                for k in range(node_hops):
                    # (k) - (k+2) egr
                    # start node id:            node_list[k]
                    # end node id:              node_list[k+2]
                    # mid node of (start, end): node_list[k+1]
                    subpath_egrs.append(feature['path_avg_egr'][node_list[k]][node_list[k+2]]) # 隔1个节点
                    remove_candidate.append(node_list[k+1])
                
                # 选取最小的 subpath_egrs
                subpath_egrs = np.array(subpath_egrs)
                min_index_list = np.argsort(subpath_egrs) # return id of sorted subpath_egrs
                
                for id in range(node_hops - hops_threshold):
                    index = min_index_list[id]
                    remove_tag[remove_candidate[index]] += 1
    
    for id in range(num):
        remove_tag[id] = remove_tag[id] / feature['bone_weight'][id]
        
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
    s_res_dataset[feature['label']] = res_X
    s_removed_dataset[feature['label']] = removed_data
    # return res_X, removed_data