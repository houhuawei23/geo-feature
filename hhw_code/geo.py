import numpy as np

import torch

import sklearn.neighbors as sk_neighbors

from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from multiprocessing.pool import Pool

def get_bone_weight(path_dict, path_index):
    """
    input:
        path_dict: (N * N / 2), path_dict['i-j'] = [node_arr, length]
        path_index: N * N, path_index[i][j] = 1, if i-j is bone
    output:
        weight_node: N, weight_node[i] = the number of bones that i is in
    """
    # print("get_bone_weight")
    num = np.shape(path_index)[0]
    weight_node = np.zeros(num)
    for i in range(num):
        for j in range(i+1, num):
            if (path_index[i][j] == 1):
                node_arr = path_dict[str(i)+'-'+str(j)]['node_arr']
                # print(node_arr)
                for node in node_arr:
                    weight_node[node] = weight_node[node] + 1
    # print("get_bone_weight done")
    weight_node = weight_node.astype(np.int32)
    return weight_node


def get_bone_path(path_dict, gdist):
    """
    input:
        path_dict: (N * N / 2), path_dict['i-j'] = [node_arr, length]
        gdist: N * N, geodesic distance
    output:
        bone_path_mask: N * N, path_index[i][j] = 1, if i-j is bone
    """
    # print("bone_path")
    num = np.shape(gdist)[0]
    bone_path_mask = - np.ones((num, num))
    # print((num*num-num)/2)
    sort_path = sorted(path_dict.items(), key=lambda x: x[1]['len'], reverse=True) # list

    # [('0-1', [[0, 1], 2]), ('0-2', [[0, 3, 2], 3]), ...]
    # print(type(sort_path))
    for _, m_path in (sort_path):
        m_list = m_path['node_arr']
        m_len = m_path['len']
        if ((bone_path_mask[m_list[0]][m_list[m_len-1]] == 0) or (bone_path_mask[m_list[m_len-1]][m_list[0]] == 0)):
            continue
        else:
            for i in range(m_len):
                for j in range(i, m_len):
                    bone_path_mask[m_list[i]][m_list[j]] = 0
                    bone_path_mask[m_list[j]][m_list[i]] = 0
            bone_path_mask[m_list[0]][m_list[m_len - 1]] = 1
            bone_path_mask[m_list[m_len - 1]][m_list[0]] = 1
    bone_path_mask = bone_path_mask.astype(np.int8)
    return bone_path_mask

def get_path_ave_egr(euclidean_dist, geodesic_dist, predecessors):
    """
    input:  
        euclidean_dist: N * N, euclidean distance
        geodesic_dist: N * N, geodesic distance
        predecessors: N * N, predecessors, 前一个节点
    return:

    """
    num = euclidean_dist.shape[0]
    path_ave_egr = np.zeros((num, num))
    path_hop = np.ones((num, num))
    path_dict = {}
    for i in range(num):
        for j in range(i + 1, num): # 可并行
            if (predecessors[i][j] != -9999):
                node_arr, length = path_node(predecessors, i, j)
                node_num = len(node_arr) - 2
                path_hop[i][j] = max(length - 2, 0)
                if (node_num == 0):
                    path_ave_egr[i][j] = 1
                    # path_dict[str(i)+'-'+str(j)] = [node_arr, length]
                    path_dict[str(i)+'-'+str(j)] = {'node_arr': node_arr, 'len': length}
                else:
                    egr = 0
                    for k in range(node_num):
                        egr = egr + \
                            (euclidean_dist[node_arr[k]][node_arr[k+2]] /
                             geodesic_dist[node_arr[k]][node_arr[k+2]])
                    path_ave_egr[i][j] = egr / node_num
                    # path_dict[str(i)+'-'+str(j)] = [node_arr, length]
                    path_dict[str(i)+'-'+str(j)] = {'node_arr': node_arr, 'len': length}
    return path_ave_egr, path_hop, path_dict

def path_node(pre_mat, start_node, goal_node):
    """
    input:
        pre_mat: N * N, predecessors, 前一个节点
        start_node: int, start node
        goal_node: int, goal node
    return:
        node_arr: list, the path node
        length: int, the length of the path
    """
    
    node_arr = []
    node_arr.append(start_node)
    pre_node = pre_mat[goal_node][start_node]

    while (goal_node != pre_node):
        node_arr.append(pre_node)
        pre_node = pre_mat[goal_node][pre_node]

    node_arr.append(goal_node)
    length = len(node_arr)
    # print("path_node done")
    node_arr = np.array(node_arr).astype(np.int16)# max 60000
    return node_arr, length


def get_geodesic_dist(knn_edist):
    """
    input:
        knn_edist: N * N, k nearest neighbor
    return:
        geodesic_dist: N * N, geodesic distance
        predecessors: N * N, predecessors, 前一个节点
    """
    csr_knn_edist = csr_matrix(knn_edist)
    geodesic_dist, predecessors = shortest_path(
        csr_knn_edist, directed=False, return_predecessors=True)
    
    return geodesic_dist, predecessors


def get_knn_edist(euclidean_dist, k):
    """
    input:
        euclidean_dist: torch.Tensor
            shape: (num_images, num_images)
        k: int, the number of neighbors
    return:
        knn_edist: torch.Tensor
            shape: (num_images, num_images)
    """
    # euclidean_dist.fill_diagonal_(float('inf'))
    # print('euclidean_dist:\n', euclidean_dist)
    _, indices = torch.topk(euclidean_dist, k + 1, dim=1, largest=False)
    mask = torch.zeros_like(euclidean_dist)
    mask.scatter_(1, indices, 1)
    knn_edist = euclidean_dist * mask
    # nan
    # knn_edist
    # knn_edist[knn_edist == 0] = float('inf')
    return knn_edist

def get_class_geo_feature(label, X, k):
    # print(f"get_class_geo_feature, label: {label}, shape: {X.shape}")
    """
input:
    label: int, the label of the class
    X: torch.Tensor, the images of the class
        shape: (num_images, 1, 28, 28)
    k: int, the number of neighbors
output:
    shared_dict: Manager.dict(), shared_dict[label] = feature

    feature: dict, the geo feature of the class
    - 'label': the label of the class
    * - 'data': tensor(size, features)
    - 'path_avg_egr': N * N, average euclidean / geodesic distance ratio
    - 'path_dict': (N * N / 2), path_dict['i-j'] = [node_arr, length]
    - 'euclidean_dist': N * N, euclidean distance
    - 'geodesic_dist': N * N, geodesic distance
    - 'bone_path_index': N * N, path_index[i][j] = 1, if i-j is bone path
    - 'bone_weight': N, weight[i] = the number of bones that i is in
    - 'label_ave_egr': tuple shape(1, 3), (min, mean, max) of bone_ave_egr
    - 'label_uhop': tuple shape(1, 3), (min, mean, max) of bone_uhop
        
    """
    feature = {}
    
    try:
        euclidean_dist = cdist(X, X, metric='euclidean')
        
        # knn_edist = get_knn_edist(euclidean_dist, k)
        knn_edist = sk_neighbors.kneighbors_graph(X, k, mode='distance').toarray()
        
        geodesic_dist, predecessors = get_geodesic_dist(knn_edist)
        
        path_ave_egr, path_hop, path_dict = get_path_ave_egr(euclidean_dist, geodesic_dist, predecessors)

        bone_path_index = get_bone_path(path_dict, geodesic_dist)
        
        bone_path_index = bone_path_index.astype(np.int8)
        
        bone_weight = get_bone_weight(path_dict, bone_path_index)
        
        bone_ave_egr = np.multiply(bone_path_index, path_ave_egr)

        class_ave_egr = np.array([bone_ave_egr.mean(), bone_ave_egr.max()]) # min always 0
      
        geodesic_dist = np.where(geodesic_dist < 1e-6, 1, geodesic_dist)
        gdist_uhop = path_hop / geodesic_dist
       
        bone_uhop = np.multiply(bone_path_index, gdist_uhop)
    
        class_uhop = np.array([bone_uhop.mean(), bone_uhop.max()] )# min always 0

        path_ave_egr = path_ave_egr.astype(np.float32)
        geodesic_dist = geodesic_dist.astype(np.float32)
        
        
        feature['label'] = label
        # feature['data'] = X
        feature['path_avg_egr'] = path_ave_egr # f64
        feature['path_dict'] = path_dict # val['node_arr] int, len, int
        # feature['euclidean_dist'] = euclidean_dist  
        feature['geodesic_dist'] = geodesic_dist # 64
        feature['bone_path_index'] = bone_path_index #f64
        feature['bone_weight'] = bone_weight # f64
        feature['class_ave_egr'] = class_ave_egr    # f64
        feature['class_uhop'] = class_uhop          # f64
        # if label == 0:
        #     print(f"feature[path_dict] dtype:{feature['path_dict']['0-1']['node_arr'].dtype}")
            
    except Exception as e:
        print(f"error in geo_class: {e}, label: {label}") 
        return 0

    # print(f"get_class_feature: {label} finished")
    return feature

def get_dataset_geo_feature(classified_dataset, k):
    """
input:
    classified_dataset: dict, the dataset classified by label
        type: dict(Tensor(size, features))
    k: int, the number of neighbors
output:
    each_class_feature: dict(feature), the geo feature of each label
        each_class_feature[label] = feature
    """

    # 计算每个类别的几何特征
    num_classes = len(classified_dataset.keys())

# serial
    # for label in classified_dataset.keys():
    #     get_class_geo_feature(label, classified_dataset[label], k, s_each_class_feature)
# serial end

# multi-process
    process_pool = Pool(num_classes)
    results = {}
    for label in classified_dataset.keys():
        # print(f"lable: {label}, size: {classified_dataset[label].shape}")
        result = process_pool.apply_async(get_class_geo_feature,  
                                          args=(label, 
                                                classified_dataset[label], 
                                                k))
        results[label] = result
    process_pool.close()
    process_pool.join()
    for label, result in results.items():
        results[label] = result.get()
# multi-process end

    
    # get global dataset egr and uhop
    egr_scope = np.array([0.0, 0.0])
    uhop_scope = np.array([0.0, 0.0])
    for label in results.keys():
        # print(f"label: {label}")
        egr_scope += results[label]['class_ave_egr']
        uhop_scope += results[label]['class_uhop']
    # print("scope", egr_scope, uhop_scope)
    egr_scope /= num_classes
    uhop_scope /= num_classes        

    return results, egr_scope, uhop_scope

from collections import defaultdict
def get_classified_dataset(data_arr, label_arr):
    
    """
    dataset: torch.utils.data.Dataset
    
    classified_dataset: dict, the dataset classified by label
        type: dict(Tensor(size, features))
    """
    
    classified_dataset = defaultdict(list)
    for image, label in zip(data_arr, label_arr):
        classified_dataset[label].append(np.array(image).reshape(-1))
    # print(classified_dataset[0])
    # stack the list:
    for label in classified_dataset.keys():
        classified_dataset[label] = np.stack(classified_dataset[label])
    
    return dict(classified_dataset)

# def get_merged_dataset()

if __name__ == '__main__':
    import os
    import time 
    import pickle

    import torchvision
    import torchvision.transforms as transforms
    
    from utils import set_seed
    set_seed(42)
    curfile_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(curfile_dir, '..', 'dataset')
    results_dir = os.path.join(curfile_dir, "results")
    dataset_name = 'mnist'    


    mnist_trainset = torchvision.datasets.MNIST(
        root=dataset_dir, train=True, download=False, transform=transforms.ToTensor())
    mnist_data_arr = mnist_trainset.data.numpy()
    mnist_label_arr = mnist_trainset.targets.numpy()

    train_size = 2000
    k = 8
    sample_index = np.random.choice(range(train_size), train_size, replace=False)
    mnist_data_arr = mnist_data_arr[sample_index].reshape(train_size, -1)
    mnist_label_arr = mnist_label_arr[sample_index]

    classified_dataset = get_classified_dataset(mnist_data_arr, mnist_label_arr)

    
    start = time.time() 
    each_class_feature, egr_scope, uhop_scope = get_dataset_geo_feature(classified_dataset, k=4)
    end = time.time()
    print(f"size:{train_size}, get_dataset_geo_feature time: {end - start}")
    
    print(f"each_class_feature keys: {each_class_feature.keys()}")
    
    print(f"egr_scope: {egr_scope}")
    print(f"uhop_scope: {uhop_scope}")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    geo_feature_path = os.path.join(
        results_dir, f"geo_{dataset_name}_trsz{train_size}_k{k}.pkl"
    )

    # with open(geo_feature_path, "wb") as f:
    #     pickle.dump(each_class_feature, f)
    #     pickle.dump(egr_scope, f)
    #     pickle.dump(uhop_scope, f)
    
    from app_utils import dataset_compress
    import utils as my_utils
    rate = 0.1
    egr_threshold = egr_scope[1] - (egr_scope[1] - egr_scope[0]) * rate
    uhop_threshold = uhop_scope[1] - (uhop_scope[1] - uhop_scope[0]) * rate
    res_data_dict, removed_data_dict = dataset_compress(classified_dataset = classified_dataset,
                                            each_class_feature = each_class_feature,
                                            rate = rate, 
                                            egr_threshold = egr_threshold, 
                                            uhop_threshold = uhop_threshold)
    # print(f"res_data size: {len(res_data)}")
    # print(f"removed_data size: {len(removed_data)}")
    # print(f"res_data keys: {res_data.keys()}")  
    # print(f"removed_data keys: {removed_data.keys()}")
    print(f"res_data[0] shape: {res_data_dict[0].shape}")
    print(f"removed_data[0] shape: {removed_data_dict[0].shape}")
    import torch.utils.data as torch_data
    res_train_X = np.concatenate([res_data_dict[label] for label in res_data_dict.keys()]).reshape(-1, 1, 28, 28).astype(np.float32)
    res_train_y = np.concatenate([np.ones(res_data_dict[label].shape[0]) * label for label in res_data_dict.keys()]).astype(np.int64)
    res_train_dataset = torch_data.TensorDataset(torch.from_numpy(res_train_X), torch.from_numpy(res_train_y))
    
    batch_size = 64
    
    res_train_iter = torch_data.DataLoader(res_train_dataset, batch_size=batch_size, shuffle=True)
    
    mnist_testset = torchvision.datasets.MNIST(
        root=dataset_dir, train=False, download=False, transform=transforms.ToTensor())
    
    mnist_test_iter = torch_data.DataLoader(mnist_testset, batch_size=batch_size, shuffle=False)

    for X, y in res_train_iter:
        print("res_iter", X.shape, y.shape)
        print(f"type: X: {X.dtype}, y: {y.dtype}")
        break
    for X, y in mnist_test_iter:
        print("test_iter", X.shape, y.shape)
        print(f"type: X: {X.dtype}, y: {y.dtype}")
        break
    from network import LeNet, train
    lr, num_epochs = 0.9, 20

    net = LeNet
    train(net, res_train_iter, mnist_test_iter, num_epochs, lr, device = my_utils.try_gpu())

    # print(f"res_train_data shape: {res_train_data.shape}")
    # dataset_augment(mnist_trainset, each_class_feature)
