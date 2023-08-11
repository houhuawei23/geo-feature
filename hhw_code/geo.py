from multiprocessing.pool import Pool
from multiprocessing import Manager
import numpy as np
import torch
import sklearn.neighbors as sk_neighbors

from scipy.spatial.distance import cdist, euclidean
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def get_class_geo_feature(label, X, k, shared_dict):
    print(f"get_class_geo_feature, label: {label}, shape: {X.shape}")
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
    - 'path_dict': (N * N / 2), path_dict['i-j'] = [node_list, length]
    - 'euclidean_dist': N * N, euclidean distance
    - 'geodesic_dist': N * N, geodesic distance
    - 'bone_path_index': N * N, path_index[i][j] = 1, if i-j is bone path
    - 'bone_weight': N, weight[i] = the number of bones that i is in
    - 'label_ave_egr': tuple shape(1, 3), (min, mean, max) of bone_ave_egr
    - 'label_uhop': tuple shape(1, 3), (min, mean, max) of bone_uhop
        
    """
    try:
# edist: N * N, euclidean distance
        print("here")
        print("X.shape: ", X.shape)
        euclidean_dist = torch.cdist(X, X, p=2)
        print(f"label: {label}, euclidean_dist.shape: {euclidean_dist.shape}")
    # print('euclidean_dist:\n', euclidean_dist)

# knn_edist: N * N, k nearest neighbor
#   if (i, j) is neighbor, knn_edist[i][j] = edist[i][j]
#   else, knn_edist[i][j] = 0
        knn_edist = get_knn_edist(euclidean_dist, k)
    # print('knn_edist:\n', knn_edist)

# gdist: N * N, geodesic distance
# predecessors: N * N, predecessors, 前一个节点
        geodesic_dist, predecessors = get_geodesic_dist(knn_edist)
    # print('geodesic_dist:\n')
    # for i in range(geodesic_dist.shape[0]):
    #     print(geodesic_dist[i])

    # print('predecessors:\n')
    # for i in range(predecessors.shape[0]):
    #     print(predecessors[i])
# path_ave_egr: N * N, average edge ratio
# path_hop: N * N, path hop, number of nodes in path
# path_dict: (N * N / 2), path_dict['i-j'] = [node_list, length]
        path_ave_egr, path_hop, path_dict = get_path_ave_egr(euclidean_dist, geodesic_dist, predecessors)

        print("mid")
    # path_ave_egr == 0 means whatt ??

    # print('path_ave_egr:\n')
    # print(path_ave_egr)

    # print('path_hop:\n', path_hop)
    
    # print('path_dict:\n')
    # for key in path_dict:
    #     print(key, path_dict[key])
# path_index: N * N, path_index[i][j] = 1, if i-j is bone
        bone_path_index = bone_path(path_dict, geodesic_dist)
# weight: N, weight[i] = the number of bones that i is in, ???? 具体含义是什么？？
        bone_weight = get_bone_weight(path_dict, bone_path_index)

        bone_ave_egr = np.multiply(bone_path_index, path_ave_egr)

        class_ave_egr = np.array([bone_ave_egr.mean(), bone_ave_egr.max()]) # min always 0
    # 单位长度跳结点数
    # gdist_uhop = geodesic_dist / path_hop
        geodesic_dist = np.where(geodesic_dist < 1e-5, 1, geodesic_dist)
        gdist_uhop = path_hop / geodesic_dist
    # 
    # gdist_uhop[np.isinf(gdist_uhop)] = 0 # 设为多少呢？
        bone_uhop = np.multiply(bone_path_index, gdist_uhop)
    
        class_uhop = np.array([bone_uhop.mean(), bone_uhop.max()] )# min always 0
    except:
        print(f"get_class_geo_feature: {label} failed")
    # print("path_hop")
    # print(path_hop)
    
    # print("geo_dist")
    # print(geodesic_dist)
    
    # print("gdist_uop")
    # print(gdist_uhop)
    
    # print("bone_uhop")
    # print(bone_uhop)
     
    # print(f"begin shared_dict[{label}]")
    feature = {}
    feature['label'] = label
    # feature['data'] = X
    feature['path_avg_egr'] = path_ave_egr
    feature['path_dict'] = path_dict
    # feature['euclidean_dist'] = euclidean_dist  
    feature['geodesic_dist'] = geodesic_dist
    feature['bone_path_index'] = bone_path_index
    feature['bone_weight'] = bone_weight
    feature['class_ave_egr'] = class_ave_egr    
    feature['class_uhop'] = class_uhop
    shared_dict[label] = feature

    print(f"get_class_feature: {label} finished")
    return True

def get_bone_weight(path_dict, path_index):
    """
    input:
        path_dict: (N * N / 2), path_dict['i-j'] = [node_list, length]
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
                node_list = path_dict[str(i)+'-'+str(j)]['node_list']
                # print(node_list)
                for node in node_list:
                    weight_node[node] = weight_node[node] + 1
    # print("get_bone_weight done")
    return weight_node


def bone_path(path_dict, gdist):
    """
    input:
        path_dict: (N * N / 2), path_dict['i-j'] = [node_list, length]
        gdist: N * N, geodesic distance
    output:
        path_index: N * N, path_index[i][j] = 1, if i-j is bone
    """
    # print("bone_path")
    num = np.shape(gdist)[0]
    path_index = - np.ones((num, num))
    # print((num*num-num)/2)
    sort_path = sorted(path_dict.items(), key=lambda x: x[1]['len'], reverse=True) # list

    # [('0-1', [[0, 1], 2]), ('0-2', [[0, 3, 2], 3]), ...]
    # print(type(sort_path))
    for _, m_path in (sort_path):
        m_list = m_path['node_list']
        m_len = m_path['len']
        if ((path_index[m_list[0]][m_list[m_len-1]] == 0) or (path_index[m_list[m_len-1]][m_list[0]] == 0)):
            continue
        else:
            for i in range(m_len):
                for j in range(i, m_len):
                    path_index[m_list[i]][m_list[j]] = 0
                    path_index[m_list[j]][m_list[i]] = 0
            path_index[m_list[0]][m_list[m_len - 1]] = 1
            path_index[m_list[m_len - 1]][m_list[0]] = 1

    # print(np.sum(path_index == 1))
    # print(np.sum(path_index == -1))
    # print(np.sum(path_index == 0))
    # print('sum: ', np.sum(path_index == 1)+np.sum(path_index == -1)+np.sum(path_index == 0))
    # print("bone_path done")
    return path_index

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
                node_list, length = path_node(predecessors, i, j)
                node_num = len(node_list) - 2
                path_hop[i][j] = max(length - 2, 0)
                if (node_num == 0):
                    path_ave_egr[i][j] = 1
                    # path_dict[str(i)+'-'+str(j)] = [node_list, length]
                    path_dict[str(i)+'-'+str(j)] = {'node_list': node_list, 'len': length}
                else:
                    egr = 0
                    for k in range(node_num):
                        egr = egr + \
                            (euclidean_dist[node_list[k]][node_list[k+2]] /
                             geodesic_dist[node_list[k]][node_list[k+2]])
                    path_ave_egr[i][j] = egr / node_num
                    # path_dict[str(i)+'-'+str(j)] = [node_list, length]
                    path_dict[str(i)+'-'+str(j)] = {'node_list': node_list, 'len': length}
    return path_ave_egr, path_hop, path_dict

def path_node(pre_mat, start_node, goal_node):
    """
    input:
        pre_mat: N * N, predecessors, 前一个节点
        start_node: int, start node
        goal_node: int, goal node
    return:
        node_list: list, the path node
        length: int, the length of the path
    """
    
    node_list = []
    node_list.append(start_node)
    pre_node = pre_mat[goal_node][start_node]

    while (goal_node != pre_node):
        node_list.append(pre_node)
        pre_node = pre_mat[goal_node][pre_node]

    node_list.append(goal_node)
    length = len(node_list)
    # print("path_node done")
    return node_list, length


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
    manager = Manager()
    s_each_class_feature = manager.dict()
    num_classes = len(classified_dataset.keys())

# serial
    # for label in classified_dataset.keys():
    #     get_class_geo_feature(label, classified_dataset[label], k, s_each_class_feature)


# serial end

# multi-process
    process_pool = Pool(num_classes)
    for label in classified_dataset.keys():
        # print(f"lable: {label}, size: {classified_dataset[label].shape}")
        process_pool.apply_async(get_class_geo_feature,  
                                 args=(label, 
                                       classified_dataset[label], 
                                       k, 
                                       s_each_class_feature))
    process_pool.close()
    process_pool.join()
# multi-process end

    s_each_class_feature = dict(s_each_class_feature)
    
    # get global dataset egr and uhop
    egr_scope = (0.0, 0.0)
    uhop_scope = (0.0, 0.0)
    for label in s_each_class_feature.keys():
        # print(f"label: {label}")
        egr_scope += s_each_class_feature[label]['class_ave_egr']
        uhop_scope += s_each_class_feature[label]['class_uhop']
    
    egr_scope /= num_classes
    uhop_scope /= num_classes        

    return s_each_class_feature, egr_scope, uhop_scope

from collections import defaultdict
def get_classified_dataset(dataset):
    """
    dataset: torch.utils.data.Dataset
    
    classified_dataset: dict, the dataset classified by label
        type: dict(Tensor(size, features))
    """
    classified_dataset = defaultdict(list)
    for image, label in dataset:
        classified_dataset[label].append(image)
    # print(classified_dataset[0])
    # stack the list:
    for label in classified_dataset.keys():
        classified_dataset[label] = torch.stack(classified_dataset[label])
    
    return dict(classified_dataset)

# def get_merged_dataset()

if __name__ == '__main__':
    import os

    import torchvision
    import torchvision.transforms as transforms
    curfile_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(curfile_dir, '..', 'dataset')

    def flatten_transform(image):
        return image.view(-1)
    transform = transforms.Compose([
        transforms.ToTensor(),
        flatten_transform
    ])

    mnist_trainset = torchvision.datasets.MNIST(
        root=dataset_dir, train=True, download=False, transform=transform)
    
    # print(mnist_trainset[0][0].shape)
    # print(mnist_trainset[0][0].shape)
    # print(mnist_trainset[0][0].reshape(28 * 28).shape)

    # get fist 250 images
    from torch.utils.data import Subset
    mnist_trainset = Subset(mnist_trainset, range(500))
    
    classified_dataset = get_classified_dataset(mnist_trainset)
    each_class_feature, egr_scope, uhop_scope = get_dataset_geo_feature(classified_dataset, k=3)
    
    # from app_utils import dataset_compress, dataset_augment
    # # get global dataset egr and uhop
    # egr_scope = (0.0, 0.0)
    # uhop_scope = (0.0, 0.0)
    # n_classes = 0
    # for label in each_class_feature.keys():
    #     n_classes += 1
    #     # print(f"label: {label}")
    #     egr_scope += each_class_feature[label]['class_ave_egr']
    #     uhop_scope += each_class_feature[label]['class_uhop']
    # print(f"egr: {egr_scope}")
    # print(f"uhop: {uhop_scope}")
        
    # egr_threshold = (egr_scope[0] + egr_scope[1]) / (2 * n_classes)
    # uhop_threshold = (uhop_scope[0] + uhop_scope[1]) / (2 * n_classes)
    # rate = 0.1
    
    # s_res_dataset, s_removed_dataset = dataset_compress(classified_dataset = classified_dataset,
    #                                                     each_class_feature = each_class_feature,
    #                                                     rate = rate, 
    #                                                     egr_threshold = egr_threshold, 
    #                                                     uhop_threshold = uhop_threshold)
    # print(len(s_res_dataset))
    # print(len(s_removed_dataset))
    
    # for label in s_removed_dataset.keys():
    #     print(f"label: {label}")
    #     print(s_removed_dataset[label].shape)
    #     print(s_res_dataset[label].shape)

    
    # dataset_augment(mnist_trainset, each_class_feature)