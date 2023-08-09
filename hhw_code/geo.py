from multiprocessing.pool import Pool
from multiprocessing import Manager
import numpy as np
import torch
import sklearn.neighbors as sk_neighbors

from scipy.spatial.distance import cdist, euclidean
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def get_geo_feature(label, X, k, shared_dict):
    """
    label: int, the label of the class
    X: torch.Tensor, the images of the class
        shape: (num_images, 1, 28, 28)
    k: int, the number of neighbors
    shared_dict: Manager.dict()
    """
# edist: N * N, euclidean distance
    euclidean_dist = torch.cdist(X, X)
    print('euclidean_dist:\n', euclidean_dist)

# knn_edist: N * N, k nearest neighbor
#   if (i, j) is neighbor, knn_edist[i][j] = edist[i][j]
#   else, knn_edist[i][j] = 0
    knn_edist = get_knn_edist(euclidean_dist, k)
    print('knn_edist:\n', knn_edist)

# gdist: N * N, geodesic distance
# predecessors: N * N, predecessors, 前一个节点
    geodesic_dist, predecessors = get_geodesic_dist(knn_edist)
    print('geodesic_dist:\n')
    for i in range(geodesic_dist.shape[0]):
        print(geodesic_dist[i])

    print('predecessors:\n')
    for i in range(predecessors.shape[0]):
        print(predecessors[i])
# path_ave_egr: N * N, average edge ratio
# path_hop: N * N, path hop, number of nodes in path
# path_dict: (N * N / 2), path_dict['i-j'] = [node_list, length]
    path_ave_egr, path_hop, path_dict = get_path_ave_egr(
        euclidean_dist, geodesic_dist, predecessors)
    
    # print('path_ave_egr:\n')
    # for i in range(path_ave_egr.shape[0]):
    #     print(path_ave_egr[i])

    # print('path_hop:\n', path_hop)
    # # print('path_dict:\n', path_dict)
    # print('path_dict:\n')
    # for key in path_dict:
    #     print(key, path_dict[key])
# path_index: N * N, path_index[i][j] = 1, if i-j is bone
    path_index = bone_path(path_dict, geodesic_dist)
# weight: N, weight[i] = the number of bones that i is in
    weight = bone_weight(path_dict, path_index)

    bone_ave_egr = np.multiply(path_index, path_ave_egr)

    label_egr = (bone_ave_egr.min(), bone_ave_egr.mean(), bone_ave_egr.max())

    bone_hop = np.multiply(path_index, path_hop)

    hop = geodesic_dist / path_hop
    print('hop:\n')
    for i in range(hop.shape[0]):
        print(hop[i])
    # bhop = np.multiply(path_index, hop)

    # hyper_uhop.append([bhop.min(), bhop.mean(), bhop.max()])

    # share_dict[label] = [dataset, ave_egr, path_dict, edist,
    #                     gdist, path_index, weight, hyper_ratio, hyper_uhop]
     
    
    return
def bone_weight(path_dict, path_index):
    """
    input:
        path_dict: (N * N / 2), path_dict['i-j'] = [node_list, length]
        path_index: N * N, path_index[i][j] = 1, if i-j is bone
    output:
        weight_node: N, weight_node[i] = the number of bones that i is in
    """
    print("bone_weight")
    num = np.shape(path_index)[0]
    weight_node = np.zeros(num)
    for i in range(num):
        for j in range(i+1, num):
            if (path_index[i][j] == 1):
                node_list = path_dict[str(i)+'-'+str(j)][0]
                # print(node_list)
                for node in node_list:
                    weight_node[node] = weight_node[node] + 1
    print("bone_weight done")
    return weight_node


def bone_path(path_dict, gdist):
    """
    input:
        path_dict: (N * N / 2), path_dict['i-j'] = [node_list, length]
        gdist: N * N, geodesic distance
    output:
        path_index: N * N, path_index[i][j] = 1, if i-j is bone
    """
    print("bone_path")
    num = np.shape(gdist)[0]
    path_index = - np.ones((num, num))
    # print((num*num-num)/2)
    sort_path = sorted(path_dict.items(), key=lambda x: x[1][1], reverse=True)
    for _, m_path in (sort_path):
        m_list = m_path[0]
        m_len = m_path[1]
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
    print("bone_path done")
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
    path_hop = np.zeros((num, num))
    path_dict = {}
    for i in range(num):
        for j in range(i + 1, num): # 可并行
            if (predecessors[i][j] != -9999):
                node_list, length = path_node(predecessors, i, j)
                node_num = len(node_list) - 2
                path_hop[i][j] = max(length, 1)
                if (node_num == 0):
                    path_ave_egr[i][j] = 1
                    path_dict[str(i)+'-'+str(j)] = [node_list, length]
                else:
                    egr = 0
                    for k in range(node_num):
                        egr = egr + \
                            (euclidean_dist[node_list[k]][node_list[k+2]] /
                             geodesic_dist[node_list[k]][node_list[k+2]])
                    path_ave_egr[i][j] = egr / node_num
                    path_dict[str(i)+'-'+str(j)] = [node_list, length]
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


def get_dataset_geo_feature(dataset, k):
    """
    dataset: torch.utils.data.Dataset
    k: int, the number of neighbors
    """
    # 处理dataset，按标签分类
    u_labels = np.unique(dataset.targets)
    num_classes = len(u_labels)

    dataset_dict = {}
    cnt = 0
    for image, label in dataset:
        cnt += 1
        # print(f'cnt: {cnt}, label: {label}')
        if label not in dataset_dict.keys():
            dataset_dict[label] = []
        dataset_dict[label].append(image)
        if cnt == 70:
            break
    for label in dataset_dict.keys():
        dataset_dict[label] = torch.stack(dataset_dict[label])
    # print('dataset_dict: ', dataset_dict)
    # for label in dataset_dict.keys():
    #     print(f'label: {label}, len(dataset_dict[label]): {len(dataset_dict[label])}')
    # print(type(dataset_dict[0]))

    # 计算每个类别的几何特征
    print(dataset_dict[0].shape)
    get_geo_feature(0, dataset_dict[0], k, None)
    # num_process = num_classes
    # process_pool = Pool(num_process)
    # shared_dict = Manager().dict()
    # for label in dataset_dict.keys():
    #     process_pool.apply_async(get_geo_feature, args=(label, dataset_dict[label], k, shared_dict))
    # process_pool.close()
    # process_pool.join()
    # print(shared_dict.keys())

    # ...

    return


if __name__ == '__main__':
    import os

    import torchvision
    import torchvision.transforms as transforms
    cwd = os.getcwd()
    dataset_dir = os.path.join(cwd, 'dataset')

    def flatten_transform(image):
        return image.view(-1)
    transform = transforms.Compose([
        transforms.ToTensor(),
        flatten_transform
    ])

    mnist_trainset = torchvision.datasets.MNIST(
        root=dataset_dir, train=True, download=True, transform=transform)
    # print(mnist_trainset[0])
    # print(mnist_trainset[0][0].reshape(28 * 28).shape)
    get_dataset_geo_feature(mnist_trainset, 6)
