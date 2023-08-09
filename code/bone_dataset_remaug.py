from scipy.spatial.distance import cdist, euclidean
from sklearn.neighbors import NearestNeighbors
import sklearn.neighbors as neigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import numpy as np
import heapq
import dataset_visual as dv

# compute the euclidean distances of all the data points in the dataset


def eucli_distance_all(mat):
    """
    Time: O(D * N^2)
    """
    edist = cdist(mat, mat, metric='euclidean')
    return edist


"""
#Test 
mat = np.array([[1,2,3],[2,3,4],[3,4,5],[0.5,0.3,0.2]])
m = eucli_distance_all(mat)
print(m)
"""

# compute the euclidean distances of the k nearest neighbours of each data point
# To approximate geodesic distance locallly.


def knn_eucli_distance(mat, k):
    """
    Time: O(D * N * logN) ~ O(D * N^2) 
    """
    nedist = neigh.kneighbors_graph(mat, k, mode='distance').toarray()
    return nedist


"""
#Test
mat = np.array([[0,0],[0.5,0],[0,0.5],[3,3],[2.5,3],[3,2.5],[3,8],[3,7.5]])
m = knn_eucli_distance(mat, 3)
print(m)
"""

# Compute the shortest path as the approximation of the gloabal geodesic distance
# With Dijkstra Algorithm.


def gdist_appro(mat):
    """
    Time: O(N * N * logN)
    """
    mat = csr_matrix(mat)
    gdist, predecessors = shortest_path(
        csgraph=mat, method='D', directed=False, indices=None, return_predecessors=True)
    return gdist, predecessors


"""
method : string ['auto'|'FW'|'D'], optional
    Algorithm to use for shortest paths. Options are:

       'auto' -- (default) select the best among 'FW', 'D', 'BF', or 'J'
                 based on the input data.

       'FW' -- Floyd-Warshall algorithm. 
                 Computational cost is approximately O[N^3]. 
                 The input csgraph will be converted to a dense representation.

       'D' -- Dijkstra's algorithm with Fibonacci heaps. 
                 Computational cost is approximately O[N(N*k + N*log(N))], where k is the average number of connected edges per node. 
                 The input csgraph will be converted to a csr representation.

       'BF' -- Bellman-Ford algorithm. 
                 This algorithm can be used when weights are negative. 
                 If a negative cycle is encountered, an error will be raised. 
                 Computational cost is approximately O[N(N^2 k)], where k is the average number of connected edges per node. 
                 The input csgraph will be converted to a csr representation.

       'J' -- Johnson's algorithm. 
                 Like the Bellman-Ford algorithm, Johnson's algorithm is designed for use when the weights are negative. 
                 It combines the Bellman-Ford algorithm with Dijkstra's algorithm for faster computation.
"""
"""
#Test
mat = np.array([[0,0],[0.5,0],[0,0.5],[3,3],[2.5,3],[3,2.5],[3,8],[3,7.5]])
m = knn_eucli_distance(mat, 3)
print(gdist_appro(m))
"""

# Reconstrcut the shortest_path from the predecessors matrix


def path_node(pre_mat, start_node, goal_node):
    node_list = []
    node_list.append(start_node)
    pre_node = pre_mat[goal_node][start_node]

    while (goal_node != pre_node):
        node_list.append(pre_node)
        pre_node = pre_mat[goal_node][pre_node]

    node_list.append(goal_node)
    length = len(node_list)
    return node_list, length


"""
#Test1
mat = np.array([[0,0],[1,0],[1,-1],[2,-1],[2,-2],[3,-2],[3,-3]])
m = knn_eucli_distance(mat,2)
#print(m)
gdist,pre_mat = gdist_appro(m)
print(gdist, pre_mat)
print(gdist, path_node(pre_mat, 1, 2))
"""

"""
#Test2
mat = np.array([[0,0],[0.5,0],[0,0.5],[3,3],[2.5,3],[3,2.5],[3,8],[3,7.5]])
m = knn_eucli_distance(mat,3)
print(m)
gdist,pre_mat = gdist_appro(m)
print(gdist, pre_mat)
print(gdist, path_node(pre_mat, 0,6))
"""


def path_aveegr(edist, gdist, predecessors):
    """
    Time: O(N^3)
    """
    num = np.shape(edist)[0]
    ave_egr = np.zeros((num, num))
    path_hop = np.ones((num, num))
    path_dict = {}
    for i in range(num):  # 这个地方需要-1？ N
        for j in range(i+1, num):  # N
            #
            if (predecessors[i][j] != -9999):
                node_list, length = path_node(predecessors, i, j)  # 从i到j的最短路径
                # node_list 最短路径上的节点
                node_num = len(node_list) - 2
                path_hop[i][j] = max(length, 1)
                if (node_num == 0):  # 邻接
                    ave_egr[i][j] = 1
                    path_dict[str(i)+'-'+str(j)] = [node_list, length]
                else:  # 非邻接
                    egr = 0
                    for k in range(node_num):  # 遍历node_list N worst
                        egr = egr + \
                            (edist[node_list[k]][node_list[k+2]] /
                             gdist[node_list[k]][node_list[k+2]])
                    ave_egr[i][j] = egr / node_num
                    path_dict[str(i)+'-'+str(j)] = [node_list, length]

    # print(len(path_dict))
    return path_dict, ave_egr, path_hop


"""
#Test
test_data = np.array([[-1,0],[1,0],[1,-1],[1.5,-1]])
edist = eucli_distance_all(test_data)
print("edist", edist)
knn_edist = knn_eucli_distance(test_data, 1)
print("knn-edist", knn_edist)
gdist, predecessors = gdist_appro(knn_edist)
print("gdist", gdist)
dict, ave_egr = path_aveegr(edist, gdist, predecessors)
print(dict, ave_egr)
"""


def bone_path(path_dict, gdist):
    """
path_dict: (N * N / 2), path_dict['i-j'] = [node_list, length]
    node_list: [i, ..., j]
    length: length of path
gdist: N * N, geodesic distance

    Time: O(N^3) ~ O(N^4) ?
    """
    num = np.shape(gdist)[0]  # N
    path_index = - np.ones((num, num))
    # print((num*num-num)/2)
    # 按照path的长度排序，从长到短
    sort_path = sorted(path_dict.items(),
                       key=lambda x: x[1][1], reverse=True)  # N^2

    for _, m_path in (sort_path):  # N^2
        m_list = m_path[0]
        m_len = m_path[1]
        # alraedy in other path
        if ((path_index[m_list[0]][m_list[m_len-1]] == 0) or (path_index[m_list[m_len-1]][m_list[0]] == 0)):
            continue
        else:  # not in other path
            for i in range(m_len):  # N worst
                for j in range(i, m_len):  # N worst
                    path_index[m_list[i]][m_list[j]] = 0
                    path_index[m_list[j]][m_list[i]] = 0
            path_index[m_list[0]][m_list[m_len - 1]] = 1
            path_index[m_list[m_len - 1]][m_list[0]] = 1

    # print(np.sum(path_index == 1))
    # print(np.sum(path_index == -1))
    # print(np.sum(path_index == 0))
    # print('sum: ', np.sum(path_index == 1)+np.sum(path_index == -1)+np.sum(path_index == 0))
    return path_index


def bone_weight(path_dict, path_index):
    """
    path_dict: (N * N / 2), path_dict['i-j'] = [node_list, length]
        node_list: [i, ..., j]
        length: length of path
    path_index: N * N, 1: in bone, 0: in path, -1: not in path
    Time: O(N^3) worst
    """
    num = np.shape(path_index)[0]
    weight_node = np.zeros(num)
    for i in range(num):  # N
        for j in range(i+1, num):  # N / 2
            if (path_index[i][j] == 1):
                node_list = path_dict[str(i)+'-'+str(j)][0]
                # print(node_list)
                for node in node_list:  # N worst
                    weight_node[node] = weight_node[node] + 1
    return weight_node


# Tag the possible removable data point
def dataset_compression_index(ave_egr, path_dict, gdist, unit_hop, ratio, path_index, weight):
    """
    ave_egr: N * N, average edge ratio
    path_dict: (N * N / 2), path_dict['i-j'] = [node_list, length]
    gdist: N * N, geodesic distance
    unit_hop: float, unit hop
    ratio: float, ratio of edge ratio
    path_index: N * N, 1: in bone, 0: in path, -1: not in path
    weight: N, weight of each data sample

    Time: O(N^3) worst
    """
    num = np.shape(ave_egr)[0]
    remove_tag = np.zeros(num)      # remove weight of each data sample
    # 遍历每一条path[i][j]
    for i in range(num):  # N
        for j in range(i+1, num):  # N / 2
            
            if (path_index[i][j] == 1): # in bone
                egr = ave_egr[i][j]
                if (egr != 0): 
                    node_list = path_dict[str(i)+'-'+str(j)][0]
                    # remove the start point and the goal point
                    node_hops = len(node_list) - 2
                    # egr = ave_egr[i][j]
                    if (egr > ratio): # egr > ratio threshold 曲率足够大

                        g = gdist[i][j]
                        if (node_hops > (int(g * unit_hop))): # 路径长度足够长
                            seg_egr = []
                            remove_candi = []
                            for k in range(node_hops):  # N worst
                                seg_egr.append(
                                    ave_egr[node_list[k]][node_list[k+2]])
                                remove_candi.append(node_list[k+1])

                            seg_egr = np.array(seg_egr)
                            minimal_index = np.argsort(seg_egr)
                            for n in range((node_hops - (int(g * unit_hop)))):
                                index = minimal_index[n]
                                remove_tag[remove_candi[index]
                                           ] = remove_tag[remove_candi[index]] + 1
    for i in range(num):
        remove_tag[i] = remove_tag[i]/weight[i]
    return remove_tag


"""
#Test
test_data = np.array([[0,0],[1,0],[1,-3],[2,-3],[2,-4],[3,-4],[4,-4],[5,-4],[5,-5],[6,-5],[6,-6],[10,-6]])
edist = eucli_distance_all(test_data)
#print("edist", edist)
knn_edist = knn_eucli_distance(test_data, 2)
#print("knn-edist", knn_edist)
gdist, predecessors = gdist_appro(knn_edist)
#print("gdist", gdist)
path_dict, ave_egr =  path_aveegr(edist, gdist, predecessors)
remove_tag = dataset_compression_index(ave_egr, path_dict, gdist, 0.4, 0.5)
print(remove_tag)
"""

# Tag the possible additional data point pair, Yes, it's a pair.


def dataset_augment_index(ave_egr, path_dict, gdist, unit_hop, ratio, path_index, weight):
    num = np.shape(ave_egr)[0]
    add_tag = np.zeros((num, num))      # add weight between two data samples

    for i in range(num):
        for j in range(i+1, num):
            if (path_index[i][j] == 1):
                egr = ave_egr[i][j]
                if (egr != 0):
                    tmp = str(i)+'-'+str(j)
                    node_list = path_dict[tmp][0]
                    # remove the start point and the goal point，and the successor
                    node_hops = len(node_list) - 4
                    if (egr < ratio):
                        g = gdist[i][j]
                        if (node_hops < (int(g * unit_hop))):
                            seg_egr = []
                            add_candi = []
                            add_node = []
                            for k in range(node_hops):
                                seg_egr.append(
                                    ave_egr[node_list[k+1]][node_list[k+3]])
                                add_candi.append(
                                    [node_list[k], node_list[k+4]])
                                add_node.append(node_list[k+2])

                            seg_egr = np.array(seg_egr)
                            maximal_index = np.argsort(-seg_egr)
                            for n in range(min(node_hops, (int(g * unit_hop)) - node_hops)):
                                index = maximal_index[n]
                                add_tag[add_candi[index][0]][add_candi[index][1]
                                                             ] = add_tag[add_candi[index][0]][add_candi[index][1]] + 1

    return add_tag


"""
#Test
test_data = np.array([[0,0],[1,0],[1,-3],[2,-3],[2,-4],[3,-4],[4,-4],[5,-4],[5,-5],[6,-5],[6,-6],[10,-6]])
edist = eucli_distance_all(test_data)
#print("edist", edist)
knn_edist = knn_eucli_distance(test_data, 2)
#print("knn-edist", knn_edist)
gdist, predecessors = gdist_appro(knn_edist)
#print("gdist", gdist)
#print(path_node(predecessors, 1, 4))
path_dict, ave_egr =  path_aveegr(edist, gdist, predecessors)
add_tag = dataset_augment_index(ave_egr, path_dict, gdist, 0.8, 0.9)
print(add_tag)
"""


def dataset_compress(dataset, remove_tag, percentage):
    """
    Time: O(N log topk)
    """
    rem_data = []
    topk = int(np.shape(dataset)[0] * percentage)
    max_num_index = heapq.nlargest(
        topk, range(len(remove_tag)), remove_tag.take)
    max_num_index.sort(reverse=True)
    for idx in (max_num_index):
        rem_data.append(dataset[idx])
        dataset = np.delete(dataset, idx, axis=0)
    return dataset, rem_data


"""
# Test
dataset = np.array([[1,2],[2,3],[3,4],[5,6],[6,7]])
remove_tag = np.array([1,5,2,4,1])
percentage = 0.4
print( dataset_compress(dataset, remove_tag, percentage) )
"""


def interpolation_optimize(dataset, row, col, path_dict, edist, sam_num):
    tmp = 0
    if (row > col):
        tmp = str(col) + '-' + str(row)
    else:
        tmp = str(row) + '-' + str(col)
    # print(str(row),str(col))
    # print(row, col, tmp)
    node_list = path_dict[tmp][0]
    # print(node_list)
    imm_node = dataset[node_list[2]]
    pre_node = node_list[0]
    start_node = dataset[row]
    goal_node = dataset[col]
    next_node = dataset[node_list[4]]
    edge_a = start_node - imm_node
    edge_b = goal_node - imm_node
    norm_a = np.sqrt(edge_a.dot(edge_a))
    norm_b = np.sqrt(edge_b.dot(edge_b))
    cos_angle = (edge_a.dot(edge_b))/(norm_a * norm_b)
    sin_angle = np.sqrt(1-(cos_angle * cos_angle))

    radius = min(norm_a, norm_b)

    lam_max = radius / (sin_angle * norm_a)
    mu_max = radius / (sin_angle * norm_b)

    new_data = []
    egr = 0
    i = 0
    sample_num = 0
    while (i <= sam_num and sample_num <= 10000):
        lam = np.random.uniform(-lam_max, lam_max*(1.001))
        mu = np.random.uniform(-mu_max, mu_max*(1.001))
        sample = lam * edge_a + mu * edge_b
        sample_num = sample_num + 1

        if ((lam * mu) <= 0 and (np.sqrt(sample.dot(sample)) <= radius)):
            new_node = imm_node + sample
            e_sn = euclidean(start_node, new_node)
            e_mn = euclidean(imm_node, new_node)
            e_ng = euclidean(new_node, goal_node)
            e_nx = euclidean(new_node, next_node)
            e_pn = euclidean(pre_node, new_node)
            e_sm = edist[row][node_list[2]]
            e_mg = edist[node_list[2]][col]
            e_gx = edist[col][node_list[4]]
            e_ps = edist[node_list[0]][row]

            if (lam <= 0):
                avg_egr = (e_sn / (e_sm + e_mn) + e_mg /
                           (e_ng + e_mn) + e_nx / (e_ng + e_gx)) / 3
            else:
                avg_egr = (e_sm / (e_sn + e_mn) + e_ng /
                           (e_mg + e_mn) + e_pn / (e_ps + e_sn)) / 3

            if (avg_egr > egr):
                new_data = new_node
                egr = avg_egr
            i = i + 1
    if (sample_num > 10000 and i < 500):
        new_data = []
    return new_data


"""
#Test
test_data = np.array([[0,0],[1,0],[1,-3],[2,-3],[2,-4],[3,-4],[4,-4],[5,-4],[5,-5],[6,-5],[6,-6],[10,-6]])
edist = eucli_distance_all(test_data)
#print("edist", edist)
knn_edist = knn_eucli_distance(test_data, 2)
#print("knn-edist", knn_edist)
gdist, predecessors = gdist_appro(knn_edist)
#print("gdist", gdist)
print(interpolation_optimize(test_data, 1, 3, predecessors, edist, 100, 0.5))
"""


def similar_test(new_data):
    return True


def dataset_augment(dataset, add_tag, percentage, edist, path_dict):
    topk = int(np.shape(dataset)[0] * percentage)
    add_data = []
    num = np.shape(add_tag)[1]
    idx = np.argsort(add_tag, axis=None)
    idx = np.flipud(idx)

    sam_num = 1000
    # radius = 1

    for i in range(topk):
        row = int(idx[i] / num)
        col = int(idx[i] % num)
        if (add_tag[row][col] == 0):
            break
        new_data = interpolation_optimize(
            dataset, row, col, path_dict, edist, sam_num)
        add_flag = similar_test(new_data)
        if (add_flag == True and new_data != []):
            add_data.append(new_data)

    print("Number of new generated data: ", len(add_data))
    if (add_data != []):
        add_data = np.array(add_data)
        dataset = np.vstack((dataset, add_data))
    return dataset, add_data


"""
#Test        
test_data = np.array([[0,0],[1,0],[1,-3],[2,-3],[2,-4],[3,-4],[4,-4],[5,-4],[5,-5],[6,-5],[6,-6],[10,-6]])
edist = eucli_distance_all(test_data)
#print("edist", edist)
knn_edist = knn_eucli_distance(test_data, 2)
#print("knn-edist", knn_edist)
gdist, predecessors = gdist_appro(knn_edist)
#print("gdist", gdist)
path_dict, ave_egr = path_aveegr(edist, gdist, predecessors)
#print("ave_egr", ave_egr)
add_tag = dataset_augment_index(ave_egr,  path_dict, predecessors, 0.8, 0.8)
#print(add_tag)
#print(dataset_augment(test_data, add_tag, 0.1, edist, predecessors))
"""


"""
test_data = np.array([[0,2],[1,6],[7,5]])
print(test_data.shape)
num = np.shape(test_data)[1]
print(num)
idx = np.argsort(test_data, axis=None)
idx = np.flipud(idx)  #从大到小
print(idx)
print("result")
for i in range(num*np.shape(test_data)[0]):
    row = int((idx[i]) / num)
    col = idx[i] % num
    print("idx: ", idx[i], " row: ", row ," col:", col)
    print("data: ", test_data[row][col])
"""


def hyper_computation(knn, label, share_dict):
    hyper_uhop = []
    hyper_ratio = []
    # load which dataset?
    dataset = dv.class_read('fashion_mnist', label)
    # dataset = dv.class_read('fashion_mnist', label)
    edist = eucli_distance_all(dataset)
    # print("edist")
    knn_edist = knn_eucli_distance(dataset, knn)
    # print("knn-edist")
    gdist, predecessors = gdist_appro(knn_edist)
    # print("gdist")
    path_dict, ave_egr, path_hop = path_aveegr(edist, gdist, predecessors)

    path_index = bone_path(path_dict, gdist)
    weight = bone_weight(path_dict, path_index)
    bave_egr = np.multiply(path_index, ave_egr)
    hyper_ratio.append([bave_egr.min(), bave_egr.mean(), bave_egr.max()])
    hop = gdist / path_hop
    bhop = np.multiply(path_index, hop)
    hyper_uhop.append([bhop.min(), bhop.mean(), bhop.max()])
    # print("dict, ave_egr")
    # print("label",label)
    share_dict[label] = [dataset, ave_egr, path_dict, edist,
                         gdist, path_index, weight, hyper_ratio, hyper_uhop]
    # print("share_dict", share_dict)
    # print("label",label)


def whole_remove(dataset, ave_egr, path_dict, gdist, path_index, weight, unit_hop, ratio, percentage):
    # generate remove tag O(N^3) worst
    remove_tag = dataset_compression_index(
        ave_egr, path_dict, gdist, unit_hop, ratio, path_index, weight)
    # print("remove_tag")
    # print(remove_tag)
    # compress dataset O(N log topk)
    rsub_data, rem_data = dataset_compress(dataset, remove_tag, percentage)
    # print("data")
    # print(rsub_data.shape)
    return rsub_data, rem_data


def whole_augment(dataset, ave_egr, path_dict, gdist, edist, path_index, weight, unit_hop, ratio, percentage):

    add_tag = dataset_augment_index(
        ave_egr, path_dict, gdist, unit_hop, ratio, path_index, weight)
    # print("add_tag", add_tag)
    asub_data, add_data = dataset_augment(
        dataset, add_tag, percentage, edist, path_dict)
    # print(np.shape(asub_data))
    return asub_data, add_data
