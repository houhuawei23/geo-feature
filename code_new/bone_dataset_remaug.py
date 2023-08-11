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
    print("eucli_distance_all")
    edist = cdist(mat, mat, metric='euclidean')
    print("eucli_distance_all done")
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
    print("knn_eucli_distance, k = ", k)
    nedist = neigh.kneighbors_graph(mat, k, mode='distance').toarray()
    print("knn_eucli_distance done, k = ", k)
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
    mat: the euclidean distance matrix of the k nearest neighbours of each data point
    gdist: the geodesic distance matrix of the k nearest neighbours of each data point
    predecessors: the predecessors matrix of the k nearest neighbours of each data point
    """
    print("gdist_appro")
    mat = csr_matrix(mat)
    gdist, predecessors = shortest_path(
        csgraph=mat, method='D', directed=False, indices=None, return_predecessors=True)
    print("gdist_appro done")
    return gdist, predecessors


"""
#Test
mat = np.array([[0,0],[0.5,0],[0,0.5],[3,3],[2.5,3],[3,2.5],[3,8],[3,7.5]])
m = knn_eucli_distance(mat, 3)
print(gdist_appro(m))
"""

# Reconstrcut the shortest_path from the predecessors matrix


def path_node(pre_mat, start_node, goal_node):
    # print("path_node")
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
    print("path_aveegr")
    num = np.shape(edist)[0]
    ave_egr = np.zeros((num, num))
    path_hop = np.ones((num, num))
    path_dict = {}
    for i in range(num):  # 这个地方需要-1？
        for j in range(i+1, num):
            if (predecessors[i][j] != -9999):
                node_list, length = path_node(predecessors, i, j)
                node_num = len(node_list) - 2
                path_hop[i][j] = max(length, 1)
                if (node_num == 0):
                    ave_egr[i][j] = 1
                    path_dict[str(i)+'-'+str(j)] = [node_list, length]
                else:
                    egr = 0
                    for k in range(node_num):
                        egr = egr + \
                            (edist[node_list[k]][node_list[k+2]] /
                             gdist[node_list[k]][node_list[k+2]])
                    ave_egr[i][j] = egr / node_num
                    path_dict[str(i)+'-'+str(j)] = [node_list, length]

    # print(len(path_dict))
    print("path_aveegr done")
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


def bone_weight(path_dict, path_index):
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


# Tag the possible removable data point
def dataset_compression_index(ave_egr, path_dict, gdist,
                              unit_hop, ratio, path_index, weight):
    """
input:
    ave_egr: N * N, average edge ratio
    path_dict: (N * N / 2), path_dict['i-j'] = [node_list, length]
    gdist: N * N, geodesic distance
    unit_hop: float, the unit hop
    ratio: float, the ratio of the average edge ratio
    path_index: N * N, the index of the bone path   
    weight: N, the weight of each data sample
    
    """
    # print("dataset_compression_index")
    num = np.shape(ave_egr)[0]
    remove_tag = np.zeros(num)      # remove weight of each data sample
    for i in range(num):
        for j in range(i+1, num):
            if (path_index[i][j] == 1):
                egr = ave_egr[i][j]
                if (egr != 0):
                    node_list = path_dict[str(i)+'-'+str(j)][0]
                    # remove the start point and the goal point
                    node_hops = len(node_list) - 2
                    # egr = ave_egr[i][j]
                    if (egr > ratio):
                        g = gdist[i][j]
                        if (node_hops > (int(g * unit_hop))):
                            seg_egr = []
                            remove_candi = []
                            for k in range(node_hops):
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

    # print("dataset_compression_index done")
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
    print("dataset_augment_index")
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
    print("dataset_augment_index done")
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
    rem_data = []
    topk = int(np.shape(dataset)[0] * percentage)
    
    max_num_index = heapq.nlargest(topk, range(len(remove_tag)), remove_tag.take)
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


def hyper_computation(dataset_name, knn, label, share_dict):
    print("hyper_computation")
    hyper_uhop = []
    hyper_ratio = []
    try:
        dataset = dv.class_read(dataset_name, label)
        # dataset = dv.class_read('fashion_mnist', label)
        edist = eucli_distance_all(dataset)
        knn_edist = knn_eucli_distance(dataset, knn)
        
        gdist, predecessors = gdist_appro(knn_edist)

        path_dict, ave_egr, path_hop = path_aveegr(edist, gdist, predecessors)

        path_index = bone_path(path_dict, gdist)

        weight = bone_weight(path_dict, path_index)

        bave_egr = np.multiply(path_index, ave_egr)

        hyper_ratio.append([bave_egr.min(), bave_egr.mean(), bave_egr.max()])

        hop = gdist / path_hop

        bhop = np.multiply(path_index, hop)

        hyper_uhop.append([bhop.min(), bhop.mean(), bhop.max()])

        share_dict[label] = [dataset, ave_egr, path_dict, edist,
                            gdist, path_index, weight, hyper_ratio, hyper_uhop]
    except:
        print("Error: hyper_compution s", label)
        exit()
"""
dataset: N * (28 * 28)

edist: N * N, euclidean distance

knn_edist: N * N, k nearest neighbor
    if (i, j) is neighbor, knn_edist[i][j] = edist[i][j]
    else, knn_edist[i][j] = 0

gdist: N * N, geodesic distance

predecessors: N * N, predecessors, 前一个节点

path_dict: (N * N / 2), path_dict['i-j'] = [node_list, length]
    node_list: [i, ..., j]
    length: length of path

ave_egr: N * N, average edge ratio

path_hop: N * N, path hop, number of nodes in path

path_index: N * N, path index
    if (i, j) is bone path, path_index[i][j] = 1
    else, path_index[i][j] = 0

weight: N, bone weight for each node

bave_egr: N * N, bone average edge ratio
    if (i, j) is bone path, bave_egr[i][j] = ave_egr[i][j]
    else, bave_egr[i][j] = 0

hyper_ratio: 1 * 1 * 3, [[min, mean, max]]  of bave_egr

hop = gdist / path_hop: N * N, average hop for each node

bhop: N * N, bone hop

hyper_uhop: 1 * 1 * 3, [[min, mean, max]] of bhop

return: pdict[i]
    0: dataset
    1: ave_egr
    2: path_dict
    3: edist
    4: gdist
    5: path_index
    6: weight
    7: hyper_ratio
    8: hyper_uhop

"""
def whole_remove(dataset, 
                 ave_egr, 
                 path_dict, 
                 gdist, 
                 path_index, 
                 weight, 
                 unit_hop, 
                 ratio, 
                 percentage):

    remove_tag = dataset_compression_index(
        ave_egr, path_dict, gdist, unit_hop, ratio, path_index, weight)
    # print("remove_tag")
    # print(remove_tag)
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
