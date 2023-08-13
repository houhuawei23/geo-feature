import os
import cProfile

import numpy as np

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from geo import *

curfile_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(curfile_dir, "..", "dataset")
results_dir = os.path.join(curfile_dir, "results")


def test_geo():
    dataset_name = "mnist"

    mnist_trainset = torchvision.datasets.MNIST(
        root=dataset_dir, train=True, download=False, transform=transforms.ToTensor()
    )
    mnist_data_arr = mnist_trainset.data.numpy()
    mnist_label_arr = mnist_trainset.targets.numpy()

    train_size = 2000
    k = 15
    sample_index = np.random.choice(range(train_size), train_size, replace=False)
    mnist_data_arr = mnist_data_arr[sample_index].reshape(train_size, -1)
    mnist_label_arr = mnist_label_arr[sample_index]

    classified_dataset = get_classified_dataset(mnist_data_arr, mnist_label_arr)

    # label = 0
    X = classified_dataset[0]
    size = X.shape[0]
    print(f"X.shape: {X.shape}")
    euc_dist = cdist(X, X, metric="euclidean")

    print(f"e_dist zero count: {(euc_dist < 1e-5).sum()} / {euc_dist.size}")
    # plt_distribution_bar(euclidean_dist.reshape(-1), 20, title="euclidean_dist", dir=results_dir, fname="euclidean_dist.png")

    knn_edist = sk_neighbors.kneighbors_graph(
        X, k, mode="distance", n_jobs=-1
    ).toarray()  # 稀疏的

    # for i in range(knn_edist.shape[0]):
    #     # 非零值所在的下标
    #     nonzero_index = knn_edist[i].nonzero()[0]
    #     print(f"row {i} nonzero index: {nonzero_index}")

    print(f"knn_edist zero count: {(knn_edist < 1e-5).sum()} / {knn_edist.size}")
    # print(f"knn_edist inf count: {(knn_edist == np.inf).sum()} / {knn_edist.size}")
    # plt_distribution_bar(knn_edist.reshape(-1), 20, title="knn_edist", dir=results_dir, fname="knn_edist.png")

    geo_dist, pre_matrix = get_geodesic_dist(knn_edist)

    print(f"geodesic_dist zero count: {(geo_dist < 1e-5).sum()} / {geo_dist.size}")
    print(f"gdist inf count: {(geo_dist == np.inf).sum()} / {geo_dist.size}")
    print(f"pre_matrix -9999 count: {(pre_matrix == -9999).sum()} / {pre_matrix.size}")
    # plt_distribution_bar(geodesic_dist.reshape(-1), 20, title="geodesic_dist", dir=results_dir, fname="geodesic_dist.png")

    path_ave_egr, path_len, path_dict = get_path_ave_egr(euc_dist, pre_matrix)
    print(
        f"path_ave_egr zero count: {(path_ave_egr < 1e-5).sum()} / {path_ave_egr.size}"
    )
    # path_len: 集中于 5 ~ 6, 最长到 12 （k=5)
    plt_distribution_bar(path_len.reshape(-1), 10, title="path_len", dir=results_dir, fname="path_len.png")
    # for key in path_dict.keys():
    #     print(f"key: {key}, value: {path_dict[key]}")
    # plt_distribution_bar(path_ave_egr.reshape(-1), 20, title="path_ave_egr", dir=results_dir, fname="path_ave_egr.png")
    # plt_distribution_bar(path_hop.reshape(-1), 20, title="path_hop", dir=results_dir, fname="path_hop.png")

    bone_path_mask = get_bone_path(size, path_dict)
    # print("bone_path_mask:\n", bone_path_mask)
    # 绝大部分都是“骨架”路径？？
    print(f"bone_path_mask True count: {(bone_path_mask == True).sum()} / {bone_path_mask.size}")
    bone_ave_egr = np.multiply(bone_path_mask, path_ave_egr)
    # print(bone_path_mask.shape)
    class_ave_egr = np.array([bone_ave_egr.mean(), bone_ave_egr.max()])  # min always 0
    # 多少算稀疏？？half??
    geo_dist = np.where(geo_dist < 1e-6, 1, geo_dist)
    
    unit_len = path_len / geo_dist

    bone_ulrn = np.multiply(bone_path_mask, unit_len)

    class_uhop = np.array([bone_ulrn.mean(), bone_ulrn.max()])  # min always 0


# draw the bar chart of the dataarr
def plt_distribution_bar(
    data,
    num_bins=10,
    xlabel="Value",
    ylabel="Count",
    title="Data Distribution",
    dir=None,
    fname=None,
):
    data = data[data != 0.0]
    # 绘制数据分布图
    fig, ax = plt.subplots()

    # 创建直方图
    n, bins, patches = ax.hist(data, bins=num_bins, edgecolor="black", alpha=0.7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # 添加网格线
    ax.grid(axis="y", alpha=0.5)

    # 添加图例
    ax.legend([f"Bin {i+1}" for i in range(num_bins)])

    # 添加数值标签
    for i, patch in enumerate(patches):
        height = patch.get_height()
        width = patch.get_x() + patch.get_width() / 2
        percent = height / data.shape[0] * 100
        ax.annotate(
            f"{percent:.1f}%",
            xy=(width, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
        # ax.annotate(f'{int(height)}',
        #             xy=(width, height),
        #             xytext=(0, 3),  # 3 points vertical offset
        #             textcoords='offset points',
        #             ha='center', va='bottom')

    plt.tight_layout()  # 自动调整布局，防止标签被截断

    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(os.path.join(dir, fname))
    plt.show()
    plt.close()


if __name__ == "__main__":
    # cProfile.run("test_geo()", 'restats', sort='cumtime')

    # import pstats
    # from pstats import SortKey

    # p = pstats.Stats('restats')

    # p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(10)
    test_geo()
