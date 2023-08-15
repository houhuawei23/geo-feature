import os
import cProfile

import numpy as np

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from geo import *

from datasets.mnist import MNIST

curfile_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(curfile_dir, "..", "dataset")
results_dir = os.path.join(curfile_dir, "results")

from sklearn.datasets import make_swiss_roll
import mpl_toolkits.mplot3d  # noqa: F401
from app_utils import class_compress_beta
import matplotlib.cm as cm


def test_swissroll():
    n_samples = 400
    noise = 0.05
    X, _ = make_swiss_roll(n_samples, noise=noise)
    # normalize
    X = (X - X.mean()) / X.std()
    label = 0
    k = 5
    feature = get_class_geo_feature_beta(label, X, k)

    rate = 0.2
    aegr_thres = 0.8 # 越平滑，aegr普遍越大
    removed_index = class_compress_beta(X, feature, rate, aegr_thres)
    removed_data = X[removed_index]
    res_X = np.delete(X, removed_index, axis=0)
    # new_X = np.vstack((res_X, removed_data))
    # new_y = np.hstack((np.zeros(res_X.shape[0]), np.ones(removed_data.shape[0])))

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection="3d", elev=7, azim=-80)
    # ax1.scatter(X[:, 0], X[:, 1], X[:, 2], color="r", edgecolor="k")
    ax1.scatter(res_X[:, 0], res_X[:, 1], res_X[:, 2], color="g", edgecolor="k")
    ax1.scatter(
        removed_data[:, 0],
        removed_data[:, 1],
        removed_data[:, 2],
        color="b",
        edgecolor="k",
    )
    # draw path
    # TODO
    
    # num_paths = 5
    # ij_list = list(get_sorted_2did(feature["path_len"], reverse=True))[:num_paths]
    # colors = cm.viridis(np.linspace(0, 1, num_paths))
    # for idx, (i, j )in enumerate(ij_list):
    #     path = get_shortest_path(feature["pre_matrix"], i, j)
    #     path_coords = X[path]

    #     # 绘制路径
    #     ax1.plot(
    #         path_coords[:, 0],
    #         path_coords[:, 1],
    #         path_coords[:, 2],
    #         color=colors[idx],
    #         linewidth=2,
    #         marker="o",
    #         markersize=5,
    #     )
    
    num_nodes = 5
    ij_list = list(get_sorted_2did(feature['path_aegr'], ignore_val=[0.0, 1.0], reverse=True))[:num_nodes]
    colors = cm.viridis(np.linspace(0, 1, num_nodes))
    for idx, (i, j)in enumerate(ij_list):
        path = get_shortest_path(feature["pre_matrix"], i, j)
        path_coords = X[path]

        # 绘制路径
        ax1.plot(
            path_coords[:, 0],
            path_coords[:, 1],
            path_coords[:, 2],
            color=colors[idx],
            linewidth=2,
            marker="o",
            markersize=5,
            label=f"{i}-{j}-{feature['path_aegr'][i][j]:.2f}"
        )
        
        
        
    ax1.legend(loc='best')
    plt.show()
    
    
    
    
    
    print("done")


def test_data_compress():
    mnist_train_ds = MNIST(dataset_dir, train=True, download=False)
    train_size = 5000
    k = 15
    sample_index = np.random.choice(range(train_size), train_size, replace=False)
    mnist_data_arr = mnist_train_ds.data[sample_index].reshape(train_size, -1)
    mnist_label_arr = mnist_train_ds.targets[sample_index]

    mnist_data_arr = (mnist_data_arr - mnist_data_arr.mean()) / mnist_data_arr.std()

    classified_dataset = get_classified_dataset(mnist_data_arr, mnist_label_arr)

    label = 0
    X = classified_dataset[label]
    print(f"X.shape: {X.shape}")
    feature = get_class_geo_feature_beta(label, X, k)

    # print(f"mnist_data_arr.shape: {mnist_data_arr.shape}")
    # print(f"mnist_label_arr.shape: {mnist_label_arr.shape}")
    # mnist_label_arr2 = mnist_label_arr.reshape(train_size)
    # print(f"mnist_label_arr.shape: {mnist_label_arr2.shape}")
    from app_utils import class_compress_beta

    res_X, removed_data = class_compress_beta(X=X, feature=feature)

    print(f"res_X.shape: {res_X.shape}")
    print(f"removed_data.shape: {removed_data.shape}")

    # visualize


def test_get_class_geo_featur_beta():
    dataset_name = "mnist"
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            # normalize the data to mean = 0, std = 1
            transforms.Normalize((0.5,), (0.5,), inplace=True),
        ]
    )  # 仅在通过mnist_trainset[0]访问数据时，才会进行数据变换
    mnist_trainset = torchvision.datasets.MNIST(
        root=dataset_dir, train=True, download=False, transform=trans
    )
    mnist_data_arr = mnist_trainset.data.numpy()
    mnist_label_arr = mnist_trainset.targets.numpy()

    train_size = 10000
    k = 15
    sample_index = np.random.choice(range(train_size), train_size, replace=False)
    mnist_data_arr = mnist_data_arr[sample_index].reshape(train_size, -1)
    # normalize
    mnist_data_arr = (mnist_data_arr - mnist_data_arr.mean()) / mnist_data_arr.std()
    # 对每张图片/样本进行标准化，还是对整个数据集进行标准化？？可以测试一下
    mnist_label_arr = mnist_label_arr[sample_index]

    classified_dataset = get_classified_dataset(mnist_data_arr, mnist_label_arr)

    label = 0
    X = classified_dataset[label]

    feature = get_class_geo_feature_beta(label, X, k)

    # save
    import pickle

    save_fname = f"mnist_sz{train_size}_k{k}_label{label}_feature.pkl"
    save_dir = os.path.join(results_dir, "features")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, save_fname), "wb") as f:
        pickle.dump(feature, f)
    print(f"feature.keys: {feature.keys()}")

    #


def test_geo():
    dataset_name = "mnist"

    mnist_trainset = torchvision.datasets.MNIST(
        root=dataset_dir, train=True, download=False, transform=transforms.ToTensor()
    )
    mnist_data_arr = mnist_trainset.data.numpy()
    mnist_label_arr = mnist_trainset.targets.numpy()

    train_size = 10000
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
    plot_bar(
        euclidean_dist.reshape(-1),
        20,
        title="euclidean_dist",
        dir=results_dir,
        fname="euclidean_dist.png",
    )

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

    geo_dist, pre_matrix = get_geo_dist(knn_edist)

    print(f"geodesic_dist zero count: {(geo_dist < 1e-5).sum()} / {geo_dist.size}")
    print(f"gdist inf count: {(geo_dist == np.inf).sum()} / {geo_dist.size}")
    print(f"pre_matrix -9999 count: {(pre_matrix == -9999).sum()} / {pre_matrix.size}")
    # plt_distribution_bar(geodesic_dist.reshape(-1), 20, title="geodesic_dist", dir=results_dir, fname="geodesic_dist.png")

    path_ave_egr, path_len = get_path_ave_egr(euc_dist, pre_matrix)
    print(
        f"path_ave_egr zero count: {(path_ave_egr < 1e-5).sum()} / {path_ave_egr.size}"
    )
    # path_len: 集中于 5 ~ 6, 最长到 12 （k=5)
    # plt_distribution_bar(path_len.reshape(-1), 10, title="path_len", dir=results_dir, fname="path_len.png")
    # for key in path_dict.keys():
    #     print(f"key: {key}, value: {path_dict[key]}")
    # plt_distribution_bar(path_ave_egr.reshape(-1), 20, title="path_ave_egr", dir=results_dir, fname="path_ave_egr.png")
    # plt_distribution_bar(path_hop.reshape(-1), 20, title="path_hop", dir=results_dir, fname="path_hop.png")

    bone_path_mask = get_bone_path(size, path_len, pre_matrix)
    # print("bone_path_mask:\n", bone_path_mask)
    # 绝大部分都是“骨架”路径？？
    print(
        f"bone_path_mask True count: {(bone_path_mask == True).sum()} / {bone_path_mask.size}"
    )
    bone_ave_egr = np.multiply(bone_path_mask, path_ave_egr)
    # print(bone_path_mask.shape)
    class_ave_egr = np.array([bone_ave_egr.mean(), bone_ave_egr.max()])  # min always 0
    # 多少算稀疏？？half??
    geo_dist = np.where(geo_dist < 1e-6, 1, geo_dist)

    unit_len = path_len / geo_dist

    bone_ulrn = np.multiply(bone_path_mask, unit_len)

    class_uhop = np.array([bone_ulrn.mean(), bone_ulrn.max()])  # min always 0


# draw the bar chart of the dataarr
def plot_bar(
    data: np.ndarray = None,
    num_bins: int = 10,
    xlabel: str = "Value",
    ylabel: str = "Frequency",
    title: str = "Histogram of the data",
    dir: str = os.path.join(results_dir, "hist"),
    fname: str = "histogram.png",
    ignore_zero: bool = True,
):
    if data is None:
        assert False, "data is None"
    # 去除0值
    if ignore_zero:
        data = data[np.abs(data) > 1e-5]
    # 检查是否所有的数值都是整数，且在有限范围内
    unique_data = np.unique(data)
    if np.all(np.equal(np.mod(unique_data, 1), 0)) and unique_data.size < 100:
        bins = np.arange(unique_data.min(), unique_data.max() + 2) - 0.5
    else:
        bins = num_bins

    # 绘制数据分布图
    fig, ax = plt.subplots()

    # 创建直方图
    n, bins, patches = ax.hist(data, bins=bins, edgecolor="black", alpha=0.7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # 添加网格线
    ax.grid(axis="y", alpha=0.5)

    # 添加图例
    # if isinstance(bins, int):#??
    #     ax.legend([f"Bin {i+1}" for i in range(bins)])
    # else:
    #     ax.legend([f"Value {int(b)}" for b in bins[:-1]])

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

    plt.tight_layout()  # 自动调整布局，防止标签被截断

    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(os.path.join(dir, fname))
    # plt.show()
    plt.close()


if __name__ == "__main__":
    test_swissroll()
    # test_data_compress()
    # test_get_class_geo_featur_beta()
    # # test_get_class_geo_featur_beta()
    # test_statement = "test_get_class_geo_featur_beta()"
    # profile_dir = os.path.join(results_dir, "profile")
    # if not os.path.exists(profile_dir):
    #     os.makedirs(profile_dir)
    # fname = "test_get_class_geo_featur_beta.txt"
    # fpath = os.path.join(profile_dir, fname)

    # cProfile.run(test_statement, fpath, sort='cumtime')

    # import pstats
    # from pstats import SortKey

    # p = pstats.Stats(fpath)

    # p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(25)
    # # test_geo()
