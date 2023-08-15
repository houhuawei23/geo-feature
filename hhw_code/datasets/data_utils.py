import os

from torch.utils import data
import torchvision
import torchvision.transforms as transforms

from utils import get_num_cores

import datasets

curfile_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(curfile_dir, "..", "dataset")


def load_data_mnist(batch_size, root=os.path.join("..", "dataset")):
    trans = transforms.ToTensor()

    mnist_train = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=trans
    )
    mnist_test = torchvision.datasets.MNIST(
        root=root, train=False, download=True, transform=trans
    )

    return (
        data.DataLoader(
            mnist_train, batch_size, shuffle=True, num_workers=get_num_cores()
        ),
        data.DataLoader(
            mnist_test, batch_size, shuffle=False, num_workers=get_num_cores()
        ),
    )


# 数据可视化




if __name__ == "__main__":
    pass
    # train_iter, test_iter = load_data_mnist(32, dataset_dir)
    # print(len(train_iter))
    # print(len(test_iter))
    # mnist_ds = MNIST_DS(root=dataset_dir,train=True)

    # download_root = os.path.join("dataset", "MNIST", "tmp")
    # url = "http://yann.lecun.com/exdb/mnist/"
    # filename = "train-images-idx3-ubyte.gz"

    # furl = f"{url}{filename}"

    # _urlretrieve(furl, os.path.join(download_root, filename))

    # mnist_train_ds = MNIST(root=dataset_dir, train=True, download=False)
    # mnist_test_ds = MNIST(root=dataset_dir, train=False, download=False)

    # print(mnist_train_ds)
    # print(mnist_test_ds)
