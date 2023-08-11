import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

from torch.utils.data import Subset # for sub dataset
from geo import get_classified_dataset, get_dataset_geo_feature
from app_utils import dataset_compress

import network


cwd = os.getcwd()
dataset_dir = os.path.join(cwd, '..', 'dataset')

transform1 = transforms.Compose([
    transforms.ToTensor(),
    lambda image: image.reshape(-1)
])



mnist_trainset = torchvision.datasets.MNIST(
    root=dataset_dir, train = True, download = False, transform = transform1)

mnist_testset = torchvision.datasets.MNIST(
    root = dataset_dir, train = False, download = True, transform = transform1)

train_size = 10000
test_size = 1000

mnist_trainset = Subset(mnist_trainset, range(train_size))
mnist_testset = Subset(mnist_testset, range(test_size))


batch_size = 256



from dataset import get_num_cores
origin_train_iter = data.DataLoader(mnist_trainset, batch_size, shuffle=True, num_workers=get_num_cores())
test_iter = data.DataLoader(mnist_testset, batch_size, shuffle=False, num_workers=get_num_cores())

lr, num_epochs = 1.3, 80
device = network.try_gpu()
origin_lenet = network.LeNet
res_lenet = network.LeNet

origin_lenet = network.train(origin_lenet, origin_train_iter, test_iter, num_epochs, lr, device)
