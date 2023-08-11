import os
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

# import multiprocessing

curfile_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(curfile_dir, '..', 'dataset')

def get_num_cores():
    num_cores = os.cpu_count()
    print('num_cores: ', num_cores)
    return num_cores

def load_data_mnist(batch_size, root=os.path.join("..", "dataset")):
    trans = transforms.Compose([
        transforms.ToTensor(),
        lambda image: image.reshape(-1) # flatten to 1d array
    ])
    mnist_train = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=trans)
    mnist_test = torchvision.datasets.MNIST(
        root=root, train=False, download=True, transform=trans)
    # print(root)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_num_cores()), 
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_num_cores()))




if __name__ == '__main__':
    train_iter, test_iter = load_data_mnist(32, dataset_dir)
    print(len(train_iter))
    print(len(test_iter))
