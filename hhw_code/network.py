import torch
from torch import nn
import time
import os
import utils

curfile_dir = os.path.dirname(os.path.abspath(__file__))    
dataset_dir = os.path.join(curfile_dir, '..', 'dataset')
pics_dir = os.path.join(curfile_dir, 'pics')

def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    
    animator = utils.TrainAnimator(xlabel='epoch', xlim=[1, num_epochs], 
                                   legend=['train loss', 'train acc', 'test acc'])
    
    
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
            
        train_l = train_l_sum / batch_count
        train_acc = train_acc_sum / n
        test_acc = utils.evaluate_accuracy(net, test_iter, device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l, train_acc, test_acc, time.time() - start))
        
        animator.add(epoch + 1, (train_l, train_acc, test_acc))
        animator.savefig(os.path.join(pics_dir, 'lenet_train.png'))
    return net








class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.reshape(self.shape)
    
LeNet = nn.Sequential(
    # todo
    Reshape(-1, 1, 28, 28),
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)


from dataset import load_data_mnist
import os

if __name__ == '__main__':
    # print(LeNet)
    # X = torch.rand(1, 1, 28, 28)
    # for layer in LeNet:
    #     X = layer(X)
    #     print(layer.__class__.__name__, 'output shape:\t', X.shape)

    lr, num_epochs, batch_size = 0.9, 10, 256

    dataset_dir = os.path.join("..", "dataset")
    train_iter, test_iter = load_data_mnist(batch_size, dataset_dir)
    
    for X, y in train_iter:
        print('X', X.shape, 'y', y.shape)
        break
        # exit()
    
    device = utils.try_gpu(0)
    test_lenet = LeNet
    train(test_lenet, train_iter, test_iter, num_epochs, lr, device)
    
    # test_lent2 = LeNet
    # train(test_lent2, train_iter, test_iter, num_epochs, lr, device)

    