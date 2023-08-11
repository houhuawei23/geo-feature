import matplotlib.pyplot as plt
import torch
from IPython import display
from IPython import get_ipython

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def evaluate_accuracy(net, data_iter, device=None):
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    acc_sum, num = 0.0, 0
    with torch.no_grad():
        # print(data_iter)
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().cpu().item()
            num += y.shape[0]
    return acc_sum / num

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """
    Set the axes for matplotlib.
    """
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def is_jupyter():
    """
    Check if the code is running in jupyter notebook.
    """
    if get_ipython() == None:
        return False
    return True

class TrainAnimator:
    """
    TrainAnimator is used to plot the training process.
    """
    def __init__(self, 
                 xlabel=None, ylabel=None, legend=None, 
                 xlim=None, ylim=None, 
                 xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), 
                 nrows=1, ncols=1,
                 figsize=(5, 3)):
        
        if legend is None: 
            legend = []
            
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        if not is_jupyter():
            plt.ion()   
        
    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        # in jupyter notebook or normal python
        if is_jupyter():
            display.display(self.fig)
            display.clear_output(wait=True)
        else:
            plt.draw()
            plt.pause(0.001)

    
    def savefig(self, fname):
        plt.savefig(fname)