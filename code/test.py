from re import L
import numpy as np
import heapq

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt
from pandas import array
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

a = np.array([[9, 4, 4], [3, 3, 9], [0, 4, 6]])
idx = np.argsort(a, axis=None)
idx = np.flipud(idx)
print(idx)

b = np.array([9,4,4,3,3,9,0,4,6])
print(heapq.nlargest(4, range(len(b)), a.take))

a = np.array([4,9,16])
print(np.sqrt(12))

p = np.random.uniform(-3,4.5)
print(p)

a = []
b = [[1,2],[2,1]]
c = [[1,2],[2,1]]
a.extend(b)
a.extend(c)
print(a)


fig = plt.figure()
ax = Axes3D(fig,auto_add_to_figure=False)
fig.add_axes(ax)
x = np.arange(1, 3, 0.1)
y = np.arange(1, 3, 0.1)
X, Y = np.meshgrid(x, y)  # 网格的创建，生成二维数组，这个是关键
Z1 = ( 2 + np.sqrt((X-1)*(X-1)+(Y-np.sqrt(3))*(Y-np.sqrt(3)))) / np.sqrt(X*X + Y*Y)
Z2 = (np.sqrt(((X-2)*(X-2))+Y*Y) + np.sqrt((X-1)*(X-1)+(Y-np.sqrt(3))*(Y-np.sqrt(3))))/2
Z3 = np.abs ((np.sqrt(((X-2)*(X-2))+Y*Y) - np.sqrt((X-1)*(X-1)+(Y-np.sqrt(3))*(Y-np.sqrt(3)))))
Z = Z1 + Z2 - Z3
plt.xlabel('x')
plt.ylabel('y')
# 将函数显示为3d,rstride和cstride代表row(行)和column(列)的跨度cmap为色图分类
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()


for k in range(-2):
    print("2")

class Path:
    def __init__(self, node_list):
        self.node_list = node_list
    

dict = {}
dict[2] = [[1,2,3], 3]
dict[0] = [[2,3], 2]
dict[3] = [[4,2,3], 3]
dict[1] = [[4,3], 2]
print(dict)

for _,value in dict.items():
    print(value[1])

s_dict = sorted(dict.items(), key=lambda x:(x[1][1]), reverse=True)
for i,v in (s_dict):
    print(v)


a = np.array([[1,2,3],[3,2,1],[3,4,5]])
b = np.array([[2,6,9],[3,2,1],[9,8,10]])
print(a.mean(), a.min(), a.max())
print(a/b)

for i in range( round(-1.3), round(2.3), 1):
    print(i)


np.savetxt("../result_add.txt",a , fmt='%f',delimiter=',')


a = [[1,2],[3,4]]
b = [[5,6],[7,8]]
print(np.multiply(a,b))

a = [1,2,3]

c = [[1, 2, 3],[4,5,6],[7,8,9]]
d = {"1":2, "2":3}
dict = {}
dict [1]=[a,b,c,d]
print(dict[1])

a = [1,2,3]
b = [[1,2,3],[4,5,6]]

c = [a,b]
print(c[1])



 
for  i in range(5):
    for j in range(10):
        for k in range(10):
            print(i,j,k)
            if (k>=5):
                break



def class_load_without_write(ndata_dict):
    X = ndata_dict[0]
    Y = [0 for i in range(len(ndata_dict[0]))]
    print(X)
    print(np.shape(X), np.shape(Y))
    for l in range(1,10):
        X_l = ndata_dict[l]
        Y_l = [l for i in range(len(ndata_dict[l]))]
        X = np.vstack((X, X_l))
        Y = np.append(Y, Y_l)
    #X = X.reshape(-1,28,28)
    #Y = Y.reshape(-1,1)
    print(X, Y)
    print(np.shape(X), np.shape(Y))
    x_train, X_test, y_train, Y_test = train_test_split(X, Y, test_size = 0.01, random_state = 42)
    x_train = np.concatenate((x_train, X_test), axis = 0)
    y_train = np.concatenate((y_train, Y_test), axis = 0)
    return x_train, y_train



label = range(0,10,1)
print(label)

def floatrange(start,stop,steps):
    return [start+float(i)*(stop-start)/(float(steps)-1) for i in range(steps)]


s = floatrange(0.1, 0.9, 6)[1:-1]
print(s)
