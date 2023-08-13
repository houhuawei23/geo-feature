## Geo-Feature

### 描述

在 NUDT 的暑期实习仓库，选题为”数据集的几何特征”。

A repo for summer internship at NUDT, on the topic of "Geometrical Characteristic of Dataset".

![reg](image/README/reg.png)

![mnist-dim2](image/README/mnist-dim2.png)

### 环境配置

建议创建conda环境，安装所需python库。

```shell
# conda environment
conda create -n geo python=3.11
conda activate geo
# for /code (based on tensorflow)
pip install -r requirements.txt
# for /hhw_code (based on pytorch)
pip install -r req-hhw_code.txt
# jupyter environment
conda install -n geo ipykernel --update-deps --force-reinstall
```

### 项目结构

- `/code/ `与 `/code_new/`：原始代码code以及简单修改后的代码code_new
- `/hhw_code/`：基于pytorch重构的代码
  - `main.ipynb`：notebook运行测试样例
  - `run_test.py`：脚本文件运行二测试样例
  - `geo.py`：实现数据集几何特征的分析，多进程处理
  - `utils.py`：通用工具函数
  - `network.py`：用于图像分类的模型、训练函数
  - `data_utils.py`：用于数据集处理、加载的相关函数
  - `app_utils.py`：应用数据集几何特征分析的结果，如数据集压缩或增强
  - `dim_reduce.py`：用于数据降维等预处理
  - `test.py`：用于测试项目中各个函数、网络模型
  - `results/`：存有数据集几何特征分析的结果，使用pickle存储
  - `pics/`：训练与测试分类网络时的图片

### 运行

#### 提示：

- 由于该算法时间复杂度约为 $O(N^2)$，建议先从较小的数据集尽行测试。
- 且算法得到的中间结果（数据集特征）较大，需要较大内存，i7 16GB 内存 在 $N=10000$ 时堪能运行。

#### 测试平台信息：

##### 软件：

- 支持 Windows 和 Linux 双平台下运行（在数据读取与存储时做了简单的适配）
- CUDA 11.7
- python 3.11
- pytorch 2.0.1

##### 硬件：

```
Processor 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz
Installed RAM 16.0 GB (15.7 GB usable)
System type	64-bit operating system, x64-based processor

NVIDIA GeForce RTX 3050 Laptop
```

#### python 库描述

req-hhw_code.txt：

* `umap-learn`：数据集降维，统一流形逼近和投影降维（UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction）。
* `numpy`：python 科学计算基础软件包，提供多维数组对象（`np.ndarray`）、各种派生对象（如屏蔽数组和矩阵）以及对数组进行快速操作的各种方法。项目中数据处理时主要数据类型为 `numpy.ndarray`。
* `scipy`：Python 科学计算基础算法，项目中用于计算欧氏距离、最短路等，效率很高。
* `scikit-learn`：开源机器学习库，支持监督和非监督学习，提供用于模型拟合、数据预处理、模型选择、模型评估的各种工具以及许多其他实用工具。项目中用于计算 $k$ 近邻距离矩阵。
* `pytorch`：深度学习框架，是一个经过优化的张量库，用于使用 GPU 和 CPU 进行深度学习。项目中用于加载数据集、构建与训练模型。
* `torchvision`：pytorch项目的一部分，由常用数据集、模型架构和计算机视觉常用图像转换组成。项目中用于加载与处理数据集。
* `matplotlib`：用于在 Python 中创建静态、动画和交互式可视化的综合库。用于可视化，也就是绘图。
* `seaborn`：基于 matplotlib 的 Python 数据可视化库。与matplotlib相比，其提供了更高级的封装，能更方便地绘制美观且信息丰富的统计图形。用于可视化，也就是绘图。

#### 其他

- 编写 `/hhw_code/test.py` 中对 `get_class_geo_feature` 进行测试时发现，计算“平均欧式-测地距离比值” (`ave_egr` 时发现，按论文中描述的计算方法，无需使用 “k-近邻测地线距离” （`geo_dist`），可以极大地化简算法，有待进一步分析确认。

- 需要对算法中间所得的数据进行进一步分析，可视化、分析其分布，从而深入理解算法，产生新想法。

- 论文中提出用 “骨干路径” （bone_path) 来避免过多地考虑子路径。但在测试时发现， “骨干路径” 非常多，非常短，当选取 200 张 mnist 0 图像，k = 5 时，骨干路径占比超过 80%， 路径长度集中在 5-6 个结点（包括起始和目的结点）。由此感到 “骨干路径” 的特征描述能力较弱，能否提出更强更有效的特征描述指标呢？

- 分析发现中间数据矩阵稀疏性较高，如何利用稀疏矩阵来更有效地计算和存储数据呢？

- 对于高维，数值较大，噪音较大的数据，欧氏距离 L2 范数容易受极端值的影响，是否能用其他范数，如 L1 范数？

- 降维 / 特征提取 后再进行“几何”特征提取？

- 只考虑局部状态，却要进行全局计算，开销较大，如何解决？

- 如何分析其他模态的数据？如携带时序信息的文本。