## Geo-Feature

### 描述
A repo for summer internship at NUDT, on the topic of "Geometrical Characteristic of Dataset".

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
    - `results/`：存有数据集几何特征分析的结果，使用pickle存储
    - `pics/`：训练与测试分类网络时的图片

### 运行

#### 提示：

- 由于该算法时间复杂度约为 $O(N^2)$，建议先从较小的数据集尽行测试。

- 且算法得到的中间结果（数据集特征）较大，需要较大内存，i7 16GB 内存 在 $N=10000$ 时堪能运行。

#### 测试平台信息：

软件：

- 支持 Windows 和 Linux 双平台下运行（在数据读取与存储时做了简单的适配）
- CUDA 11.7
- python 3.11
- pytorch 2.0.1

硬件：
```
Processor 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz
Installed RAM 16.0 GB (15.7 GB usable)
System type	64-bit operating system, x64-based processor

NVIDIA GeForce RTX 3050 Laptop
```