# chatGPT 分析报告
## 接下来请你逐文件分析下面的工程
#### [0/20] 请对下面的程序文件做一个概述: 
D:\geomtrcial_feature_code_AAAI2023\bone_dataset_remaug.py

该程序文件主要包括以下几个函数：
1. `eucli_distance_all(mat)`: 计算数据集中所有数据点之间的欧氏距离。
2. `knn_eucli_distance(mat, k)`: 计算每个数据点的k个最近邻的欧氏距离，用于近似地度量局部测地距离。
3. `gdist_appro(mat)`: 使用Dijkstra算法计算全局测地距离的近似最短路径。
4. `path_node(pre_mat, start_node, goal_node)`: 从前任矩阵重构最短路径。
5. `path_aveegr(edist, gdist, predecessors)`: 计算每对数据点之间的路径平均实际值（path_aveegr）和路径跳数（path_hop）。
6. `bone_path(path_dict, gdist)`: 标记可能可移除的数据点。
7. `bone_weight(path_dict, path_index)`: 计算数据点之间的权重。
8. `dataset_compression_index(ave_egr, path_dict, gdist, unit_hop, ratio, path_index, weight)`: 基于路径平均实际值和权重，标记可能可移除的数据点。
9. `dataset_augment_index(ave_egr, path_dict, gdist, unit_hop, ratio, path_index, weight)`: 基于路径平均实际值和权重，标记可能可添加的数据点。
10. `dataset_compress(dataset, remove_tag, percentage)`: 根据标记的可移除数据点，压缩数据集。
11. `interpolation_optimize(dataset, row, col, path_dict, edist, sam_num)`: 插值优化，计算两个数据点之间的插值。

## [1/20] 请对下面的程序文件做一个概述: D:\geomtrcial_feature_code_AAAI2023\bone_moons.py

这个程序文件名为`bone_moons.py`，它主要做了以下几个任务：
1. 导入了一些所需的库，包括`tensorflow`、`numpy`、`matplotlib`和一些自定义的模块。
2. 使用`make_moons`函数生成了一个包含800个样本的数据集，并将这些样本进行了可视化。
3. 定义了一个名为`whole_remove`的函数，该函数实现了数据集的压缩操作，并返回压缩后的数据集。
4. 定义了一个名为`whole_augment`的函数，该函数实现了数据集的扩充操作，并返回扩充后的数据集。
5. 定义了一个名为`polt_swissroll`的函数，该函数用于可视化数据集的样本。
6. 调用了`whole_remove`函数进行数据集压缩，压缩后的数据集存储在`dataset_cafter`变量中，并将压缩后的数据集及对应的样本进行可视化。
7. 调用了`whole_augment`函数进行数据集扩充，扩充后的数据集存储在`dataset_aafter`变量中，并将原始数据集及新增的样本进行可视化。

## [2/20] 请对下面的程序文件做一个概述: D:\geomtrcial_feature_code_AAAI2023\bone_swissroll.py

该程序文件名为`bone_swissroll.py`，主要包括以下功能：

1. 导入必要的库和模块。
2. 定义了一个`generate_Swissroll`函数，用于生成一个维度为3的Swissroll数据集。
3. 定义了一个`whole_remove`函数，用于对数据集进行删除操作，实现数据集的压缩。
4. 定义了一个`whole_augment`函数，用于对数据集进行增加操作，实现数据集的扩充。
5. 定义了一个`polt_swissroll`函数，用于绘制Swissroll数据集的三维图形。
6. 生成一个Swissroll数据集，并将其可视化。
7. 调用`whole_remove`函数，对数据集进行删除操作，并将结果可视化。
8. 调用`whole_augment`函数，对数据集进行增加操作，并将结果可视化。
9. 将生成的数据集和经过操作后的数据集保存到文件中。

该程序主要功能是对Swissroll数据集进行压缩和扩充操作，并将结果可视化。

## [3/20] 请对下面的程序文件做一个概述: D:\geomtrcial_feature_code_AAAI2023\cpu.py

这个程序文件名为cpu.py，它主要是用来获取并打印出当前计算机的CPU核心数量。

该程序通过导入os模块来使用一些与操作系统相关的功能。首先，它使用os.cpu_count()函数来获取当前计算机的CPU核心数量，并将其减去1赋值给变量NUM_WORKERS。然后，它打印出NUM_WORKERS的值。

接下来，它使用os.sched_getaffinity(0)函数来获取当前进程可运行在的CPU核心集合，并通过len()函数获取其长度。然后，它将其减去1赋值给变量NUM_WORKERS。最后，它再次打印出NUM_WORKERS的值。

这个程序主要的目的是获取计算机的CPU核心数量，并在屏幕上显示出来。

## [4/20] 请对下面的程序文件做一个概述: D:\geomtrcial_feature_code_AAAI2023\dataset_visual.py

该程序文件是一个用于处理数据集的可视化工具。它使用TensorFlow、Keras、PIL、imageio、NumPy等库来加载、处理和可视化图像数据集。

程序文件主要包括以下功能：
1. make_mnist函数：用于加载MNIST数据集，并将图像保存到指定目录中。
2. make_fashion_mnist函数：用于加载Fashion MNIST数据集，并将图像保存到指定目录中。
3. class_read函数：用于从数据集目录中读取指定标签的图像数据。
4. class_write函数：用于将数据集中的图像数据按照指定的格式和路径保存为图片文件。
5. umap_visual函数：使用UMAP算法对指定标签的图像数据进行降维并可视化。
6. t_sne_visual函数：使用t-SNE算法对指定标签的图像数据进行降维并可视化。

该程序文件的主要目的是为了处理并可视化MNIST和Fashion MNIST数据集中的图像数据。它提供了加载数据集、保存图像、读取图像、保存图像文件和可视化降维结果等功能。

## [5/20] 请对下面的程序文件做一个概述: D:\geomtrcial_feature_code_AAAI2023\data_load.py

该程序文件名为data_load.py，主要功能是加载数据集并进行预处理。文件中引入了多个Python库，如math、numpy、sklearn、matplotlib、pandas、datetime、tensorflow等。主要包含以下函数：

1. img_load函数：根据给定路径加载图像数据，并返回图像数组X和标签数组Y。

2. load_mnist函数：根据给定的标志flag和路径path加载MNIST数据集。若flag为False，则使用原始数据集加载方式，否则使用本地数据集加载方式。返回训练集和测试集的特征数据和标签数据。

3. load_fashion_mnist函数：根据给定的标志flag和路径path加载Fashion-MNIST数据集。若flag为False，则使用原始数据集加载方式，否则使用本地数据集加载方式。返回训练集和测试集的特征数据和标签数据。

该文件主要用于数据加载和预处理，方便后续模型的训练和测试。可以根据需要选择加载原始数据集或本地数据集，并对数据进行预处理和标准化等操作。

## [6/20] 请对下面的程序文件做一个概述: D:\geomtrcial_feature_code_AAAI2023\fashion_mnist_demo.py

[Local Message] 警告，线程6在执行过程中遭遇问题, Traceback：

```
Traceback (most recent call last):
  File ".\crazy_functions\crazy_utils.py", line 207, in _req_gpt
    gpt_say = predict_no_ui_long_connection(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".\request_llm\bridge_all.py", line 363, in predict_no_ui_long_connection
    return method(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".\request_llm\bridge_chatgpt.py", line 87, in predict_no_ui_long_connection
    raise RuntimeError("OpenAI拒绝了请求：" + error_msg)
RuntimeError: OpenAI拒绝了请求：{    "error": {        "message": "Rate limit reached for default-gpt-3.5-turbo in organization org-biZBvszKmbEdT4Br4TqmoleB on requests per min. Limit: 3 / min. Please try again in 20s. Contact us through our help center at help.openai.com if you continue to have issues. Please add a payment method to your account to increase your rate limit. Visit https://platform.openai.com/account/billing to add a payment method.",        "type": "requests",        "param": null,        "code": "rate_limit_exceeded"    }}
```

[Local Message] 警告，线程6在执行过程中遭遇问题, Traceback：

```
Traceback (most recent call last):
  File ".\crazy_functions\crazy_utils.py", line 207, in _req_gpt
    gpt_say = predict_no_ui_long_connection(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".\request_llm\bridge_all.py", line 363, in predict_no_ui_long_connection
    return method(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".\request_llm\bridge_chatgpt.py", line 87, in predict_no_ui_long_connection
    raise RuntimeError("OpenAI拒绝了请求：" + error_msg)
RuntimeError: OpenAI拒绝了请求：{    "error": {        "message": "Rate limit reached for default-gpt-3.5-turbo in organization org-biZBvszKmbEdT4Br4TqmoleB on requests per min. Limit: 3 / min. Please try again in 20s. Contact us through our help center at help.openai.com if you continue to have issues. Please add a payment method to your account to increase your rate limit. Visit https://platform.openai.com/account/billing to add a payment method.",        "type": "requests",        "param": null,        "code": "rate_limit_exceeded"    }}
```

[Local Message] 警告，线程6在执行过程中遭遇问题, Traceback：

```
Traceback (most recent call last):
  File ".\crazy_functions\crazy_utils.py", line 207, in _req_gpt
    gpt_say = predict_no_ui_long_connection(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".\request_llm\bridge_all.py", line 363, in predict_no_ui_long_connection
    return method(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".\request_llm\bridge_chatgpt.py", line 87, in predict_no_ui_long_connection
    raise RuntimeError("OpenAI拒绝了请求：" + error_msg)
RuntimeError: OpenAI拒绝了请求：{    "error": {        "message": "Rate limit reached for default-gpt-3.5-turbo in organization org-biZBvszKmbEdT4Br4TqmoleB on requests per min. Limit: 3 / min. Please try again in 20s. Contact us through our help center at help.openai.com if you continue to have issues. Please add a payment method to your account to increase your rate limit. Visit https://platform.openai.com/account/billing to add a payment method.",        "type": "requests",        "param": null,        "code": "rate_limit_exceeded"    }}
```



## [7/20] 请对下面的程序文件做一个概述: D:\geomtrcial_feature_code_AAAI2023\local.py

这个程序文件名为local.py，它包含了两个函数make_rem_mnist和make_rem_fashion_mnist，以及一些导入的库和注释代码。make_rem_mnist函数从tf.keras.datasets.mnist加载数据集，然后根据给定的百分比拆分训练集，并使用Image库将图像保存到指定路径中，最后返回True。make_rem_fashion_mnist函数与make_rem_mnist函数类似，但从tf.keras.datasets.fashion_mnist加载数据集。文件中还有一些被注释掉的代码。最后，程序调用make_rem_fashion_mnist函数。

## [8/20] 请对下面的程序文件做一个概述: D:\geomtrcial_feature_code_AAAI2023\main_add_parl.py

这个名为main_add_parl.py的文件是一个Python源代码文件。它首先导入了一些必要的库和模块，如numpy、multiprocessing和sklearn等。然后定义了一些函数，如label_parallel、floatrange、hyper_computation_parl和class_load_without_write。最后，在主函数中进行一系列迭代和计算，包括对超参数的计算、加载数据集、对数据进行处理、进行模型训练和评估等操作。整个程序的目的是对模型进行优化和测试，并输出结果。

## [9/20] 请对下面的程序文件做一个概述: D:\geomtrcial_feature_code_AAAI2023\main_add_parl_f.py

这个程序文件主要包括几个函数和一个主函数。函数包括:
1. `label_parallel`函数用于并行处理数据集的标签。
2. `floatrange`函数用于生成一个浮点数范围的列表。
3. `hyper_computation_parl`函数用于并行计算超参数。
4. `class_load_without_write`函数用于加载数据集并将不同类别的数据合并在一起。

主函数`__main__`根据给定的参数进行循环迭代，并调用其他函数对数据进行处理和模型训练。最后输出训练模型的评估结果和一些参数信息。

## [10/20] 请对下面的程序文件做一个概述: D:\geomtrcial_feature_code_AAAI2023\main_random.py

这个程序文件是一个Python脚本，其文件名为main_random.py。它包含了导入所需的库和模块，并定义了一些函数。

其中主要的函数是：
- `label_parallel(info_list, unit_hop, ratio, percentage, l)`：使用一组参数处理数据集，然后将结果保存到文件中。
- `floatrange(start, stop, steps)`：生成一个由起始值、结束值和步长确定的浮点数列表。
- `hyper_computation_parl(k, shared_dict)`：并行计算超参数。
- `class_load_without_write(ndata_dict)`：加载数据集并将其划分为训练集和测试集。

另外还定义了两个加载MNIST数据集的函数：`load_mnist_random()`和`load_fashion_mnist_random()`。

最后，主函数`main()`加载Fashion MNIST数据集，调用不同的模型函数进行训练和评估。

## [11/20] 请对下面的程序文件做一个概述: D:\geomtrcial_feature_code_AAAI2023\main_rem_fa.py

[Local Message] 警告，线程11在执行过程中遭遇问题, Traceback：

```
Traceback (most recent call last):
  File ".\crazy_functions\crazy_utils.py", line 207, in _req_gpt
    gpt_say = predict_no_ui_long_connection(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".\request_llm\bridge_all.py", line 363, in predict_no_ui_long_connection
    return method(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".\request_llm\bridge_chatgpt.py", line 87, in predict_no_ui_long_connection
    raise RuntimeError("OpenAI拒绝了请求：" + error_msg)
RuntimeError: OpenAI拒绝了请求：{    "error": {        "message": "Rate limit reached for default-gpt-3.5-turbo in organization org-biZBvszKmbEdT4Br4TqmoleB on requests per min. Limit: 3 / min. Please try again in 20s. Contact us through our help center at help.openai.com if you continue to have issues. Please add a payment method to your account to increase your rate limit. Visit https://platform.openai.com/account/billing to add a payment method.",        "type": "requests",        "param": null,        "code": "rate_limit_exceeded"    }}
```

[Local Message] 警告，线程11在执行过程中遭遇问题, Traceback：

```
Traceback (most recent call last):
  File ".\crazy_functions\crazy_utils.py", line 207, in _req_gpt
    gpt_say = predict_no_ui_long_connection(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".\request_llm\bridge_all.py", line 363, in predict_no_ui_long_connection
    return method(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".\request_llm\bridge_chatgpt.py", line 87, in predict_no_ui_long_connection
    raise RuntimeError("OpenAI拒绝了请求：" + error_msg)
RuntimeError: OpenAI拒绝了请求：{    "error": {        "message": "Rate limit reached for default-gpt-3.5-turbo in organization org-biZBvszKmbEdT4Br4TqmoleB on requests per min. Limit: 3 / min. Please try again in 20s. Contact us through our help center at help.openai.com if you continue to have issues. Please add a payment method to your account to increase your rate limit. Visit https://platform.openai.com/account/billing to add a payment method.",        "type": "requests",        "param": null,        "code": "rate_limit_exceeded"    }}
```

[Local Message] 警告，线程11在执行过程中遭遇问题, Traceback：

```
Traceback (most recent call last):
  File ".\crazy_functions\crazy_utils.py", line 207, in _req_gpt
    gpt_say = predict_no_ui_long_connection(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".\request_llm\bridge_all.py", line 363, in predict_no_ui_long_connection
    return method(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".\request_llm\bridge_chatgpt.py", line 87, in predict_no_ui_long_connection
    raise RuntimeError("OpenAI拒绝了请求：" + error_msg)
RuntimeError: OpenAI拒绝了请求：{    "error": {        "message": "Rate limit reached for default-gpt-3.5-turbo in organization org-biZBvszKmbEdT4Br4TqmoleB on requests per min. Limit: 3 / min. Please try again in 20s. Contact us through our help center at help.openai.com if you continue to have issues. Please add a payment method to your account to increase your rate limit. Visit https://platform.openai.com/account/billing to add a payment method.",        "type": "requests",        "param": null,        "code": "rate_limit_exceeded"    }}
```



## [12/20] 请对下面的程序文件做一个概述: D:\geomtrcial_feature_code_AAAI2023\main_rem_newparl.py

这个程序文件名为main_rem_newparl.py。它包括了一系列import语句，导入了需要的库和模块。程序定义了一些函数，包括label_parallel、floatrange、hyper_computation_parl和class_load_without_write。程序的主函数是main，它使用了多进程来处理数据，并在训练和测试过程中调用了其他函数来进行计算。整个程序的目的是进行数据处理和模型训练，以提取图像中的特征并进行分类。

## [13/20] 请对下面的程序文件做一个概述: D:\geomtrcial_feature_code_AAAI2023\main_rem_random.py

该程序文件名为`main_rem_random.py`，主要功能是从mnist数据集中加载随机样本，并使用FNN模型对其进行训练和评估。主要包含了以下功能：

1. 导入必要的库和模块。
2. 定义了`label_parallel`函数，用于在多个进程中执行子任务。
3. 定义了`floatrange`函数，用于生成指定范围内的浮点数列表。
4. 定义了`hyper_computation_parl`函数，用于并行计算超参数。
5. 定义了`class_load_without_write`函数，用于加载分类数据并进行预处理。
6. 定义了`load_mnist_random`函数，用于从指定路径加载mnist数据集，并进行数据拆分和预处理。
7. 定义了`main`函数，用于执行主程序逻辑。
8. 在`__name__ == "__main__"`的条件下，调用`main`函数开始程序的执行。

总的来说，该程序是一个用于训练和评估FNN模型的mnist数据集处理和模型训练的程序文件。

## [14/20] 请对下面的程序文件做一个概述: D:\geomtrcial_feature_code_AAAI2023\mnist_demo.py

[Local Message] 警告，线程14在执行过程中遭遇问题, Traceback：

```
Traceback (most recent call last):
  File ".\crazy_functions\crazy_utils.py", line 207, in _req_gpt
    gpt_say = predict_no_ui_long_connection(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".\request_llm\bridge_all.py", line 363, in predict_no_ui_long_connection
    return method(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".\request_llm\bridge_chatgpt.py", line 87, in predict_no_ui_long_connection
    raise RuntimeError("OpenAI拒绝了请求：" + error_msg)
RuntimeError: OpenAI拒绝了请求：{    "error": {        "message": "Rate limit reached for default-gpt-3.5-turbo in organization org-biZBvszKmbEdT4Br4TqmoleB on requests per min. Limit: 3 / min. Please try again in 20s. Contact us through our help center at help.openai.com if you continue to have issues. Please add a payment method to your account to increase your rate limit. Visit https://platform.openai.com/account/billing to add a payment method.",        "type": "requests",        "param": null,        "code": "rate_limit_exceeded"    }}
```

[Local Message] 警告，线程14在执行过程中遭遇问题, Traceback：

```
Traceback (most recent call last):
  File ".\crazy_functions\crazy_utils.py", line 207, in _req_gpt
    gpt_say = predict_no_ui_long_connection(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".\request_llm\bridge_all.py", line 363, in predict_no_ui_long_connection
    return method(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".\request_llm\bridge_chatgpt.py", line 87, in predict_no_ui_long_connection
    raise RuntimeError("OpenAI拒绝了请求：" + error_msg)
RuntimeError: OpenAI拒绝了请求：{    "error": {        "message": "Rate limit reached for default-gpt-3.5-turbo in organization org-biZBvszKmbEdT4Br4TqmoleB on requests per min. Limit: 3 / min. Please try again in 20s. Contact us through our help center at help.openai.com if you continue to have issues. Please add a payment method to your account to increase your rate limit. Visit https://platform.openai.com/account/billing to add a payment method.",        "type": "requests",        "param": null,        "code": "rate_limit_exceeded"    }}
```

[Local Message] 警告，线程14在执行过程中遭遇问题, Traceback：

```
Traceback (most recent call last):
  File ".\crazy_functions\crazy_utils.py", line 207, in _req_gpt
    gpt_say = predict_no_ui_long_connection(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".\request_llm\bridge_all.py", line 363, in predict_no_ui_long_connection
    return method(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".\request_llm\bridge_chatgpt.py", line 87, in predict_no_ui_long_connection
    raise RuntimeError("OpenAI拒绝了请求：" + error_msg)
RuntimeError: OpenAI拒绝了请求：{    "error": {        "message": "Rate limit reached for default-gpt-3.5-turbo in organization org-biZBvszKmbEdT4Br4TqmoleB on requests per min. Limit: 3 / min. Please try again in 20s. Contact us through our help center at help.openai.com if you continue to have issues. Please add a payment method to your account to increase your rate limit. Visit https://platform.openai.com/account/billing to add a payment method.",        "type": "requests",        "param": null,        "code": "rate_limit_exceeded"    }}
```



## [15/20] 请对下面的程序文件做一个概述: D:\geomtrcial_feature_code_AAAI2023\network.py

该文件是一个Python源代码文件，名为network.py。该文件导入了一些必要的库，包括math，numpy，tensorflow等。代码定义了一些函数，用于在MNIST数据集上构建和训练不同类型的神经网络模型。其中的函数包括mnist_fnn，mnist_rnn和mnist_cnn。这些函数分别使用全连接神经网络，循环神经网络和卷积神经网络来构建和训练模型，并返回模型的评估值。还有一些注释和未被调用的代码，可能是为了调试或测试而留下的。

## [16/20] 请对下面的程序文件做一个概述: D:\geomtrcial_feature_code_AAAI2023\old_version.py

这个程序文件名为old_version.py，包含一些导入的库以及一些函数和主函数。主要功能如下：
1. 导入了numpy、bone_dataset_remaug、dataset_visual、data_load、network等库。
2. 定义了一些函数，如label_parallel、floatrange、hyper_computation_parl、class_load_without_write等。
3. 主函数是main()，其中进行了一系列操作：调用hyper_computation_parl函数计算超参数，加载mnist数据集，调用class_load_without_write函数加载数据，调用mnist_fnn函数进行训练和测试，最后将结果保存到文件中。

总之，这个程序文件是一个旧版本的代码，用于处理mnist数据集，进行超参数计算、数据加载和模型训练。

## [17/20] 请对下面的程序文件做一个概述: D:\geomtrcial_feature_code_AAAI2023\par_network.py

这个程序文件是一个用于实现手写数字识别的神经网络模型的代码文件。它包含了三个不同的函数 `mnist_fnn`、`mnist_rnn`和`mnist_cnn`，分别对应了使用全连接神经网络、循环神经网络和卷积神经网络的三种模型。这三个模型都是用来对手写数字图片进行训练和测试的。其中 `p_mnist_nn` 函数是一个综合函数，用来比较三种模型的性能，并返回它们的损失和准确率。

## [18/20] 请对下面的程序文件做一个概述: D:\geomtrcial_feature_code_AAAI2023\test.py

这个程序文件主要是一个Python脚本，主要的功能包括导入模块、定义函数和变量、进行数据处理和绘图操作。以下是程序的概述：

- 导入模块：
  - 从re模块中导入L函数
  - 导入numpy模块并将其命名为np
  - 导入heapq模块
  - 导入mpl_toolkits.mplot3d模块中的Axes3D类
  - 导入matplotlib模块中的pyplot函数，并将其命名为plt
  - 导入pandas模块中的array函数
  - 导入sklearn模块中的train_test_split函数

- 定义变量：
  - 定义变量a为一个3x3的numpy数组
  - 通过调用argsort函数对数组a进行排序并返回索引，然后将索引进行反转排序并赋值给变量idx
  - 打印变量idx
  - 定义变量b为一个包含9个元素的numpy数组
  - 使用nlargest函数找到数组中最大的4个元素的索引并打印索引
  - 定义变量a为一个包含3个元素的numpy数组
  - 打印对12进行开方的结果
  - 生成一个介于-3和4.5之间的随机数并打印
  - 定义变量a为空列表
  - 定义变量b和c为包含两个子列表的列表
  - 将b和c中的元素添加到a中
  - 打印a

- 绘制3D图形：
  - 创建一个图形对象fig
  - 创建一个3D坐标轴对象ax，并将其添加到fig中
  - 定义变量x和y为包含1到3的数列
  - 使用meshgrid函数将x和y生成网格，并将结果赋值给变量X和Y
  - 根据给定的数学函数生成Z1、Z2和Z3三个二维数组
  - 计算Z1 + Z2 - Z3，并将结果赋值给变量Z
  - 设置x轴和y轴的标签
  - 使用plot_surface函数绘制3D图形，并使用rainbow色图分类显示
  - 显示图形

- 循环和条件判断：
  - 使用for循环打印"2"
  - 创建一个名为Path的类，并定义了一个构造函数
  - 定义一个字典对象dict，并向其中添加四个键值对，键为整数，值为包含列表和整数的列表
  - 打印字典dict中键对应的列表中的第二个元素
  - 对字典dict按值的第二个元素进行降序排序，并将结果赋值给变量s_dict
  - 使用for循环遍历s_dict并打印值
  - 定义两个二维数组a和b并进行各种数据运算
  - 使用for循环打印一系列数值
  - 使用savetxt函数将数组a保存到文件中
  - 定义两个二维数组a和b并进行元素级别的乘法运算
  - 创建一个名为dict的字典对象，并向其中添加一个键值对，键为整数，值为包含列表和字典的列表
  - 打印字典dict中键对应的列表
  - 打印包含不同类型元素的列表c
  - 使用嵌套的三重循环，并在某个条件满足时跳出循环

- 定义函数：
  - 定义一个名为class_load_without_write的函数，接受一个字典作为输入，进行一系列数据处理并返回结果

- 打印列表label的内容

- 定义一个名为floantrange的函数，接受三个参数，并返回一个等差数列

- 打印调用floantrange函数的结果

## [19/20] 请对下面的程序文件做一个概述: D:\geomtrcial_feature_code_AAAI2023\test_local.py

这是一个名为test_local.py的程序文件。它导入了一个名为dataset_visual的模块，并调用了该模块中的make_mnist函数，并传递了True作为参数。

