## 步骤，所有代码基于python3运行

1. **运行 `read-split.py`**
   - 该步骤会处理数据文件，并进行数据拆分。

2. **运行 `read-input.py`**
   - 该步骤将处理输入数据，并进行必要的数据准备。

3. **运行 `read-treat.py`**
   - 该步骤将执行统一数据大小。

4. **运行 `train-all-fast.py`**
   - 该步骤用于训练模型。

# 安装依赖

此项目运行所需的 Python 库及其安装命令如下：

### 必要的库

- **numpy**: 用于高效的数值计算，特别是处理数组。
  - 安装命令: `pip install numpy`

- **pandas**: 用于数据处理和分析，尤其是操作数据框（DataFrame）。
  - 安装命令: `pip install pandas`

- **pytorch ≥2.4.1 with CUDA ≥12.1**: 用于深度学习训练和模型构建，CUDA 用于加速计算。
  - 安装命令:
    - 对于有 CUDA 支持的版本，可以使用如下命令安装：
      ```bash
      pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
      ```
    - 如果没有 CUDA 支持，直接安装 PyTorch：
      ```bash
      pip install torch==2.4.1
      ```

- **torch_scatter和torch_cluster**: 用于在图神经网络中进行操作的库，支持稀疏张量操作。
  - 安装命令: `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html`
  - 安装命令: `pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.1+cu121.html`
- **e3nn**: 用于处理图形卷积神经网络中的对称性操作，专门用于处理对称张量、群表示和等变神经网络。
  - 安装命令: `pip install e3nn`

- **torch.utils.tensorboard**: 用于在训练过程中记录日志并可视化模型训练过程。
  - 安装命令: `pip install tensorboard`

- **scikit-learn**: 提供常用的机器学习工具，如数据预处理和模型评估等。
  - 安装命令: `pip install scikit-learn`

- **torch.amp (Automatic Mixed Precision)**: 提供混合精度训练，帮助提高训练速度。
  - 已包含在 PyTorch 安装包中，无需单独安装。


#  输入文件要求

`read-split.py` 和 `read-input.py` 是两个用于数据处理的脚本，用户无需手动将笛卡尔坐标转换为广义坐标。然而，对于用户提供的原始文件格式，我们有一定的要求。请参考我们提供的 `data.csv` 文件格式。

#### 示例内容：
40

……energy=-29654.71595871336 pbc="F F F"

C 4.65116481 -0.87536020 -0.95341924 -0.00226694 -0.00099080 -0.00055906 6

C 4.52871416 0.49243101 -0.68728938 0.00005086 0.00149402 -0.00094421 6

C 3.26602428 1.06821027 -0.44491504 -0.00220344 -0.00100710 0.00046995 6

C 2.10756895 0.23795426 -0.46852147 0.00088549 0.00012105 -0.00051350 6

……

#### 文件格式要求：

1. **编码格式**：文件应使用 UTF-8 编码。
2. **多帧结构**：多个分子结构应保存在同一个文件中，不同的结构之间通过表头区分。每个结构的表头应包含类似 `energy=xxxxxx` 的字符串，用于提取该结构的能量。
3. **坐标内容**：文件应包含 **8 列数据**，具体格式如下：
   - **原子名称**（如 C、H、O、N 等）
   - **笛卡尔坐标**：每个原子的 x、y、z 坐标。
   - **原子受力**：每个原子在 x、y、z 方向的受力值。
   - **原子序号**：每个原子的序号（如 6、1、8、7 等）。

请确保输入文件符合上述格式要求，便于数据处理和广义坐标转换。
