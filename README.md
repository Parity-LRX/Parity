## 步骤，所有代码基于python3运行

1. **运行 `read-split.py`**
   - 该步骤会处理数据文件，并进行数据拆分。

2. **运行 `read-input.py`**
   - 该步骤将处理输入数据，并进行必要的数据准备。

3. **运行 `read-0.py`**
   - 该步骤将执行数据的过滤和保存。

4. **运行 `train-force.py`**
   - 该步骤用于训练深度学习模型。

## 需要安装的库

1. **numpy**: 用于高效的数值计算，特别是处理数组。
   - 安装命令: `pip install numpy`

2. **pandas**: 用于数据处理和分析，尤其是操作数据框（DataFrame）。
   - 安装命令: `pip install pandas`

3. **pytorch ≥2.4.1 with CUDA ≥12.2**: 用于深度学习训练和模型构建，CUDA 用于加速计算。
   - 安装命令: 
     - 对于有 CUDA 支持的版本，可以使用如下命令安装：
       ```bash
       pip install torch==2.4.1+cu12.2 torchvision==0.15.1+cu12.2 torchaudio==2.4.1+cu12.2
       ```
     - 如果没有 CUDA 支持，直接安装 PyTorch：
       ```bash
       pip install torch==2.4.1
       ```

## 安装其他可能需要的库

1. **torch.utils.tensorboard**: 用于在训练过程中记录日志并可视化模型训练过程。
   - 安装命令: `pip install tensorboard`
   
2. **scikit-learn**: 提供常用的机器学习工具，如数据预处理和模型评估等。
   - 安装命令: `pip install scikit-learn`

3. **torch.amp (Automatic Mixed Precision)**: 提供混合精度训练，帮助提高训练速度。
   - 已包含在 PyTorch 安装包中，无需单独安装。
