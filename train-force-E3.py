import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import time
import os
from sklearn.preprocessing import MinMaxScaler
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
from e3nn import o3
from e3nn.o3 import Irreps, spherical_harmonics, TensorProduct
from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)
torch.amp.autocast(device_type='cuda', enabled=True)
# 训练模型参数
epoch_numbers = 100
learning_rate = 0.00001
embed_size = 64
num_heads = 16  # 多头注意力头数
num_layers = 12  # Transformer层数
main_hidden_sizes1 = [100,100,100]
main_hidden_sizes2 = [100,100,100]
main_hidden_sizes3 = [64,64]
input_size_value = 6
patience_opim = 5
patience = 5  # 早停参数
dropout_value = 0.3
#定义一个映射，E_trans = E/energy_shift_value + energy_shift_value2
energy_shift_value = 100
energy_shift_value2 = 0
force_shift_value = 1
a = 1/100
b = 10
update_param = 5
max_norm_value = 1
batch_size = 1
#定义RMSE损失函数
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()
    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))
criterion = RMSELoss()
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = MinMaxScaler(feature_range=(0, 1))
#max_atom = 42 #如果要用虚原子，则开启并设置max_atom
# 定义Transformer嵌入网络
class EmbedNet(nn.Module):
    def __init__(self, input_size, embed_size, num_heads, num_layers, dropout_rate=dropout_value):
        super(EmbedNet, self).__init__()
        # 输入嵌入层
        self.embedding = nn.Linear(input_size, embed_size)
        # 位置编码：使用预计算位置编码的方式
        self.positional_encoding = PositionalEncoding(embed_size, dropout_rate)
        # 多个 Transformer 编码层，每个编码层都包含自注意力和前馈网络
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(embed_size, num_heads, dropout_rate) for _ in range(num_layers)])
        # MLP 层用于生成各种相同维度的矩阵
        self.mlp = nn.Sequential(
            nn.Linear(3, embed_size),
            nn.Tanh(),
            nn.Linear(embed_size, embed_size) )
        self.mlp2 = nn.Sequential(
            nn.Linear(3, embed_size),
            nn.Tanh(),
            nn.Linear(embed_size, embed_size))
        self.mlp3 = nn.Sequential(
            nn.Linear(1, embed_size),  
            nn.Tanh(),
            nn.Linear(embed_size, embed_size))
        self.mlp4 = nn.Sequential(
            nn.Linear(31, embed_size),  
            nn.Tanh(),
            nn.Linear(embed_size, embed_size))
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.feed_forward_1 = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_size * 4, embed_size))
        self.one_hot_mlp = nn.Sequential(
            nn.Linear(128, embed_size),
            nn.Tanh(),
            nn.Linear(embed_size, embed_size)
        )
        # 定义线性变换层，用于生成 Q, K 和 V
        self.q_linear_1 = nn.Linear(embed_size, embed_size)
        self.k_linear_1 = nn.Linear(embed_size, embed_size)
        self.v_linear_1 = nn.Linear(embed_size, embed_size)
        self.q_linear_2 = nn.Linear(embed_size, embed_size)
        self.k_linear_2 = nn.Linear(embed_size, embed_size)
        self.v_linear_2 = nn.Linear(embed_size, embed_size)
        self.q_linear_3 = nn.Linear(embed_size, embed_size)
        self.k_linear_3 = nn.Linear(embed_size, embed_size)
        self.v_linear_3 = nn.Linear(embed_size, embed_size)
        self.tensor_product = TensorProduct(
            irreps_in1="1x0e + 1x1o + 1x2e + 1x3o",  # Y_combined 对应 l=0, l=1, l=2，包含 9 个分量
            irreps_in2="1x0e + 1x1o + 1x2e + 1x3o",  
            irreps_out = "1x0e + 1x1o + 1x2e + 1x3o + 1x1e + 1x2o + 1x3e",  # 输出结果，lmax截断到3
            instructions=[
                (0, 0, 0, "uvw", True, 1.0),  
                (0, 1, 1, "uvw", True, 1.0),  
                (0, 2, 2, "uvw", True, 1.0),  
                (0, 3, 3, "uvw", True, 1.0),  
                (1, 0, 1, "uvw", True, 1.0),  
                (1, 1, 0, "uvw", True, 1.0),  
                (1, 1, 4, "uvw", True, 1.0),  
                (1, 1, 2, "uvw", True, 1.0),  
                (1, 2, 1, "uvw", True, 1.0),  
                (1, 2, 5, "uvw", True, 1.0),  
                (1, 2, 3, "uvw", True, 1.0),
                (1, 3, 2, "uvw", True, 1.0),  
                (1, 3, 6, "uvw", True, 1.0),  
                (2, 0, 2, "uvw", True, 1.0),  
                (2, 1, 1, "uvw", True, 1.0),  
                (2, 1, 5, "uvw", True, 1.0),  
                (2, 1, 3, "uvw", True, 1.0), 
                (2, 2, 0, "uvw", True, 1.0),  
                (2, 2, 4, "uvw", True, 1.0),  
                (2, 2, 2, "uvw", True, 1.0),   
                (2, 2, 6, "uvw", True, 1.0), 
                (3, 0, 3, "uvw", True, 1.0),
                (3, 1, 2, "uvw", True, 1.0),
                (3, 1, 6, "uvw", True, 1.0),
                (3, 2, 1, "uvw", True, 1.0),  
                (3, 2, 5, "uvw", True, 1.0),  
                (3, 2, 3, "uvw", True, 1.0), 
                (3, 3, 0, "uvw", True, 1.0),
                (3, 3, 4, "uvw", True, 1.0),  
                (3, 3, 2, "uvw", True, 1.0),  
                (3, 3, 6, "uvw", True, 1.0), ])
    def calculate_attention(self, Q, K, V, H, embed_size, num_heads, num_layers):
    # 检查 embed_size 是否能被 num_heads 整除
        assert embed_size % num_heads == 0 #embed_size 必须是 num_heads 的整数倍
        head_dim = embed_size // num_heads  # 每个头的维度
        def split_heads(x, num_heads):
            N, embed_size = x.size(0), x.size(1)
            head_dim = embed_size // num_heads
            x = x.view(N, num_heads, head_dim)  # (N, num_heads, head_dim)
            return x.permute(1, 0, 2)  # (num_heads, N, head_dim)
        def combine_heads(x):
            x = x.permute(1, 0, 2).contiguous()  # (N, num_heads, head_dim)
            N, num_heads, head_dim = x.size()
            return x.view(N, num_heads * head_dim)  # (N, embed_size)
        # 1. 分割为多头
        K = split_heads(K, num_heads)  # (num_heads, N, head_dim)
        V = split_heads(V, num_heads)
        H = split_heads(H, num_heads)  # (num_heads, N, head_dim)
        # 计算多个注意力层
        for _ in range(num_layers):
            Q = split_heads(Q, num_heads)  # (num_heads, N, head_dim)
            attention_scores_qk = torch.matmul(Q, K.transpose(-2, -1))  # (num_heads, N, N)
            attention_scores_qk /= math.sqrt(head_dim)  # 缩放因子 √d_head
            attention_scores_h = torch.matmul(H, H.transpose(-2, -1))  # (num_heads, N, N)
            attention_scores_h /= math.sqrt(head_dim)
            attention_scores = attention_scores_qk+ attention_scores_h  # (num_heads, N, N)
            attention_weights = F.softmax(attention_scores, dim=-1)  # (num_heads, N, N)
            context = torch.matmul(attention_weights, V)  # (num_heads, N, head_dim)
            context = combine_heads(context)  # (N, embed_size)
            Q_combined = combine_heads(Q)  # (N, embed_size)
            Q = Q_combined + self.dropout_1(context)
            Q = self.layer_norm_1(Q)
            net_output = self.feed_forward_1(Q)  # 残差连接 + 层归一化
            Q = Q + self.dropout_2(net_output) # 这里 Q 是更新后的输入
            Q = self.layer_norm_2(Q)
        return Q
    def forward(self, R):
        # 第五列进行 One-Hot 编码
        R5_one_hot = F.one_hot(R[:, 4].long(), num_classes=128).float()
        O = self.one_hot_mlp(R5_one_hot)  # 使用 MLP 对 One-Hot 编码进行处理
        G_input = R[:, [0, 4, 5]]  # 第1, 5, 6列
        G = self.mlp(G_input)  # 经过 MLP 生成 G
        Z = R[:, 1:4]  # 取第2, 3, 4列作为 Z 
        H = self.mlp2(Z)
        Si = R[:,[0]]
        S = self.mlp3(Si)
        G_t = G.transpose(0, 1) 
        Z_t = Z.transpose(0, 1) 
        A = torch.matmul(torch.matmul(Z, Z_t),G)
        H = self.positional_encoding(H)
        A = self.positional_encoding(A) 
        G = self.positional_encoding(G)
        S = self.positional_encoding(S)
        O = self.positional_encoding(O)
        # 通过线性变换生成 Q, K 和 V
        #QG = self.q_linear_3(G)
        #KG = self.k_linear_3(G) 
        #VG = self.v_linear_3(G)
        QA = self.q_linear_1(A) 
        KA = self.k_linear_1(A)
        #VA = self.v_linear_1(A)
        VO = self.v_linear_3(O)
        QS = self.q_linear_2(S) 
        KS = self.k_linear_2(S)  
        VS = self.v_linear_2(S)
        f = fit_net(Si)
        Y = o3.spherical_harmonics(0, Z,normalize=True)  # l=0
        Y = f * Y
        Y1 = o3.spherical_harmonics(1, Z,normalize=True)  # l=1
        Y1 = f * Y1
        Y2 = o3.spherical_harmonics(2, Z,normalize=True)  # l=2
        Y2 = f * Y2
        Y3 = o3.spherical_harmonics(3, Z,normalize=True)  # l=2
        Y3 = f * Y3
        Y_combined = torch.cat([Y, Y1, Y2, Y3], dim=-1)  # 合并三种阶数的球谐函数
        # 耦合
        N = self.tensor_product(Y_combined, Y_combined)
        N = self.mlp4(N)
        N = self.positional_encoding(N)
        #print(f"Nshape:{N.shape}")
        # 通过多个 Transformer Encoder 层分别对多个矩阵进行处理
        for layer in self.encoder_layers:
            #H = layer(QH, KH, VH)
            A = layer(QA, KA, VO)
            #G = layer(QG, KG, VG)
            #S = layer(QS, KS, VS)
        # 将三个矩阵（Z, A, G）连接起来
        S_attention = self.calculate_attention(QS, KS,VS, N, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers)
        output = torch.cat([A,S_attention], dim=-1)
        return output
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout_rate=dropout_value, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        # 创建位置编码矩阵
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))
        pe = torch.zeros(max_len, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 形状为 [1, max_len, embed_size]
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[0, :x.size(0), :]  # 添加位置编码到输入中
        return self.dropout(x)
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_rate=dropout_value):
        super(TransformerEncoderLayer, self).__init__()
        # 多头自注意力机制
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),  # 扩展4倍用于激活后再压缩
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_size * 4, embed_size))
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # Dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    def forward(self, Q, K, V):
        # 1. 自注意力层
        attn_output, _ = self.self_attn(Q, K, V)  # Self-attention
        Q = Q + self.dropout1(attn_output)  # 残差连接
        Q = self.norm1(Q)  # 层归一化
        # 2. 前馈网络层
        ff_output = self.feed_forward(Q)
        Q = Q + self.dropout2(ff_output)  # 残差连接
        Q = self.norm2(Q)  # 层归一化
        return Q
# 定义主神经网络
class MainNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate=0.5):
        super(MainNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layer_norms.append(nn.LayerNorm(hidden_sizes[0]))
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.layer_norms.append(nn.LayerNorm(hidden_sizes[i + 1]))
        # 输出层
        self.output = nn.Linear(hidden_sizes[-1], 1)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, M):
        x = M
        for layer, ln in zip(self.layers, self.layer_norms):
            x = layer(x)
            x = F.tanh(x)
            x = ln(x)  # 使用 LayerNorm
            x = self.dropout(x)
        Y = self.output(x)
        return Y
#backup
class MainNet2(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate=dropout_value):
        super(MainNet2, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        self.output = nn.Linear(hidden_sizes[-1], 1)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, M):
        x = M
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        Y = self.output(x)
        return Y
class CustomDataset(Dataset):
    def __init__(self, input_file_path, read_file_path, energy_file_path):
        self.input_data = pd.read_hdf(input_file_path)
        self.read_data = pd.read_hdf(read_file_path)
        self.energy_df = pd.read_hdf(energy_file_path)
        self.energy_shift = energy_shift_value  # 能量变换的偏移量
        self.energy_df['Transformed_Energy'] = ((self.energy_df['Energy'] / self.energy_shift)+energy_shift_value2)  # 对能量进行变换，数据放缩到合适的大小
        # 创建数据块
        self.input_data_blocks = self._create_data_blocks(self.input_data)
        self.read_data_blocks = self._create_data_blocks(self.read_data)
    def _create_data_blocks(self, data):
        #根据浮动值 128128.0 分割数据块
        blocks = []
        current_block = []
        stop_value = 128128.0  # 分隔符的浮动值
        # 遍历每一行数据，检查是否遇到128128.0
        for index, row in data.iterrows():
            if stop_value in row.values:  # 如果当前行包含 128128.0
                if current_block:
                    blocks.append(pd.DataFrame(current_block, columns=data.columns))
                    current_block = []  # 清空当前块
            else:
                current_block.append(row.values)       
        # 处理最后一个数据块（如果没有以 128128 结束）
        if current_block:
            blocks.append(pd.DataFrame(current_block, columns=data.columns))
        return blocks
    def __len__(self):
        return len(self.input_data_blocks)
    def __getitem__(self, idx):
        """ 获取指定索引的数据块 """
        # 获取 train 数据块和 read 数据块
        input_block = self.input_data_blocks[idx].dropna()  # train、val输入数据块
        read_block = self.read_data_blocks[idx].dropna()  # read的数据块
        if input_block.empty or read_block.empty:
            return None, None, None# 处理空块
        input_tensor = torch.tensor(input_block.values, dtype=torch.float32, device=device)
        read_tensor = torch.tensor(read_block.values, dtype=torch.float32, device=device)
        #print(f"read_tensor shape: {read_tensor.shape}")
        # 获取目标能量
        target_energy = torch.tensor(self.energy_df['Transformed_Energy'].iloc[idx], dtype=torch.float32, device=device)
        return input_tensor, read_tensor, target_energy
    # 加载数据集
train_dataset = CustomDataset('train-0.h5', 'read_train.h5', 'energy_train.h5')#如果不删除贡献为0的原子，则用train.h5，下同
val_dataset = CustomDataset('val-0.h5', 'read_val.h5', 'energy_val.h5')
# 数据集块数量
print(f"Train dataset has {len(train_dataset)} blocks.")#确认trainset的数量
print(f"Validation dataset has {len(val_dataset)} blocks.")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
val_blocks = [
    (input_tensor.to(device), read_tensor.to(device), target_energy.to(device))
    for input_tensor, read_tensor, target_energy in [val_dataset[i] for i in range(len(val_dataset))]]
cached_R = None
def compute_R(block, cache=True):#R的定义需要包含S、广义坐标（求导得到x、y、z方向力）、原子序号和环境原子序号
    global cached_R
    # 如果已经缓存且使用缓存，则直接返回
    if cached_R is not None and cache:
        return cached_R
    s_values = block[:, 1]
    x_values = block[:, 2] 
    y_values = block[:, 3] 
    z_values = block[:, 4] 
    a_values = block[:, 5] 
    e_values = block[:, 6] 
    R = torch.stack([s_values, x_values, y_values, z_values, a_values, e_values], dim=1).to(device)
    R.requires_grad_()
    if cache:
        cached_R = R  # 缓存 R
    return R
# 定义计算 T 矩阵的函数
def compute_T(embed_net, R):
    embed_output = embed_net(R)
    #print(f"Number of elements in T: {embed_output.numel()}")#可以用来确认G里面的元素数量是否合理
    return embed_output.requires_grad_()
# 定义计算 M 矩阵的函数
def compute_M(T):
    return torch.mm(T.T, T).requires_grad_()
def compute_E(R, embed_value):
    # 原子序号对应network
    embed_net = {
        0: embed_net0,
        1: embed_net1,
        6: embed_net2,
        7: embed_net3,
        8: embed_net4}.get(embed_value, embed_net0)
    main_net = {
        0: main_net0,
        1: main_net1,
        6: main_net1,
        7: main_net1,
        8: main_net1} .get(embed_value, main_net0)
    T = compute_T(embed_net, R)
    M = compute_M(T)
    E = main_net(M.view(1, -1))
    return E
# 初始化嵌入网络和两个主网络
embed_net1 = EmbedNet(input_size=input_size_value, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_value).to(device)
embed_net2 = EmbedNet(input_size=input_size_value, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_value).to(device)
embed_net3 = EmbedNet(input_size=input_size_value, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_value).to(device)
embed_net4 = EmbedNet(input_size=input_size_value, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_value).to(device)
embed_net0 = EmbedNet(input_size=input_size_value, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_value).to(device)
main_net1 = MainNet(input_size=embed_size * embed_size*4, hidden_sizes=main_hidden_sizes1, dropout_rate=dropout_value).to(device)#给虚原子的embednet
main_net2 = MainNet(input_size=embed_size * embed_size*4, hidden_sizes=main_hidden_sizes1, dropout_rate=dropout_value).to(device)
main_net3 = MainNet(input_size=embed_size * embed_size*4, hidden_sizes=main_hidden_sizes1, dropout_rate=dropout_value).to(device)
main_net4 = MainNet(input_size=embed_size * embed_size*4, hidden_sizes=main_hidden_sizes1, dropout_rate=dropout_value).to(device)
main_net0 = MainNet2(input_size=embed_size * embed_size*4, hidden_sizes=main_hidden_sizes2, dropout_rate=dropout_value).to(device)#给虚原子的mainnet
fit_net = MainNet2(input_size=1, hidden_sizes=main_hidden_sizes3, dropout_rate=dropout_value).to(device)#给权重函数的fit_net
optimizer1 = torch.optim.Adam(
    list(embed_net1.parameters()) + list(embed_net2.parameters()) + list(embed_net3.parameters()) + list(embed_net4.parameters()) + list(embed_net0.parameters()) +
    list(main_net1.parameters()) + list(main_net2.parameters()) + list(main_net3.parameters()) + list(main_net4.parameters()) + list(main_net0.parameters()),
    lr=learning_rate)
scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.9, patience=patience_opim)
# 检查是否存在之前保存的模型文件
checkpoint_path = 'combined_model.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    embed_net1.load_state_dict(checkpoint['embed_net1_state_dict'])
    embed_net2.load_state_dict(checkpoint['embed_net2_state_dict'])
    embed_net3.load_state_dict(checkpoint['embed_net3_state_dict'])
    embed_net4.load_state_dict(checkpoint['embed_net4_state_dict'])
    embed_net0.load_state_dict(checkpoint['embed_net0_state_dict'])
    main_net1.load_state_dict(checkpoint['main_net1_state_dict'])
    main_net2.load_state_dict(checkpoint['main_net2_state_dict'])
    main_net3.load_state_dict(checkpoint['main_net3_state_dict'])
    main_net4.load_state_dict(checkpoint['main_net4_state_dict'])
    main_net0.load_state_dict(checkpoint['main_net0_state_dict'])
    optimizer1.load_state_dict(checkpoint['optimizer1_state_dict'])
    scheduler1.load_state_dict(checkpoint["scheduler_state_dict"])
    a = checkpoint["a"]
    b = checkpoint["b"]
    batch_count = checkpoint["batch_count"]
    print("Loaded model from checkpoint.")
else:
    print("No checkpoint found. Starting training from scratch.")

results = []
loss_out = []
scaler = GradScaler()
best_val_loss = float('inf')
patience_counter = 0
batch_count = 0
#writer = SummaryWriter(log_dir='runs/transformer')
# 开始训练
for epoch in range(1, epoch_numbers + 1):
    start_time_epoch = time.time()
    epoch_energy_loss = 0.0
    epoch_force_loss = 0.0
    all_nets = [embed_net0, embed_net1, embed_net2, embed_net3, embed_net4,
            main_net0, main_net1, main_net2, main_net3, main_net4]
    all_parameters = [param for net in all_nets for param in net.parameters()]
    for batch_idx, batch in enumerate(train_loader):
        start_time_batch = time.time()
        # 过滤 None 数据
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            continue
        batch_count += 1
        if batch_count % update_param == 0:  # 每n个 batch 更新一次
            a *= 1 / 0.9 
            b *= 0.9 
            a = min(a, 1)  # 限制 a 最大为 1
            b = max(b, 1)  # 限制 b 最小为 1
            print(f"Updated a: {a}, b: {b} (after {batch_count} batches)")
        # 解包批次数据
        input_tensors, read_tensors, target_energies = zip(*batch)
        input_tensors = [t.to(device) for t in input_tensors]
        read_tensors = [t.to(device) for t in read_tensors]
        target_energies = torch.stack(target_energies).to(device)
        batch_energy_loss = 0.0
        batch_force_loss = 0.0
        E_sum_all = []
        for input_tensor, read_tensor, target_energy in zip(input_tensors, read_tensors, target_energies):
            optimizer1.zero_grad()
            fx_pred_all, fy_pred_all, fz_pred_all = [], [], []
            fx_ref = read_tensor[:, 5] * force_shift_value
            fy_ref = read_tensor[:, 6] * force_shift_value
            fz_ref = read_tensor[:, 7] * force_shift_value
            dimensions = input_tensor[:, 0].unique().tolist()
            E_sums_per_molecule = []
            fx_pred_per_molecule, fy_pred_per_molecule, fz_pred_per_molecule = [], [], []
            
            E_sum = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
            for dim in dimensions:
                mask = input_tensor[:, 0] == dim
                filtered_block = input_tensor[mask]
                embed_value = filtered_block[0, 5].item()
                R = compute_R(filtered_block)
                E = compute_E(R, embed_value)
                E.backward(retain_graph=True)
                E_sums_per_molecule.append(E.item())
                #print(f"E:{E_sums_per_molecule}")
                fx_pred = -R.grad[:, 1]
                fy_pred = -R.grad[:, 2]
                fz_pred = -R.grad[:, 3]
                fx_pred_all.append(fx_pred.sum().item())
                fy_pred_all.append(fy_pred.sum().item())
                fz_pred_all.append(fz_pred.sum().item())
                #print(E*energy_shift_value)
                R.grad.zero_() 
            total_E_sum = sum(E_sums_per_molecule)
            print(f"Total E_sum for this molecule: {total_E_sum * energy_shift_value}")
            E_sum_all.append(total_E_sum) 
            #print(E_sum_all)
            fx_pred_all = torch.tensor(fx_pred_all, device=device).view(-1)
            print(f"froce_x:{fx_pred_all / force_shift_value}")
            fy_pred_all = torch.tensor(fy_pred_all, device=device).view(-1)
            fz_pred_all = torch.tensor(fz_pred_all, device=device).view(-1)
        force_loss = (
            criterion(fx_pred_all, fx_ref.detach().to(device).view(-1)) +
            criterion(fy_pred_all, fy_ref.detach().to(device).view(-1)) +
            criterion(fz_pred_all, fz_ref.detach().to(device).view(-1))) / (3*len(target_energies))
        batch_force_loss += force_loss.item()
        E_sum_tensor = torch.tensor(E_sum_all, device=device,requires_grad=True).view(-1)
        print(E_sum_all)
        energy_loss = criterion(E_sum_tensor, target_energies) / len(target_energies)
        batch_energy_loss += energy_loss.item()
        total_loss = (a * energy_loss + b * force_loss)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=max_norm_value)
        optimizer1.step()
        # 学习率调整
        scheduler1.step(total_loss.item())
        current_lr1 = scheduler1.get_last_lr()
        end_time_batch = time.time()
        print(f"Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, "
              f"Energy Loss: {batch_energy_loss}, Force Loss: {batch_force_loss}, "
              f"Learning Rate: {current_lr1[0]}",f"batch time: {end_time_batch - start_time_batch:.2f} seconds")
    end_time_epoch = time.time()
    epoch_energy_loss += batch_energy_loss
    epoch_force_loss += batch_force_loss
    print(f"Epoch {epoch} completed in {end_time_epoch - start_time_epoch:.2f} seconds. "
          f"Total Energy Loss: {epoch_energy_loss:.4f}, Total Force Loss: {epoch_force_loss:.4f}")
    # 验证集评估
    total_energy_loss_val = 0.0
    total_force_loss_val = 0.0
    embed_net1.eval()
    embed_net2.eval()
    embed_net3.eval()
    embed_net4.eval()
    embed_net0.eval()
    main_net1.eval()
    main_net2.eval()
    main_net3.eval()
    main_net4.eval()
    main_net0.eval()
    #with torch.no_grad():
    for input_tensor, read_tensor, target_energy in val_blocks:  # 使用预加载的数据
        if input_tensor is None or read_tensor is None or target_energy is None:
            continue  # 跳过空块
        input_tensor = input_tensor.to(device)
        read_tensor = read_tensor.to(device)
        target_E_val = target_energy.view(1).to(device)
        fx_pred_sum_val = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
        fy_pred_sum_val = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
        fz_pred_sum_val = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
        fx_pred_all_val = []
        fy_pred_all_val = []
        fz_pred_all_val = []
        fx_ref_val = read_tensor[:, 5]* force_shift_value  # x 方向参考力
        fy_ref_val = read_tensor[:, 6]* force_shift_value  # y 方向参考力
        fz_ref_val = read_tensor[:, 7]* force_shift_value   # z 方向参考力
        E_sum_val = torch.zeros(1, dtype=torch.float32, device=device).to(device)
        for dim in dimensions:
            mask = input_tensor[:, 0] == dim  # 第一列是维度列，选择当前维度的行
            filtered_block = input_tensor[mask]  # 获取该维度的数据
            embed_value = filtered_block[0, 5].item()  # 假设第一行的某列代表嵌入值
            R_val = compute_R(filtered_block).requires_grad_(True)
            E_val = compute_E(R_val, embed_value).requires_grad_(True)
            E_val.backward(retain_graph=True)  
            E_sum_val = E_sum_val + E_val.sum()           
            fx_pred_val = -R_val.grad[:, 1]  # 对 x 坐标求导
            fy_pred_val = -R_val.grad[:, 2]  # 对 y 坐标求导
            fz_pred_val = -R_val.grad[:, 3]  # 对 z 坐标求导 
            fx_pred_sum_val = fx_pred_val.sum()
            fy_pred_sum_val = fy_pred_val.sum()
            fz_pred_sum_val = fz_pred_val.sum()
            fx_pred_all_val.append(fx_pred_sum_val)
            fy_pred_all_val.append(fy_pred_sum_val)
            fz_pred_all_val.append(fz_pred_sum_val)
            R_val.grad.zero_()
        energy_loss_val = criterion(E_sum_val, target_E_val)
        fx_pred_all_val = torch.tensor(fx_pred_all_val, device=device).view(-1)
        fy_pred_all_val = torch.tensor(fy_pred_all_val, device=device).view(-1)
        fz_pred_all_val = torch.tensor(fz_pred_all_val, device=device).view(-1)
        fx_ref_val = fx_ref_val.detach().to(device).view(-1)
        fy_ref_val = fy_ref_val.detach().to(device).view(-1)
        fz_ref_val = fz_ref_val.detach().to(device).view(-1)
        force_loss_val = (
            criterion(fx_pred_all_val, fx_ref_val) +
            criterion(fy_pred_all_val, fy_ref_val) +
            criterion(fz_pred_all_val, fz_ref_val)) / 3
    total_energy_loss_val += energy_loss_val.item() / len(val_dataset)
    total_force_loss_val += force_loss.item() /len(val_dataset)
    total_val_loss1 = (total_energy_loss_val + total_force_loss_val)
    # 早停机制
    if total_val_loss1 < best_val_loss:
        best_val_loss = total_val_loss1
        patience_counter = 0  
    else:
        patience_counter += 1 
    if patience_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch}.")
        break 
    embed_net1.train()
    embed_net2.train()
    embed_net3.train()
    embed_net4.train()
    embed_net0.train()
    main_net1.train()
    main_net2.train()
    main_net3.train()
    main_net4.train()
    main_net0.train()

    print(f"""Epoch {epoch}/{epoch_numbers},
        Total Loss: {total_loss.item():.6f},
        Energy Loss_val: {total_energy_loss_val:.6f},
        Force Loss_val: {total_force_loss_val:.6f},
        Current learning rate1: {current_lr1[0]}, """)
    #loss_out.append({'Epoch': epoch, 'Train Total_Loss1': total_loss1,'Train Total_Loss2': total_loss2,'Val Total_Loss1': val_total_loss1, 'Val Total_Loss2': val_total_loss2, 'learning rate1': current_lr1[0], 'learning rate2': current_lr2[0]})
    # 每 n个 epoch 保存一次模型
    if epoch % 1 == 0:
        torch.save({
            'embed_net1_state_dict': embed_net1.state_dict(),
            'embed_net2_state_dict': embed_net2.state_dict(),
            'embed_net3_state_dict': embed_net3.state_dict(),
            'embed_net4_state_dict': embed_net4.state_dict(),
            'embed_net0_state_dict': embed_net0.state_dict(),
            'main_net1_state_dict': main_net1.state_dict(),
            'main_net2_state_dict': main_net2.state_dict(),
            'main_net3_state_dict': main_net3.state_dict(),
            'main_net4_state_dict': main_net4.state_dict(),
            'main_net0_state_dict': main_net0.state_dict(),
            'optimizer1_state_dict': optimizer1.state_dict(),
            "scheduler_state_dict": scheduler1.state_dict(),
            "a": a, 
            "b": b, 
            "batch_count": batch_count,}, f'combined_model_epoch_{epoch}.pth')
        print(f"Model saved at epoch {epoch} as 'combined_model_epoch_{epoch}.pth'.")
        #loss_out_df = pd.DataFrame(loss_out)
        #loss_out_df.to_csv(f'epoch_{epoch}_loss.csv', index=False)
"""""
    # 嵌入网络输入
    dummy_input_embed = torch.randn(41, 5, device=device)
    writer.add_graph(embed_net1, dummy_input_embed)
    writer.add_graph(embed_net2, dummy_input_embed)
    # 主网络输入
    #dummy_input_main = torch.randn(1, 32*32, device=device)
    #writer.add_graph(main_net1, dummy_input_main)  # 记录第一个主网络
    writer.add_scalar('Learning Rate/Optimizer1', optimizer1.param_groups[0]['lr'], epoch)
    # 记录模型参数的直方图
    for name, param in embed_net1.named_parameters():
        writer.add_histogram(f'EmbedNet1/{name}', param.data.cpu().numpy(), epoch)
    for name, param in embed_net2.named_parameters():
        writer.add_histogram(f'EmbedNet2/{name}', param.data.cpu().numpy(), epoch)
    for name, param in main_net1.named_parameters():
        writer.add_histogram(f'MainNet1/{name}', param.data.cpu().numpy(), epoch)
writer.close()
"""""
# 保存模型
torch.save({
        'epoch': epoch,
        'embed_net1_state_dict': embed_net1.state_dict(),
        'embed_net2_state_dict': embed_net2.state_dict(),
        'embed_net3_state_dict': embed_net3.state_dict(),
        'embed_net4_state_dict': embed_net4.state_dict(),
        'embed_net0_state_dict': embed_net0.state_dict(),
        'main_net1_state_dict': main_net1.state_dict(),
        'main_net2_state_dict': main_net2.state_dict(),
        'main_net3_state_dict': main_net3.state_dict(),
        'main_net4_state_dict': main_net4.state_dict(),
        'main_net0_state_dict': main_net0.state_dict(),
        'optimizer1_state_dict': optimizer1.state_dict(),
        "scheduler_state_dict": scheduler1.state_dict(),
        "a": a, 
        "b": b, 
        "batch_count": batch_count,}, checkpoint_path)
result_df = pd.DataFrame(results)
result_df.to_csv('results.csv', index=False)