import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import math
import time
import os
from sklearn.preprocessing import MinMaxScaler
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# 训练模型参数
torch.autograd.set_detect_anomaly(True)
torch.amp.autocast(device_type='cuda', enabled=True)
epoch_numbers = 100
learning_rate = 0.01
embed_size = 32
num_heads = 8  # 多头注意力头数
num_layers = 4  # Transformer层数
main_hidden_sizes1 = [100,100]
main_hidden_sizes2 = [1,1]
patience_opim = 3
patience = 5  # 早停参数
criterion = nn.SmoothL1Loss()
batch_size = 128
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = MinMaxScaler(feature_range=(0, 1))
#max_atom = 42 #如果要用虚原子，则开启并设置max_atom

# 定义Transformer嵌入网络
class EmbedNet(nn.Module):
    def __init__(self, input_size, embed_size, num_heads, num_layers, dropout_rate=0.0):
        super(EmbedNet, self).__init__()
        # 输入嵌入层
        self.embedding = nn.Linear(input_size, embed_size)
        # 位置编码：使用预计算位置编码的方式
        self.positional_encoding = PositionalEncoding(embed_size, dropout_rate)
        # 多个 Transformer 编码层，每个编码层都包含自注意力和前馈网络
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(embed_size, num_heads, dropout_rate) for _ in range(num_layers)])
    def forward(self, R):
        # 嵌入输入并应用位置编码
        x = self.embedding(R)  # 输入形状：[batch_size, sequence_length, input_size] -> [batch_size, sequence_length, embed_size]
        x = self.positional_encoding(x)  # 添加位置编码
        # 逐层应用 Transformer Encoder 层
        for layer in self.encoder_layers:
            x = layer(x)  # 输出形状：[batch_size, sequence_length, embed_size]
        return x
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout_rate=0.0, max_len=5000):
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
        x = x + self.pe[:, :x.size(1), :]  # 添加位置编码到输入中
        return self.dropout(x)
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_rate=0.0):
        super(TransformerEncoderLayer, self).__init__()
        # 多头自注意力机制
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),  # 扩展4倍用于激活后再压缩
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_size * 4, embed_size)
        )
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # Dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    def forward(self, x):
        # 1. 自注意力层
        attn_output, _ = self.self_attn(x, x, x)  # Self-attention
        x = x + self.dropout1(attn_output)  # 残差连接
        x = self.norm1(x)  # 层归一化
        # 2. 前馈网络层
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)  # 残差连接
        x = self.norm2(x)  # 层归一化
        return x
    
# 定义主神经网络
class MainNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate=0):
        super(MainNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        self.output = nn.Linear(hidden_sizes[-1], 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, M):
        x = M
        for layer in self.layers:
            x = F.tanh(layer(x))
            x = self.dropout(x)
        Y = self.output(x)
        return Y
#backup
class MainNet2(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate=0):
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
        self.energy_shift = 1000  # 能量变换的偏移量
        self.energy_df['Transformed_Energy'] = self.energy_df['Energy'] / self.energy_shift  # 对能量进行变换，数据放缩到合适的大小
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
            return None, None  # 处理空块
        input_tensor = torch.tensor(input_block.values, dtype=torch.float32, device=device)
        read_tensor = torch.tensor(read_block.values, dtype=torch.float32, device=device)
        print(f"read_tensor shape: {read_tensor.shape}")
        # 获取目标能量
        target_energy = torch.tensor(self.energy_df['Transformed_Energy'].iloc[idx], dtype=torch.float32, device=device)
        return input_tensor, read_tensor, target_energy
    # 加载数据集
train_dataset = CustomDataset('train-0.h5', 'read_train.h5', 'energy_train.h5')#如果不删除贡献为0的原子，则用train.h5，下同
val_dataset = CustomDataset('val-0.h5', 'read_val.h5', 'energy_val.h5')
# 数据集块数量
print(f"Train dataset has {len(train_dataset)} blocks.")#确认trainset的数量
print(f"Validation dataset has {len(val_dataset)} blocks.")
# 将所有数据块加载到显存
train_blocks = [
    (input_tensor.to(device), read_tensor.to(device), target_energy.to(device))
    for input_tensor, read_tensor, target_energy in [train_dataset[i] for i in range(len(train_dataset))]
]
val_blocks = [
    (input_tensor.to(device), read_tensor.to(device), target_energy.to(device))
    for input_tensor, read_tensor, target_energy in [val_dataset[i] for i in range(len(val_dataset))]
]
#全局缓存，R和R_R_T是不变的，加上这个速度能提升10%
cached_R = None
cached_R_R_T = None
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
def compute_R_R_T(R, cache=True):#用以构造可对广义坐标求导的对称矩阵
    global cached_R_R_T
    if cached_R_R_T is not None and cache:
        return cached_R_R_T
    R_R_T = torch.mm(R, R.T)
    if cache:
        cached_R_R_T = R_R_T  # 缓存 R_R_T
    return R_R_T
# 定义计算 G 矩阵的函数
def compute_G(embed_net, R):
    input_tensor = R[:, [0, 4, 5]].to(device)  # 选择R第2、6、7列,如果读取其他列，记得修改embednet的input_size参数
    embed_output = embed_net(input_tensor)
    #print(f"Number of elements in G: {embed_output.numel()}")#可以用来确认G里面的元素数量是否合理
    return embed_output.requires_grad_()
# 定义计算 M 矩阵的函数
def compute_M(G, R_R_T):
    return torch.mm(G.T, torch.mm(R_R_T, G)).requires_grad_()
def compute_E(R, embed_value):
    # 原子序号对应network
    embed_net = {
        0: embed_net0,
        1: embed_net1,
        6: embed_net2,
        7: embed_net3,
        8: embed_net4
    }.get(embed_value, embed_net0)
    main_net = {
        0: main_net0,
        1: main_net1,
        6: main_net2,
        7: main_net3,
        8: main_net4} .get(embed_value, main_net0)
    G = compute_G(embed_net, R)
    M = compute_M(G, torch.mm(R, R.T))
    E = main_net(M.view(1, -1))
    return E
# 初始化嵌入网络和两个主网络
embed_net1 = EmbedNet(input_size=3, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, dropout_rate=0).to(device)
embed_net2 = EmbedNet(input_size=3, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, dropout_rate=0).to(device)
embed_net3 = EmbedNet(input_size=3, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, dropout_rate=0).to(device)
embed_net4 = EmbedNet(input_size=3, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, dropout_rate=0).to(device)
embed_net0 = EmbedNet(input_size=3, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, dropout_rate=0).to(device)
main_net1 = MainNet(input_size=embed_size * embed_size, hidden_sizes=main_hidden_sizes1, dropout_rate=0).to(device)#给虚原子的embednet
main_net2 = MainNet(input_size=embed_size * embed_size, hidden_sizes=main_hidden_sizes1, dropout_rate=0).to(device)
main_net3 = MainNet(input_size=embed_size * embed_size, hidden_sizes=main_hidden_sizes1, dropout_rate=0).to(device)
main_net4 = MainNet(input_size=embed_size * embed_size, hidden_sizes=main_hidden_sizes1, dropout_rate=0).to(device)
main_net0 = MainNet2(input_size=embed_size * embed_size, hidden_sizes=main_hidden_sizes2, dropout_rate=0).to(device)#给虚原子的mainnet


optimizer1 = torch.optim.Adam(
    list(embed_net1.parameters()) + list(embed_net2.parameters()) + list(embed_net3.parameters()) + list(embed_net4.parameters()) + 
    list(main_net1.parameters()) + list(main_net2.parameters()),
    lr=learning_rate)
scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, factor=0.1, patience=patience_opim)
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
    print("Loaded model from checkpoint.")
else:
    print("No checkpoint found. Starting training from scratch.")

results = []
loss_out = []
scaler = GradScaler()
best_val_loss = float('inf')
patience_counter = 0
writer = SummaryWriter(log_dir='runs/transformer')
#dimensions = torch.arange(0, max_atom, dtype=torch.float32)
# 开始训练
for epoch in range(1, epoch_numbers + 1):
    start_time = time.time()
    total_energy_loss = 0.0
    total_force_loss = 0.0
    total_loss = 0.0  # 总损失（能量损失 + 力损失）

    for input_tensor, read_tensor, target_energy in train_blocks:  # 使用预加载的输入数据
        if input_tensor is None or read_tensor is None or target_energy is None:
            continue  # 跳过空块
        
        input_tensor = input_tensor.to(device)
        read_tensor = read_tensor.to(device).requires_grad_(True)
        target_energy = target_energy.view(1).to(device)
        # 计算每个block的总能量 E_sum
        E_sum = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
        fx_pred_sum = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
        fy_pred_sum = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
        fz_pred_sum = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
        fx_pred_all = []
        fy_pred_all = []
        fz_pred_all = []
        fx_ref = read_tensor[:, 5]  # x 方向参考力
        fy_ref = read_tensor[:, 6]  # y 方向参考力
        fz_ref = read_tensor[:, 7]  # z 方向参考力
        dimensions = input_tensor[:, 0].unique().tolist()  # 获取所有维度
        #print(f"fx_ref:{fx_ref}")
        for dim in dimensions:
            mask = input_tensor[:, 0] == dim  # 第一列是维度列，选择当前维度的行
            filtered_block = input_tensor[mask]  # 获取该维度的数据
            embed_value = filtered_block[0, 4].item()  # 假设第一行的某列代表嵌入值
            R = compute_R(filtered_block)
            R_R_T = compute_R_R_T(R)
            E = compute_E(R, embed_value)
            E.backward(retain_graph=True)
            
            # 计算预测力
            fx_pred = -R.grad[:, 1]  # 对 x 坐标求导
            fy_pred = -R.grad[:, 2]  # 对 y 坐标求导
            fz_pred = -R.grad[:, 3]  # 对 z 坐标求导
            #print(f"fz_pred:{fz_pred}")
            # 汇总预测的力
            fx_pred_sum = fx_pred.sum()
            fy_pred_sum = fy_pred.sum()
            fz_pred_sum = fz_pred.sum()
            # 记录每个原子（dimension）的预测力
            fx_pred_all.append(fx_pred_sum)
            fy_pred_all.append(fy_pred_sum)
            fz_pred_all.append(fz_pred_sum)
        # 输出预测力
        #print(f"fz_pred_all: {fz_pred_all}")
        #print(f"fz_pred_atom value: {fz_pred_atom.item()}")  # 获取标量值
        #print(f"fz_pred_atom shape before access: {fz_pred_atom.shape}")  # 0维标量张量，shape为空

        E_sum = E_sum + E.sum()
        energy_loss = criterion(E_sum, target_energy)
        total_energy_loss += energy_loss.item() / batch_size

        fx_pred_all = torch.tensor(fx_pred_all, device=device).view(-1)
        fy_pred_all = torch.tensor(fy_pred_all, device=device).view(-1)
        fz_pred_all = torch.tensor(fz_pred_all, device=device).view(-1)
        fx_ref = fx_ref.detach().to(device).view(-1)
        fy_ref = fy_ref.detach().to(device).view(-1)
        fz_ref = fz_ref.detach().to(device).view(-1)
        force_loss = (
            criterion(fx_pred_all, fx_ref) +
            criterion(fy_pred_all, fy_ref) +
            criterion(fz_pred_all, fz_ref))

        total_force_loss += force_loss.item() / batch_size
        # 计算总损失
        total_loss = energy_loss + force_loss
        optimizer1.zero_grad()
        total_loss.backward()
        optimizer1.step()
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
            fx_ref_val = read_tensor[:, 5]  # x 方向参考力
            fy_ref_val = read_tensor[:, 6]  # y 方向参考力
            fz_ref_val = read_tensor[:, 7]  # z 方向参考力
            E_sum_val = torch.zeros(1, dtype=torch.float32, device=device).to(device)
            for dim in dimensions:
                mask = input_tensor[:, 0] == dim  # 第一列是维度列，选择当前维度的行
                filtered_block = input_tensor[mask]  # 获取该维度的数据
                embed_value = filtered_block[0, 4].item()  # 假设第一行的某列代表嵌入值
                R_val = compute_R(filtered_block)
                R_val.requires_grad_(True)
                R_R_T_val = compute_R_R_T(R_val)
                E_val = compute_E(R_val, embed_value)
                E_val.requires_grad_(True)

                E_val.backward(retain_graph=True)            
                fx_pred_val = -R_val.grad[:, 1]  # 对 x 坐标求导
                fy_pred_val = -R_val.grad[:, 2]  # 对 y 坐标求导
                fz_pred_val = -R_val.grad[:, 3]  # 对 z 坐标求导
                fx_pred_sum_val = fx_pred_val.sum()
                fy_pred_sum_val = fy_pred_val.sum()
                fz_pred_sum_val = fz_pred_val.sum()
                fx_pred_all_val.append(fx_pred_sum_val)
                fy_pred_all_val.append(fy_pred_sum_val)
                fz_pred_all_val.append(fz_pred_sum_val)
            E_sum_val = E_sum_val + E_val.sum() 
            energy_loss_val = criterion(E_sum_val, target_E_val)
            total_energy_loss_val += energy_loss_val.item() / batch_size
            fx_pred_all_val = torch.tensor(fx_pred_all_val, device=device).view(-1)
            fy_pred_all_val = torch.tensor(fy_pred_all_val, device=device).view(-1)
            fz_pred_all_val = torch.tensor(fz_pred_all_val, device=device).view(-1)
            fx_ref_val = fx_ref_val.detach().to(device).view(-1)
            fy_ref_val = fy_ref_val.detach().to(device).view(-1)
            fz_ref_val = fz_ref_val.detach().to(device).view(-1)
            force_loss_val = (
                criterion(fx_pred_all_val, fx_ref_val) +
                criterion(fy_pred_all_val, fy_ref_val) +
                criterion(fz_pred_all_val, fz_ref_val))
            total_force_loss_val += force_loss.item() / batch_size
            avg_val_loss1 = total_energy_loss_val + total_force_loss_val
    # 早停机制
    if avg_val_loss1 < best_val_loss:
        best_val_loss = avg_val_loss1
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
    scheduler1.step(total_loss)
    current_lr1 = scheduler1.get_last_lr()
    end_time = time.time()  # 记录 epoch 结束时间
    epoch_time = end_time - start_time  # 计算 epoch 时间
    print(f"""Epoch {epoch}/{epoch_numbers},
        Energy Loss: {total_energy_loss:.6f}, 
        Force Loss: {total_force_loss:.6f}, 
        Total Loss: {total_loss.item():.6f},
        Energy Loss_val: {total_energy_loss_val:.6f},
        Force Loss_val: {total_force_loss_val:.6f},
        Current learning rate1: {current_lr1[0]}, 
        Time: {epoch_time:.4f} seconds""")
    #loss_out.append({'Epoch': epoch, 'Train Total_Loss1': total_loss1,'Train Total_Loss2': total_loss2,'Val Total_Loss1': val_total_loss1, 'Val Total_Loss2': val_total_loss2, 'learning rate1': current_lr1[0], 'learning rate2': current_lr2[0]})
    # 每 n个 epoch 保存一次模型
    if epoch % 1 == 0:
        torch.save({
            'embed_net1_state_dict': embed_net1.state_dict(),
            'embed_net2_state_dict': embed_net2.state_dict(),
            'embed_net3_state_dict': embed_net3.state_dict(),
            'embed_net4_state_dict': embed_net4.state_dict(),
            'embed_net5_state_dict': embed_net0.state_dict(),
            'main_net1_state_dict': main_net1.state_dict(),
            'main_net2_state_dict': main_net2.state_dict(),
            'optimizer1_state_dict': optimizer1.state_dict(),
        }, f'combined_model_epoch_{epoch}.pth')
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
    }, checkpoint_path)
result_df = pd.DataFrame(results)
result_df.to_csv('results.csv', index=False)