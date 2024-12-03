import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius_graph
from torch.utils.data import Dataset, DataLoader
from torch_scatter import scatter
from e3nn import o3
from e3nn import nn as e3nn_nn
from e3nn.o3 import Irreps, spherical_harmonics, TensorProduct,FullyConnectedTensorProduct
from e3nn.math import soft_one_hot_linspace, soft_unit_step
import math
import time
import os
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*TorchScript.*")
torch.autograd.set_detect_anomaly(True)
torch.amp.autocast(device_type='cuda', enabled=True)
max_atom = 20
# 训练模型参数
epoch_numbers = 100
learning_rate = 0.000001
embed_size = 20
num_heads = 4  # 多头注意力头数
num_layers = 4  # Transformer层数
main_hidden_sizes1 = [100,100]
main_hidden_sizes2 = [100,100]
main_hidden_sizes3 = [32]
input_size_value = 6 #R的维度
invariant = 0.5
equivariant = 1 - invariant
"""embnet中e3层参数"""
irreps_input_conv = o3.Irreps("1x0e + 1x1o + 1x2e + 1x3o")
irreps_output_conv = o3.Irreps("1x0e + 1x1o + 1x2e")
irreps_input = o3.Irreps("1x0e + 1x1o + 1x2e + 1x3o")
irreps_query = o3.Irreps("5x0e + 5x1o")
irreps_key = o3.Irreps("1x0e + 5x1o") 
irreps_output = o3.Irreps("1x0e + 1x1o + 1x2e + 1x3o") # 与v的不可约表示一致
irreps_sh_conv = o3.Irreps.spherical_harmonics(lmax=2)
irreps_sh_transformer = o3.Irreps.spherical_harmonics(3)
number_of_basis = 8 #e3nn中基函数的数量
emb_number = 32
max_radius = 6
"""mainnet中e3层参数"""
irreps_input_conv_main = o3.Irreps(f"{max_atom}x0e+ {max_atom}x1o + {max_atom}x2e+{max_atom}x3o")
irreps_output_conv_main = o3.Irreps("1x0e")
irreps_input_conv_main_2 = irreps_output_conv_main
irreps_output_conv_main_2 = o3.Irreps("1x0e")

patience_opim = 10
patience = 10  # 早停参数
dropout_value = 0.4
#定义一个映射，E_trans = E/energy_shift_value + energy_shift_value2
energy_shift_value = 1
energy_shift_value2 = 0
force_shift_value = 1
#a和b分别是energy_loss和force_loss的初始系数，update_param是这俩参数更新频率，n个batch更新一次
a = 1 / 10
b = 10
update_param = 5
max_norm_value = 1 #梯度裁剪参数
batch_size = 4
#定义RMSE损失函数
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()
    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))
criterion_2 = RMSELoss()
criterion = nn.MSELoss()
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#确定数据放缩范围
energy_df = pd.read_hdf("energy_train.h5")
energy_max = energy_df['Energy'].max()
energy_min = energy_df['Energy'].min()
# 定义Transformer嵌入网络
class EmbedNet(nn.Module):
    def __init__(self, input_size, embed_size, num_heads, num_layers, dropout_rate=dropout_value):
        self.cache = {}
        super(EmbedNet, self).__init__()
        # 输入嵌入层
        self.embedding = nn.Linear(input_size, embed_size)
        # 位置编码：使用预计算位置编码的方式
        self.positional_encoding = PositionalEncoding(embed_size, dropout_rate)
        # 多个 Transformer 编码层，每个编码层都包含自注意力和前馈网络
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(embed_size, num_heads, dropout_rate) for _ in range(num_layers)])
        self.fitnet = MainNet2(input_size=1, hidden_sizes=main_hidden_sizes3, dropout_rate=dropout_value).to(device)
        self.s_attention = MultiHeadAttention(embed_size, num_heads, num_layers, dropout_rate)
        self.e3_conv_emb = embE3Conv(max_atom, number_of_basis, max_radius, irreps_input_conv, irreps_sh_conv, irreps_output_conv)
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
            nn.Linear(20, embed_size),  
            nn.Tanh(),
            nn.Linear(embed_size, embed_size))
        self.mlp5 = nn.Sequential(
            nn.Linear(25, embed_size),  
            nn.Tanh(),
            nn.Linear(embed_size, embed_size))
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.feed_forward_1 = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_size * 4, embed_size))
        self.one_hot_mlp = nn.Sequential(
            nn.Linear(128, embed_size),
            nn.Tanh(),
            nn.Linear(embed_size, embed_size))
        self.one_hot_mlp_2 = nn.Sequential(
            nn.Linear(10, embed_size),
            nn.Tanh(),
            nn.Linear(embed_size, 1))
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
        self.tensor_product_3 = o3.FullyConnectedTensorProduct(
            irreps_in1="1x0e + 1x1o + 1x2e",           
            irreps_in2="1x0e + 1x1o + 1x2e", 
            irreps_out="1x0e + 1x1o + 1x2e + 1x3o", 
            shared_weights=True,  # 是否共享权重
            internal_weights=True,  # 使用内部生成的权重
            normalization="component"  # 特征归一化方式，可选值 "component" 或 "norm"
        )
        self.h_q = o3.Linear(irreps_input, irreps_query).to(device)
        self.tp_k = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh_transformer, irreps_key, shared_weights=False).to(device)
        self.fc_k = e3nn_nn.FullyConnectedNet([number_of_basis, emb_number, self.tp_k.weight_numel], act=torch.nn.functional.silu).to(device)
        self.tp_v = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh_transformer, irreps_output, shared_weights=False).to(device)
        self.fc_v = e3nn_nn.FullyConnectedNet([number_of_basis, emb_number, self.tp_v.weight_numel], act=torch.nn.functional.silu).to(device)
        self.dot = o3.FullyConnectedTensorProduct(irreps_query, irreps_key, "0e").to(device)
    def e3_transformer(self, f, pos):
        origin = torch.zeros_like(pos[0]).to(device)  # 中心原子坐标
        # 构造 edge_src 和 edge_dst
        edge_src = torch.zeros(pos.shape[0], dtype=torch.long).to(device)  # 中心原子索引 0
        edge_dst = torch.arange(0, pos.shape[0], dtype=torch.long).to(device)  # 邻域原子索引
        edge_vec = (pos - origin.unsqueeze(0)).to(device)  # (N, 3)
                # 用于缓存的唯一标识符
        edge_vec_key = edge_vec
        # 先检查缓存中是否已有计算结果
        if edge_vec_key in self.cache:
            # 从缓存中获取已计算的edge_sh
            edge_sh = self.cache[edge_vec_key]
        else:
            # 如果缓存中没有，进行计算并缓存结果
            edge_sh = o3.spherical_harmonics(irreps_sh_transformer, edge_vec, True, normalization='component').to(device)
            # 将计算的edge_sh存入缓存
            self.cache[edge_vec_key] = edge_sh
        # 计算边长
        edge_length = edge_vec.norm(dim=1)  # 每条边的长度
        edge_length_embedded = soft_one_hot_linspace(
            edge_vec.norm(dim=1),
            0.0,
            max_radius,
            number=number_of_basis,
            basis='gaussian',
            cutoff=True).to(device)
        edge_length_embedded = edge_length_embedded.mul(number_of_basis**0.5)
        edge_weight_cutoff = soft_unit_step(10 * (1 - edge_length / max_radius))
        batch_norm = e3nn_nn.BatchNorm(irreps=irreps_input).to(device)
        #fc_k = e3nn_nn.FullyConnectedNet([number_of_basis, 8, self.tp_k.weight_numel], act=torch.nn.functional.silu).to(device)
        #fc_v = e3nn_nn.FullyConnectedNet([number_of_basis, 8, self.tp_v.weight_numel], act=torch.nn.functional.silu).to(device)
        """"
        #多层注意力，非常影响计算速度，需要输出的不可约表示与输入的一致
        for _ in range(num_layers):
            q = self.h_q(f)
            k = self.tp_k(f[edge_src], edge_sh, self.fc_k(edge_length_embedded))
            v = self.tp_v(f[edge_src], edge_sh, self.fc_v(edge_length_embedded))
            exp = edge_weight_cutoff[:, None] * self.dot(q[edge_dst], k).exp()
            z = scatter(exp, edge_dst, dim=0, dim_size=len(f))
            z[z == 0] = 1
            alpha = exp / z[edge_dst]
            f_new = scatter(alpha.relu().sqrt() * v, edge_dst, dim=0, dim_size=len(f)).to(device)
            # 残差连接
            f = f + f_new 
            batch_norm = e3nn_nn.BatchNorm(irreps=irreps_input).to(device)
            f = batch_norm(f)
        return f
        """
        q = self.h_q(f)
        k = self.tp_k(f[edge_src], edge_sh, self.fc_k(edge_length_embedded))
        v = self.tp_v(f[edge_src], edge_sh, self.fc_v(edge_length_embedded))
        exp = edge_weight_cutoff[:, None] * self.dot(q[edge_dst], k).exp()
        z = scatter(exp, edge_dst, dim=0, dim_size=len(f))
        z[z == 0] = 1
        alpha = exp / z[edge_dst]
        return scatter(alpha.relu().sqrt() * v, edge_dst, dim=0, dim_size=len(f)).to(device)
        
    def forward(self, R):
        # 进行 One-Hot 编码
        #R5_one_hot = F.one_hot(R[:, 4].long(), num_classes=128).float()
        #O = self.one_hot_mlp(R5_one_hot)  # 使用 MLP 对 One-Hot 编码进行处理
        R6_one_hot = F.one_hot(R[:, 5].long(), num_classes=10).float()
        B = self.one_hot_mlp_2(R6_one_hot)  # 使用 MLP 对 One-Hot 编码进行处理
        #B = fit_net(B)
        Y_sh = o3.Irreps.spherical_harmonics(lmax=2)
        G_key = tuple(R[:, [0, 4, 5]])  # 使用 G 的形状作为缓存的键
        if G_key in self.cache:
            # 如果缓存中存在 G，则直接使用
            G = self.cache[G_key]
        else:
            # 如果缓存中没有，计算并缓存
            G = R[:, [0, 4, 5]]  # 第1, 5, 6列
            G = self.mlp(G)  # 经过 MLP 生成 G
            G = o3.spherical_harmonics(Y_sh, G, normalize=True, normalization='component').to(device)
            self.cache[G_key] = G  # 将计算结果存入缓存
        Z = R[:, 1:4]  # 取第2, 3, 4列作为 Z 
        #H = self.mlp2(Z)
        #Si = R[:,[0]]
        #S = self.mlp3(B)
        Y_combined = o3.spherical_harmonics(Y_sh, B*Z, True, normalization='component').to(device)
        A = self.tensor_product_3(G, Y_combined)
        #J = self.e3_conv_emb(A,Z)
        I = self.e3_transformer(A,Z)
        """
        AN = self.mlp5(A)
        AN = self.positional_encoding(AN)
        CN = E
        QA = self.q_linear_1(AN) 
        KA = self.k_linear_1(AN)
        VA = self.v_linear_1(AN)
        """
        #G_t = G.transpose(0, 1) 
        #Z_t = Z.transpose(0, 1) 
        #A = torch.matmul(torch.matmul(Z, Z_t),G)
        #H = self.positional_encoding(H)
        #G = self.positional_encoding(G)
        #S = self.positional_encoding(S)
        #O = self.positional_encoding(O)
        # 通过线性变换生成 Q, K 和 V

        #KO = self.k_linear_3(O)
        #VO = self.v_linear_3(O)
        #对多个矩阵进行处理
        #for layer in self.encoder_layers:
            #H = layer(QH, KH, VH)
            #A = layer(QA, KO, VO)
            #G = layer(QG, KG, VG)
            #S = layer(QS, KS, VS)
        # 将三个矩阵（Z, A, G）连接起来
        #S_attention = self.s_attention(QA,KA,VA,CN)

        output = torch.cat([I], dim=-1)
        #output = S_attention
        return output
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert embed_size % num_heads == 0, "embed_size 必须是 num_heads 的整数倍"
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = embed_size // num_heads
        # Dropout 和归一化层
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        # 前向传播网络
        self.feed_forward_1 = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.Tanh(),
            nn.Linear(4 * embed_size, embed_size)
        )
    def split_heads(self, x):
        N, embed_size = x.size(0), x.size(1)
        x = x.view(N, self.num_heads, self.head_dim)  # (N, num_heads, head_dim)
        return x.permute(1, 0, 2)  # (num_heads, N, head_dim)
    def combine_heads(self, x):
        """
        将多头合并为单个张量。
        """
        x = x.permute(1, 0, 2).contiguous()  # (N, num_heads, head_dim)
        N, num_heads, head_dim = x.size()
        return x.view(N, num_heads * head_dim)  # (N, embed_size)
    def forward(self, Q, K, V, H):
        # 1. 分割为多头
        K = self.split_heads(K)  # (num_heads, N, head_dim)
        V = self.split_heads(V)
        H = self.split_heads(H)
        for _ in range(self.num_layers):
            Q = self.split_heads(Q)  # (num_heads, N, head_dim)
            # 计算注意力得分
            attention_scores_qk = torch.matmul(Q, K.transpose(-2, -1))  # (num_heads, N, N)
            attention_scores_qk /= math.sqrt(self.head_dim)  # 缩放因子 √d_head

            attention_scores_h = torch.matmul(H, H.transpose(-2, -1))  # (num_heads, N, N)
            attention_scores_h /= math.sqrt(self.head_dim)
            
            attention_scores = attention_scores_qk + attention_scores_h  # (num_heads, N, N)
            attention_weights = F.softmax(attention_scores, dim=-1)  # (num_heads, N, N)
            context = torch.matmul(attention_weights, V)  # (num_heads, N, head_dim)
            context = self.combine_heads(context)  # (N, embed_size)
            Q_combined = self.combine_heads(Q)  # (N, embed_size)
            Q = Q_combined + self.dropout_1(context)
            Q = self.layer_norm_1(Q)
            # 前向传播网络
            net_output = self.feed_forward_1(Q)
            Q = Q + self.dropout_2(net_output)
            Q = self.layer_norm_2(Q)
        return Q
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
            nn.Linear(embed_size, embed_size * 4),
            nn.Tanh(),
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
        attn_output, _ = self.self_attn(Q, K, V) 
        Q = Q + self.dropout1(attn_output)  # 残差连接
        Q = self.norm1(Q)  # 层归一化
        # 2. 前馈网络层
        ff_output = self.feed_forward(Q)
        Q = Q + self.dropout2(ff_output)  # 残差连接
        Q = self.norm2(Q)  # 层归一化
        return Q

class embE3Conv(nn.Module):
    def __init__(self, max_atom, number_of_basis, max_radius, irreps_input_conv, irreps_sh_conv, irreps_output_conv):
        super(embE3Conv, self).__init__()
        self.device = device
        self.irreps_input_conv = irreps_input_conv
        self.irreps_sh_conv = irreps_sh_conv
        self.irreps_output_conv = irreps_output_conv
        self.number_of_basis = number_of_basis
        self.max_atom = max_atom
        self.max_radius = max_radius
        # 相关层初始化
        self.tp = o3.FullyConnectedTensorProduct(self.irreps_input_conv, self.irreps_sh_conv, self.irreps_output_conv, shared_weights=False).to(device)
        self.fc = e3nn_nn.FullyConnectedNet([number_of_basis, emb_number, self.tp.weight_numel], torch.nn.functional.silu).to(device)
        
    def forward(self, f_in, pos):
        origin = torch.zeros_like(pos[0]).to(device)  # 中心原子坐标
        # 构造 edge_src 和 edge_dst
        edge_src = torch.zeros(pos.shape[0], dtype=torch.long).to(device)  # 中心原子索引 0
        edge_dst = torch.arange(0, pos.shape[0], dtype=torch.long).to(device)  # 邻域原子索引
        edge_vec = (pos - origin.unsqueeze(0))  # (N, 3)
        num_neighbors = len(edge_src) / self.max_atom

        sh = o3.spherical_harmonics(self.irreps_sh_conv, edge_vec, normalize=True, normalization='component')
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            edge_length, 
            0.0, 
            self.max_radius, 
            self.number_of_basis, 
            basis='gaussian', 
            cutoff=True).mul(self.number_of_basis ** 0.5).to(device)

        return scatter(self.tp(f_in[edge_src], sh, self.fc(edge_length_embedded)), edge_dst, dim=0, dim_size=self.max_atom).div(num_neighbors ** 0.5).to(device)

# 定义主神经网络
class E3Conv(nn.Module):
    def __init__(self, max_radius, number_of_basis, irreps_input_conv, irreps_output, hidden_dim):
        super(E3Conv, self).__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.irreps_output = o3.Irreps(irreps_output)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)
        
        # 初始化 TensorProduct 和 FullyConnectedNet
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_input_conv, self.irreps_sh, self.irreps_output, shared_weights=False
        )
        self.fc = e3nn_nn.FullyConnectedNet([number_of_basis, hidden_dim, self.tp.weight_numel], torch.nn.functional.silu)

    def forward(self, f_in, pos):
        edge_src, edge_dst = radius_graph(pos, self.max_radius, max_num_neighbors=len(pos) - 1)
        edge_vec = pos[edge_dst] - pos[edge_src]
        num_nodes = pos.size(0)
        num_neighbors = len(edge_src) / num_nodes
        # 计算球谐函数和基函数
        sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization='component')
        emb = soft_one_hot_linspace(
            edge_vec.norm(dim=1), 0.0, self.max_radius, self.number_of_basis, basis='gaussian', cutoff=True
        ).mul(self.number_of_basis**0.5)

        # 应用 TensorProduct 和 FullyConnectedNet
        out = scatter(
            self.tp(f_in[edge_src], sh, self.fc(emb)),
            edge_dst,
            dim=0,
            dim_size=num_nodes
        ).div(num_neighbors**0.5)
        return out
class E3_TransformerLayer(nn.Module):
    def __init__(self, max_radius, number_of_basis, irreps_input,irreps_query, irreps_key,irreps_output, irreps_sh, hidden_dim):
        super(E3_TransformerLayer,self).__init__()
        self.irreps_sh = irreps_sh
        #self.irreps_input =irreps_input
        #self.irreps_query = irreps_query
       # self.irreps_key = irreps_key
        #self.irreps_output = irreps_output
        #self.irreps_sh = irreps_sh
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.h_q = o3.Linear(irreps_input, irreps_query)
        self.tp_k = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_key, shared_weights=False)
        self.tp_v = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_output, shared_weights=False)
        self.fc_k = e3nn_nn.FullyConnectedNet([number_of_basis, hidden_dim, self.tp_k.weight_numel], torch.nn.functional.silu)
        self.fc_v = e3nn_nn.FullyConnectedNet([number_of_basis, hidden_dim, self.tp_v.weight_numel], torch.nn.functional.silu)
        self.dot = o3.FullyConnectedTensorProduct(irreps_query, irreps_key, "0e").to(device)
    def forward(self, f, pos):
        edge_src, edge_dst = radius_graph(pos, max_radius)
        edge_vec = pos[edge_src] - pos[edge_dst]
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='smooth_finite',
            cutoff=True
        ).mul(self.number_of_basis**0.5)
        edge_weight_cutoff = soft_unit_step(10 * (1 - edge_length / self.max_radius))
        # 计算球谐函数
        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, True, normalization='component')
        # 计算 q, k, 
        q = self.h_q(f)
        k = self.tp_k(f[edge_src], edge_sh, self.fc_k(edge_length_embedded))
        v = self.tp_v(f[edge_src], edge_sh, self.fc_v(edge_length_embedded))
        for _ in range(num_layers):
            exp = edge_weight_cutoff[:, None] * self.dot(q[edge_dst], k).exp()
            z = scatter(exp, edge_dst, dim=0, dim_size=len(f))
            z[z == 0] = 1
            alpha = exp / z[edge_dst]
        return scatter(alpha.relu().sqrt() * v, edge_dst, dim=0, dim_size=len(f))
class MainNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate=dropout_value):
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
            x = F.silu(layer(x))
            x = self.dropout(x)
        Y = self.output(x)
        return Y
class CustomDataset(Dataset):
    def __init__(self, input_file_path, read_file_path, energy_file_path):
        self.input_data = pd.read_hdf(input_file_path)
        self.read_data = pd.read_hdf(read_file_path)
        self.energy_df = pd.read_hdf(energy_file_path)
        self.energy_max = energy_max
        self.energy_min = energy_min
        self.energy_df['Transformed_Energy'] = (
            2 * (self.energy_df['Energy'] - self.energy_min) / (self.energy_max - self.energy_min) - 1
        )
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
    def restore_energy(self, normalized_energy):
        return ((normalized_energy + 1) / 2) * (self.energy_max - self.energy_min) + self.energy_min
    def restore_force(self, normalized_force):
        force_range = (self.energy_max - self.energy_min)/2  # 与能量归一化因子相同
        return normalized_force * force_range
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
train_dataset = CustomDataset('train-fix.h5', 'read_train.h5', 'energy_train.h5')#如果不删除贡献为0的原子，则用train.h5，下同
val_dataset = CustomDataset('val-fix.h5', 'read_val.h5', 'energy_val.h5')
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
#def compute_M(T):
    #return torch.mm(T.T, T).requires_grad_()
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
    #M = compute_M(T)
    E = T.view(1,-1)
    return E
# 初始化嵌入网络和两个主网络
embed_net1 = EmbedNet(input_size=input_size_value, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_value).to(device)
embed_net2 = EmbedNet(input_size=input_size_value, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_value).to(device)
embed_net3 = EmbedNet(input_size=input_size_value, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_value).to(device)
embed_net4 = EmbedNet(input_size=input_size_value, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_value).to(device)
embed_net0 = EmbedNet(input_size=input_size_value, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_value).to(device)
main_net1 = MainNet(input_size=10*max_atom , hidden_sizes=main_hidden_sizes1, dropout_rate=dropout_value).to(device)
main_net2 = MainNet(input_size=embed_size * embed_size*4, hidden_sizes=main_hidden_sizes1, dropout_rate=dropout_value).to(device)
main_net3 = MainNet(input_size=embed_size * embed_size*4, hidden_sizes=main_hidden_sizes1, dropout_rate=dropout_value).to(device)
main_net4 = MainNet(input_size=embed_size * embed_size*4, hidden_sizes=main_hidden_sizes1, dropout_rate=dropout_value).to(device)
main_net0 = MainNet2(input_size=embed_size * embed_size*4, hidden_sizes=main_hidden_sizes2, dropout_rate=dropout_value).to(device)#给虚原子的mainnet
fit_net = MainNet2(input_size=42, hidden_sizes=main_hidden_sizes3, dropout_rate=dropout_value).to(device)#给权重函数的fit_net
e3conv_layer = E3Conv(max_radius, number_of_basis, irreps_input_conv_main,irreps_output_conv_main, emb_number).to(device)
e3conv_layer2 = E3Conv(max_radius, number_of_basis, irreps_input_conv_main_2,irreps_output_conv_main_2, emb_number).to(device)
e3trans = E3_TransformerLayer(max_radius, number_of_basis, irreps_input_conv_main, irreps_query, irreps_key, irreps_output_conv_main, irreps_sh_transformer, emb_number).to(device)
optimizer1 = torch.optim.AdamW(
    list(embed_net1.parameters()) + list(embed_net2.parameters()) + list(embed_net3.parameters()) + list(embed_net4.parameters()) + list(embed_net0.parameters()) +
    list(main_net1.parameters()) + list(e3conv_layer.parameters()) +list(e3conv_layer2.parameters()) +list(fit_net.parameters()) + list(e3trans.parameters()),
    lr=learning_rate,weight_decay=0.01)
#scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.1, patience=patience_opim)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=patience_opim, gamma=0.5)

# 检查是否存在之前保存的模型文件
checkpoint_path = 'combined_model.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    embed_net1.load_state_dict(checkpoint['embed_net1_state_dict'])
    embed_net2.load_state_dict(checkpoint['embed_net2_state_dict'])
    embed_net3.load_state_dict(checkpoint['embed_net3_state_dict'])
    embed_net4.load_state_dict(checkpoint['embed_net4_state_dict'])
    embed_net0.load_state_dict(checkpoint['embed_net0_state_dict'])
    #main_net1.load_state_dict(checkpoint['main_net1'])
    fit_net.load_state_dict(checkpoint['fit_net_state_dict'])
    e3conv_layer.load_state_dict(checkpoint['e3conv_layer_state_dict'])
    optimizer1.load_state_dict(checkpoint['optimizer1_state_dict'])
    scheduler1.load_state_dict(checkpoint["scheduler_state_dict"])
    #a = checkpoint["a"]
    #b = checkpoint["b"]
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
            """"
            a *= 0.9 
            b *= 1/0.9 
            a = max(a, 0.01) 
            b = min(b, 1000)  
            print(f"Updated a: {a}, b: {b} (after {batch_count} batches)")
            """
        # 解包批次数据
        input_tensors, read_tensors, target_energies = zip(*batch)
        input_tensors = [t.to(device) for t in input_tensors]
        read_tensors = [t.to(device) for t in read_tensors]
        target_energies = torch.stack(target_energies).to(device)
        batch_energy_loss = 0.0
        energy_loss = 0.0
        energy_rmse = 0.0
        batch_force_loss = 0.0
        force_loss = 0.0
        force_rmse = 0.0
        E_sum_all = []
        for input_tensor, read_tensor, target_energy in zip(input_tensors, read_tensors, target_energies):
            optimizer1.zero_grad()
            fx_pred_all, fy_pred_all, fz_pred_all = [], [], []
            fx_ref = read_tensor[:, 5] * force_shift_value
            fy_ref = read_tensor[:, 6] * force_shift_value
            fz_ref = read_tensor[:, 7] * force_shift_value
            dimensions = input_tensor[:, 0].unique().tolist()
            fx_pred_per_molecule, fy_pred_per_molecule, fz_pred_per_molecule = [], [], []
            pos = read_tensor[:,[1,2,3]]
            pos.requires_grad = True
            all_E = [] 
            E_sum = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
            for dim in dimensions:
                mask = input_tensor[:, 0] == dim
                filtered_block = input_tensor[mask]
                embed_value = filtered_block[0, 5].item()
                R = compute_R(filtered_block)
                E = compute_E(R, embed_value)
                all_E.append(E)
            E_cat = all_E_tensor = torch.cat(all_E, dim=0)
            E_conv = e3trans(E_cat,pos).sum()
            #E_conv = e3conv_layer2(E_conv,pos)
            #E_conv = E_conv.reshape(1,-1)
            #E_conv = fit_net(E_conv)
            E_conv.backward(retain_graph=True)
            fx_pred_conv = -pos.grad[:, 0]
            fy_pred_conv = -pos.grad[:, 1]
            fz_pred_conv = -pos.grad[:, 2]
            pos.grad.zero_()
            print(f"Total E_sum for this molecule: {train_dataset.restore_energy(E_conv)}")
            E_sum_all.append(E_conv)
            #print(E_sum_all)
            fx_pred_conv_batch = fx_pred_conv.clone().detach().to(device).view(-1)
            print(f"froce_x:{train_dataset.restore_force(fx_pred_conv_batch)}")
            fy_pred_conv_batch = fy_pred_conv.clone().detach().to(device).view(-1)
            fz_pred_conv_batch = fz_pred_conv.clone().detach().to(device).view(-1)
        force_loss = (
            criterion(fx_pred_conv_batch, fx_ref.detach().to(device).view(-1)) +
            criterion(fy_pred_conv_batch, fy_ref.detach().to(device).view(-1)) +
            criterion(fz_pred_conv_batch, fz_ref.detach().to(device).view(-1))) / 3 / len(dimensions)
        force_rmse = train_dataset.restore_force((
            criterion_2(fx_pred_conv_batch, fx_ref.detach().to(device).view(-1)) +
            criterion_2(fy_pred_conv_batch, fy_ref.detach().to(device).view(-1)) +
            criterion_2(fz_pred_conv_batch, fz_ref.detach().to(device).view(-1))) / 3 /len(dimensions))
        batch_force_loss += force_loss.item()
        E_sum_tensor = torch.tensor(E_sum_all, device=device,requires_grad=True).view(-1)
        #print(E_sum_all)
        energy_loss = criterion(E_sum_tensor, target_energies)
        energy_rmse = train_dataset.restore_force(criterion_2(E_sum_tensor, target_energies))
        batch_energy_loss += energy_loss.item()
        total_loss = (a * energy_loss + b * force_loss)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=max_norm_value)
        optimizer1.step()
        # 学习率调整
        scheduler1.step()
        current_lr1 = scheduler1.get_last_lr()
        end_time_batch = time.time()
        print(f"Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, "
              f"Energy Loss: {energy_loss}, Energy RMSE:{energy_rmse}, Force Loss: {force_loss}, Force RMSE:{force_rmse} "
              f"Learning Rate: {current_lr1[0]}",f"batch time: {end_time_batch - start_time_batch:.2f} seconds")
        total_energy_loss_val = 0.0
        total_force_loss_val = 0.0
        embed_net1.eval()
        embed_net2.eval()
        embed_net3.eval()
        embed_net4.eval()
        embed_net0.eval()
        e3conv_layer.eval()
        main_net1.eval()
        main_net2.eval()
        main_net3.eval()
        main_net4.eval()
        main_net0.eval()
        #with torch.no_grad():
        E_sum_all_val = []
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
            pos_val = read_tensor[:,[1,2,3]]
            pos_val.requires_grad = True
            fx_ref_val = read_tensor[:, 5]* force_shift_value  # x 方向参考力
            fy_ref_val = read_tensor[:, 6]* force_shift_value  # y 方向参考力
            fz_ref_val = read_tensor[:, 7]* force_shift_value   # z 方向参考力
            all_E_val = [] 
            E_sum_val = torch.zeros(1, dtype=torch.float32, device=device).to(device)
            for dim in dimensions:
                mask = input_tensor[:, 0] == dim  # 第一列是维度列，选择当前维度的行
                filtered_block = input_tensor[mask]  # 获取该维度的数据
                embed_value = filtered_block[0, 5].item()  # 假设第一行的某列代表嵌入值
                R_val = compute_R(filtered_block).requires_grad_(True)
                E_val = compute_E(R_val, embed_value).requires_grad_(True)
                all_E_val.append(E_val)           
            E_cat_val = all_E_tensor = torch.cat(all_E_val, dim=0)
            E_conv_val = e3conv_layer(E_cat_val,pos_val)
            #E_conv_val = e3conv_layer2(E_conv_val,pos_val)
            E_conv_val = E_conv_val.reshape(1,-1)
            E_conv_val = fit_net(E_conv_val)
            E_conv_val = E_conv_val.sum()
            E_conv_val.backward(retain_graph=True)
            E_sum_all_val.append(E_conv_val) 
            print(f"Total E_sum_val for this molecule: {val_dataset.restore_energy(E_conv_val)}")
            fx_pred_conv_val = -pos_val.grad[:, 0]
            fy_pred_conv_val = -pos_val.grad[:, 1]
            fz_pred_conv_val = -pos_val.grad[:, 2]
            
            fx_pred_conv_batch_val = fx_pred_conv_val.clone().detach().to(device).view(-1)
            fy_pred_conv_batch_val = fy_pred_conv_val.clone().detach().to(device).view(-1)
            fz_pred_conv_batch_val = fz_pred_conv_val.clone().detach().to(device).view(-1)
                        
        fx_ref_val = fx_ref_val.detach().to(device).view(-1)
        fy_ref_val = fy_ref_val.detach().to(device).view(-1)
        fz_ref_val = fz_ref_val.detach().to(device).view(-1)
        force_loss_val = val_dataset.restore_force((
            criterion_2(fx_pred_conv_batch_val, fx_ref_val) +
            criterion_2(fx_pred_conv_batch_val, fy_ref_val) +
            criterion_2(fx_pred_conv_batch_val, fz_ref_val)) / 3 / len(dimensions))
        E_sum_val_tensor = torch.tensor(E_sum_all_val, device=device,requires_grad=True).view(-1)
        energy_loss_val = val_dataset.restore_force(criterion_2(E_sum_tensor, target_energies))
        total_energy_loss_val = energy_loss_val.item()
        total_force_loss_val = force_loss.item()
        total_val_loss1 = (total_energy_loss_val + total_force_loss_val)
        print(f"""Epoch {epoch}/{epoch_numbers},
            Total Loss _val: {total_val_loss1},
            Energy RMSE_val: {total_energy_loss_val},
            Force RMSE_val: {total_force_loss_val},
            Current learning rate1: {current_lr1[0]}, """)
        loss_out.append({
                "epoch": epoch,
                "batch_count":batch_count,
                "Energy Loss": energy_loss.item(), 
                "Energy RMSE":energy_rmse.item(), 
                "Force Loss": force_loss.item(), 
                "Force RMSE":force_rmse.item(),
                "total_loss": total_energy_loss_val,
                "energy_rmse_val": total_energy_loss_val,
                "force_rmse_val": total_force_loss_val,
                "learning_rate1": current_lr1[0]
            })
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
        e3conv_layer.train()
        main_net1.train()
        main_net2.train()
        main_net3.train()
        main_net4.train()
        main_net0.train()
            # 每 n个 epoch 保存一次模型
        if batch_count % 1 == 0:
            torch.save({
                'embed_net1_state_dict': embed_net1.state_dict(),
                'embed_net2_state_dict': embed_net2.state_dict(),
                'embed_net3_state_dict': embed_net3.state_dict(),
                'embed_net4_state_dict': embed_net4.state_dict(),
                'embed_net0_state_dict': embed_net0.state_dict(),
                'main_net1_state_dict': main_net1.state_dict(),
                'fit_net_state_dict': fit_net.state_dict(),
                'e3conv_layer_state_dict': e3conv_layer.state_dict(),
                'optimizer1_state_dict': optimizer1.state_dict(),
                "scheduler_state_dict": scheduler1.state_dict(),
                "a": a, 
                "b": b, 
                "batch_count": batch_count,}, f'combined_model_batch_count_{batch_count}.pth')
            print(f"Model saved at batch_count {batch_count} as 'combined_model_batch_count_{batch_count}.pth'.")
            loss_out_df = pd.DataFrame(loss_out)
            loss_out_df.to_csv(f'epoch_{epoch}_batch_count_{batch_count}_loss.csv', index=False)
    # 早停机制
    if total_val_loss1 < best_val_loss:
        best_val_loss = total_val_loss1
        patience_counter = 0  
    else:
        patience_counter += 1 
    if patience_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch}.")
        break 
    end_time_epoch = time.time()
    epoch_energy_loss += batch_energy_loss
    epoch_force_loss += batch_force_loss
    print(f"Epoch {epoch} completed in {end_time_epoch - start_time_epoch:.2f} seconds. "
          f"Total Energy Loss: {epoch_energy_loss:.4f}, Total Force Loss: {epoch_force_loss:.4f}")
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
        'fit_net_state_dict': fit_net.state_dict(),
        'e3conv_layer_state_dict': e3conv_layer.state_dict(),
        'optimizer1_state_dict': optimizer1.state_dict(),
        "scheduler_state_dict": scheduler1.state_dict(),
        "a": a, 
        "b": b, 
        "batch_count": batch_count,}, checkpoint_path)
result_df = pd.DataFrame(results)
result_df.to_csv('results.csv', index=False)