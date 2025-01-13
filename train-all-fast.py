import pandas as pd
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius_graph
from torch.utils.data import Dataset, DataLoader
from torch_scatter import scatter
from e3nn import o3
from e3nn import nn as e3nn_nn
from e3nn.o3 import Irreps, spherical_harmonics, TensorProduct,FullyConnectedTensorProduct
from e3nn.math import soft_one_hot_linspace, soft_unit_step
from e3nn.nn import Gate
import math
import time
import os
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*TorchScript.*")
torch.autograd.set_detect_anomaly(True)
torch.amp.autocast(device_type='cuda', enabled=True)
max_atom = 10
# 训练模型参数
epoch_numbers = 100
learning_rate = 0.0001
embed_size = 32 #G矩阵的MLP隐藏层
embed_size_2 = 32  #O和B的MLP隐藏层
num_heads = 4  # 多头注意力头数
num_layers = 4  # Transformer层数
input_size_value = 6 #R的维度
invariant = 0.5
equivariant = 1 - invariant
main_hidden_sizes1 = [4]
main_hidden_sizes2 = [16,8]
main_hidden_sizes3 = [4] #one-hot编码后MLP隐藏层
"""embnet中e3层参数"""
channel_in = 48
irreps_input_conv = o3.Irreps("1x0e + 1x1o + 1x2e + 1x3o")
irreps_output_conv = o3.Irreps(f"{channel_in}x0e + {channel_in}x1o + {channel_in}x2e")
irreps_iutput_conv_2 = o3.Irreps(f"{channel_in}x0e + {channel_in}x1o + {channel_in}x2e")
irreps_output_conv_2 = o3.Irreps(f"{channel_in}x0e + {channel_in}x1o + {channel_in}x2e")
irreps_input = o3.Irreps("1x0e + 1x1o + 1x2e + 1x3o")
irreps_query = o3.Irreps("10x0e + 10x1o")
irreps_key = o3.Irreps("10x0e + 10x1o") 
irreps_output = o3.Irreps("10x0e + 10x1o + 10x2e") # 与v的不可约表示一致
irreps_sh_conv = o3.Irreps.spherical_harmonics(lmax=2)
irreps_sh_transformer = o3.Irreps.spherical_harmonics(lmax=2)
emb_number = [64,64,64] #嵌入网络e3MLP最好和主网络e3MLP隐藏层大小一致，层数多一层
number_of_basis = 4 #e3nn中基函数的数量
max_radius = 8
function_type = 'gaussian'
"""mainnet中e3层参数"""
embedding_value = max_atom * 9  * channel_in#irreps_input_conv_main的维度
irreps_input_conv_main = o3.Irreps(f"{max_atom * channel_in}x0e + {max_atom * channel_in}x1o + {max_atom * channel_in}x2e")
irreps_output_conv_main = o3.Irreps(f"{max_atom * 5}x0e + {max_atom * 10}x1o + {max_atom * 10}x2e")
irreps_input_conv_main_2 = irreps_output_conv_main
irreps_output_conv_main_2 = o3.Irreps("50x0e")
irreps_query_main = o3.Irreps("20x0e + 20x1o")
irreps_key_main = o3.Irreps("5x0e + 5x1o")
hidden_dim_sh = o3.Irreps("10x0e")
emb_number_main = [64,64]
emb_number_main_2 = [64,64,64]
number_of_basis_main = 15
max_radius_main = 30
function_type_main = 'gaussian'


main_hidden_sizes4 = [4]
input_dim_weight = 1 #要和卷积层输出通道数一致
dropout_value = 0

patience_opim = 30
gamma_value = 0.98
patience = 1  # 早停参数

#定义一个映射，E_trans = E/energy_shift_value + energy_shift_value2
energy_shift_value = 1
energy_shift_value2 = 0
force_shift_value = 1
force_coefficient = 1000
#a和b分别是energy_loss和force_loss的初始系数，update_param是这俩参数更新频率，n个batch更新一次
a = 1
b = 10
mollifier_sigma = 1
lambda_reg_value = 1000
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
#criterion = nn.SmoothL1Loss(beta=0.5)
criterion = nn.MSELoss()
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#确定数据放缩范围
energy_df = pd.read_hdf("energy_train.h5")
energy_max = energy_df['Energy'].max()
energy_min = energy_df['Energy'].min()
energy_mean = energy_df['Energy'].mean()
energy_std = energy_df['Energy'].std()
# 定义Transformer嵌入网络
class EmbedNet(nn.Module):
    def __init__(self, input_size, embed_size, num_heads, num_layers, dropout_rate=0.1):
        super(EmbedNet, self).__init__()
        self.fitnet = MainNet2(input_size=1, hidden_sizes=main_hidden_sizes3, dropout_rate=dropout_rate).to(device)
        self.fitnet_2 = MainNet2(input_size=1, hidden_sizes=main_hidden_sizes3, dropout_rate=dropout_rate).to(device)
        self.e3_conv_emb = embE3Conv(max_atom, number_of_basis, max_radius, irreps_input_conv, irreps_sh_conv, irreps_output_conv).to(device)
        self.linear_layer = o3.Linear(irreps_output_conv, irreps_output_conv)
        self.mlp = nn.Sequential(
            nn.Linear(3, embed_size),
            nn.SiLU(),
            nn.Linear(embed_size, 3)
        )
        self.one_hot_mlp = nn.Sequential(
            nn.Linear(10, embed_size_2),
            nn.SiLU(),
            nn.Linear(embed_size_2, 1)
        )
        self.one_hot_mlp_2 = nn.Sequential(
            nn.Linear(10, embed_size_2),
            nn.SiLU(),
            nn.Linear(embed_size_2, 1)
        )
        self.tensor_product_3 = o3.FullyConnectedTensorProduct(
            irreps_in1="1x0e + 1x1o + 1x2e",
            irreps_in2="1x0e + 1x1o + 1x2e",
            irreps_out="1x0e + 1x1o + 1x2e + 1x3o",
            shared_weights=True,
            internal_weights=True,
            normalization="norm"
        )
        self.h_q = o3.Linear(irreps_input, irreps_query).to(device)
        self.tp_k = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh_transformer, irreps_key, shared_weights=False).to(device)
        self.fc_k = e3nn_nn.FullyConnectedNet([number_of_basis] + emb_number + [self.tp_k.weight_numel], act=torch.nn.functional.silu).to(device)
        self.tp_v = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh_transformer, irreps_output, shared_weights=False).to(device)
        self.fc_v = e3nn_nn.FullyConnectedNet([number_of_basis] + emb_number + [self.tp_v.weight_numel], act=torch.nn.functional.silu).to(device)
        self.dot = o3.FullyConnectedTensorProduct(irreps_query, irreps_key, "0e").to(device)

    def e3_transformer(self, f, pos):
        """
        支持批量输入的 e3_transformer。
        输入 f 的形状为 (batch_size, num_nodes, feature_dim)。
        输入 pos 的形状为 (batch_size, num_nodes, 3)。
        输出形状为 (batch_size, num_nodes, output_dim)。
        """
        batch_size, num_nodes, _ = pos.shape
        origin = torch.zeros(batch_size, 3, device=device)  # 每个样本的中心原子坐标
        edge_src, edge_dst = radius_graph(pos, max_radius, batch=torch.arange(batch_size, device=device).repeat_interleave(num_nodes))
        edge_vec = pos.view(-1, 3)[edge_dst] - pos.view(-1, 3)[edge_src]  # (num_edges, 3)
        edge_sh = o3.spherical_harmonics(irreps_sh_transformer, edge_vec, True, normalization='norm').to(device)
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            edge_length,
            0.0,
            max_radius,
            number=number_of_basis,
            basis='gaussian',
            cutoff=True
        ).mul(number_of_basis**0.5).to(device)
        edge_weight_cutoff = soft_unit_step(10 * (1 - edge_length / max_radius))

        q = self.h_q(f.view(-1, f.shape[-1])).view(batch_size, num_nodes, -1)
        k = self.tp_k(f.view(-1, f.shape[-1])[edge_src], edge_sh, self.fc_k(edge_length_embedded))
        v = self.tp_v(f.view(-1, f.shape[-1])[edge_src], edge_sh, self.fc_v(edge_length_embedded))

        exp = edge_weight_cutoff[:, None] * self.dot(q.view(-1, q.shape[-1])[edge_dst], k).exp()
        z = scatter(exp, edge_dst, dim=0, dim_size=batch_size * num_nodes)
        z[z == 0] = 1
        alpha = exp / z[edge_dst]
        f_new = scatter(alpha.relu().sqrt() * v, edge_dst, dim=0, dim_size=batch_size * num_nodes)
        return f_new.view(batch_size, num_nodes, -1)

    def forward(self, R):
        """
        支持批量输入的 forward。
        输入 R 的形状为 (batch_size, num_nodes, 6)。
        输出形状为 (batch_size, num_nodes, output_dim)。
        """
        # 假设 batch_size 由 len(dimensions) 决定
        batch_size = R.size(0)
        num_nodes = max_atom  # 从 R 中获取 num_nodes

        # One-Hot 编码
        R5_one_hot = F.one_hot(R[:, :, 4].long(), num_classes=10).to(torch.float64)  # (batch_size, num_nodes, 10)
        O = self.one_hot_mlp(R5_one_hot)  # (batch_size, num_nodes, 1)
        O = self.fitnet(O)  # (batch_size, num_nodes, 1)
        
        R6_one_hot = F.one_hot(R[:, :, 5].long(), num_classes=10).to(torch.float64)  # (batch_size, num_nodes, 10)
        B = self.one_hot_mlp_2(R6_one_hot)  # (batch_size, num_nodes, 1)
        B = self.fitnet_2(B)  # (batch_size, num_nodes, 1)

        # 生成 G
        K = R[:, :, 0].unsqueeze(-1)  # (batch_size, num_nodes, 1)
        G = torch.cat([K, O, B], dim=-1)  # (batch_size, num_nodes, 3)
        G = self.mlp(G)  # (batch_size, num_nodes, 3)
        
        # 计算球谐函数
        G = o3.spherical_harmonics(o3.Irreps.spherical_harmonics(lmax=2), G, normalize=True, normalization='component').to(device)

        # 生成 Y_combined
        Z = R[:, :, 1:4]  # (batch_size, num_nodes, 3)
        Y_combined = o3.spherical_harmonics(o3.Irreps.spherical_harmonics(lmax=2), Z, True, normalization='component').to(device)

        # 计算 A
        A = self.tensor_product_3(G, Y_combined)  # (batch_size, num_nodes, feature_dim)

        # 调用 e3_conv_emb
        J_flat = self.e3_conv_emb(A, Z)  # (batch_size * num_nodes, output_dim)
    
        # 将结果重塑为 (batch_size, num_nodes, output_dim)
        J = J_flat.view(batch_size, num_nodes, -1)  # (batch_size, num_nodes, output_dim)
        J = self.linear_layer(J)

        return J

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
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_input_conv, self.irreps_sh_conv, self.irreps_output_conv, shared_weights=False
        ).to(device)
        self.fc = e3nn_nn.FullyConnectedNet(
            [number_of_basis] + emb_number + [self.tp.weight_numel], torch.nn.functional.silu
        ).to(device)

    def forward(self, f_in, pos):
        """
        支持批量输入的 forward。
        输入 f_in 的形状为 (batch_size, num_nodes, feature_dim)。
        输入 pos 的形状为 (batch_size, num_nodes, 3)。
        输出形状为 (batch_size, num_nodes, output_dim)。
        """
        batch_size, num_nodes, _ = pos.shape

        # 将输入展平为 (batch_size * num_nodes, feature_dim) 和 (batch_size * num_nodes, 3)
        f_in_flat = f_in.view(-1, f_in.shape[-1])  # (batch_size * num_nodes, feature_dim)
        pos_flat = pos.view(-1, 3)  # (batch_size * num_nodes, 3)

        # 构造 edge_src 和 edge_dst
        edge_src, edge_dst = radius_graph(
            pos_flat, self.max_radius, batch=torch.arange(batch_size, device=device).repeat_interleave(num_nodes)
        )  # edge_src 和 edge_dst 的形状为 (num_edges,)

        # 计算边向量
        edge_vec = pos_flat[edge_dst] - pos_flat[edge_src]  # (num_edges, 3)

        # 计算球谐函数
        sh = o3.spherical_harmonics(self.irreps_sh_conv, edge_vec, normalize=True, normalization='norm').to(device)

        # 计算边长和嵌入
        edge_length = edge_vec.norm(dim=1)  # (num_edges,)
        edge_length_embedded = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            number=self.number_of_basis,
            basis=function_type,
            cutoff=True
        ).mul(self.number_of_basis ** 0.5).to(device)  # (num_edges, number_of_basis)

        # 计算卷积结果
        tp_output = self.tp(f_in_flat[edge_src], sh, self.fc(edge_length_embedded))  # (num_edges, output_dim)

        # 使用 scatter 聚合结果
        output_flat = scatter(
            tp_output, edge_dst, dim=0, dim_size=batch_size * num_nodes
        ).div((len(edge_src) / (batch_size * num_nodes)) ** 0.5)  # (batch_size * num_nodes, output_dim)

        # 将结果重塑为 (batch_size, num_nodes, output_dim)
        output = output_flat.view(batch_size, num_nodes, -1)  # (batch_size, num_nodes, output_dim)

        return output

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
        self.fc = e3nn_nn.FullyConnectedNet(
            [number_of_basis] + hidden_dim + [self.tp.weight_numel],
            torch.nn.functional.silu
        )

    def forward(self, f_in, pos):
        edge_src, edge_dst = radius_graph(pos, self.max_radius, max_num_neighbors=len(pos) - 1)
        edge_vec = pos[edge_dst] - pos[edge_src]
        num_nodes = pos.size(0)
        num_neighbors = len(edge_src) / num_nodes
        # 计算球谐函数和基函数
        sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization='component')
        emb = soft_one_hot_linspace(
            edge_vec.norm(dim=1), 0.0, self.max_radius, self.number_of_basis, basis='smooth_finite', cutoff=True
        ).mul(self.number_of_basis**0.5)

        out = scatter(
            self.tp(f_in[edge_src], sh, self.fc(emb)),
            edge_dst,
            dim=0,
            dim_size=num_nodes
        ).div(num_neighbors**0.5)
        return out
class E3_TransformerLayer(nn.Module):
    def __init__(self, max_radius, number_of_basis, irreps_input,irreps_query, irreps_key,irreps_output, irreps_sh, hidden_dim_sh, hidden_dim):
        super(E3_TransformerLayer,self).__init__()
        self.irreps_sh = irreps_sh
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.h_q = o3.Linear(irreps_input, irreps_query)
        self.tp_k = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_key, shared_weights=False)
        self.tp_v = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_output, shared_weights=False)
        self.fc_k = e3nn_nn.FullyConnectedNet([number_of_basis] + hidden_dim + [self.tp_k.weight_numel], torch.nn.functional.silu)
        self.fc_v = e3nn_nn.FullyConnectedNet([number_of_basis] + hidden_dim + [self.tp_v.weight_numel], torch.nn.functional.silu)
        self.dot = o3.FullyConnectedTensorProduct(irreps_query, irreps_key, "0e").to(device)
        self.linear_layer = o3.Linear(irreps_output, hidden_dim_sh)
        self.non_linearity = nn.SiLU() 
        self.linear_layer_2 = o3.Linear(hidden_dim_sh, o3.Irreps("1x0e"))
    def forward(self, f, pos):
        edge_src, edge_dst = radius_graph(pos,self.max_radius)
        edge_vec = pos[edge_src] - pos[edge_dst]
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis=function_type_main,
            cutoff=True
        ).mul(self.number_of_basis**0.5)
        #print(edge_length_embedded.shape)
        #self.plot_edge_length_embedded(edge_length_embedded)
        edge_weight_cutoff = soft_unit_step(5 * (1 - edge_length / self.max_radius))
        # 计算球谐函数
        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, True, normalization='component')
        # 计算 q, k, 如果需要开启多层transformer，则参考embedded net里面equitransformer模块里面的用法，但是输出和输入的不可约表示必须要一致
        q = self.h_q(f)
        k = self.tp_k(f[edge_src], edge_sh, self.fc_k(edge_length_embedded))
        v = self.tp_v(f[edge_src], edge_sh, self.fc_v(edge_length_embedded))
        exp = edge_weight_cutoff[:, None] * self.dot(q[edge_dst], k).exp()
        z = scatter(exp, edge_dst, dim=0, dim_size=len(f))
        z[z == 0] = 1
        alpha = (exp / z[edge_dst])
        f_new = scatter(alpha.relu().sqrt() * v, edge_dst, dim=0, dim_size=len(f))
        f_new = self.linear_layer(f_new)
        f_new = self.non_linearity(f_new)
        f_new = self.linear_layer_2(f_new)
        return f_new
    def plot_edge_length_embedded(self, edge_length_embedded):
        """
        绘制 edge_length_embedded 矩阵
        """
        plt.figure(figsize=(10, 6))
        plt.imshow(edge_length_embedded.cpu().detach().numpy(), cmap='viridis', aspect='auto')
        plt.colorbar(label='Value')
        plt.xlabel('Basis Index')
        plt.ylabel('Edge Index')
        plt.title('Edge Length Embedded Matrix')
        plt.show()
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
            x = F.silu(x)
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
    
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob = dropout_value):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear2 = nn.Linear(out_dim, out_dim)
        
        # 残差连接的输入和输出维度必须相同
        self.match_dim = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.tanh(out)
        out = self.dropout(out)
        out = self.linear2(out)
        
        # 将残差添加到输出
        out += self.match_dim(residual)
        return out
class WeightedDynamicMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_prob):
        super(WeightedDynamicMLP, self).__init__()
        
        # 共享的特征提取 MLP（对每个特征）
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # 动态构建特征提取 MLP
        feature_layers = []
        in_dim = input_dim
        #for hidden_dim in hidden_dims:
           #feature_layers.append(ResidualBlock(in_dim, hidden_dim, dropout_prob))
            #in_dim = hidden_dim
        for hidden_dim in hidden_dims:
            feature_layers.append(nn.Linear(in_dim, hidden_dim))  # 不使用残差
            feature_layers.append(nn.Tanh())  # 激活函数
            feature_layers.append(nn.Dropout(p=dropout_prob))  # Dropout
            in_dim = hidden_dim
        self.feature_mlp = nn.Sequential(*feature_layers)
        
        # 用于计算权重的 MLP（对每个特征）
        self.weight_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),  # 输入到第一个隐藏层
            nn.SiLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dims[0], 1)  # 输出权重
        )
        
        # 最终输出映射
        final_layers = []
        #for i in range(len(hidden_dims)-1):
            #final_layers.append(ResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout_prob))
        
        final_layers.append(nn.Linear(hidden_dims[-1], output_dim))  # 输出层
        self.final_mlp = nn.Sequential(*final_layers)

    def forward(self, x):
        # 获取输入特征数量 n
        n = x.size(1)
        
        # 展平输入，将 (1, n) 转换为 (n, 1)
        x = x.view(-1, self.input_dim)
        
        # 提取特征：通过共享 MLP
        features = self.feature_mlp(x)  # 输出 (n, hidden_dim)
        
        # 计算权重：通过权重网络
        raw_weights = self.weight_mlp(x)  # 输出 (n, 1)
        
        # 动态调整 Softmax：根据输入维度 n 调整权重
        weights = F.softmax(raw_weights * (n ** 0), dim=0)  # 使用维度调整
        #weights = F.silu(raw_weights)  # 使用 tanh 进行调整
        
        # 聚合：特征加权求和
        global_feature = (weights * features).sum(dim=0, keepdim=True)  # 输出 (1, hidden_dim)
        
        # 最终输出：通过 MLP 映射到标量
        output = self.final_mlp(global_feature)  # 输出 (1, 1)
        
        return output
#加载数据
class CustomDataset(Dataset):
    def __init__(self, input_file_path, read_file_path, energy_file_path):
        self.input_data = pd.read_hdf(input_file_path)
        self.read_data = pd.read_hdf(read_file_path)
        self.energy_df = pd.read_hdf(energy_file_path)
        
        # 计算能量的最大值和最小值
        self.energy_min = energy_min
        self.energy_max = energy_max
        
        # 最小值-最大值归一化
        self.energy_df['Transformed_Energy'] = (
            self.energy_df['Energy'] - self.energy_min
        ) / (self.energy_max - self.energy_min)
        
        # 创建数据块
        self.input_data_blocks = self._create_data_blocks(self.input_data)
        self.read_data_blocks = self._create_data_blocks(self.read_data)
    
    def _create_data_blocks(self, data):
        # 根据浮动值 128128.0 分割数据块
        blocks = []
        current_block = []
        stop_value = 128128.0  # 分隔符的浮动值
        
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
        # 反归一化：还原到原始能量
        return normalized_energy * (self.energy_max - self.energy_min) + self.energy_min
    
    def restore_force(self, normalized_force):
        # 使用与能量相同的标准差进行反归一化（此处未涉及最小值-最大值归一化）
        return normalized_force * (self.energy_max - self.energy_min)
    
    def __len__(self):
        return len(self.input_data_blocks)
    
    def __getitem__(self, idx):
        """ 获取指定索引的数据块 """
        # 获取 train 数据块和 read 数据块
        input_block = self.input_data_blocks[idx].dropna()  # train、val 输入数据块
        read_block = self.read_data_blocks[idx].dropna()  # read 的数据块
        
        if input_block.empty or read_block.empty:
            return None, None, None  # 处理空块
        
        input_tensor = torch.tensor(input_block.values, dtype=torch.float64, device=device)
        read_tensor = torch.tensor(read_block.values, dtype=torch.float64, device=device)
        
        # 获取目标能量
        target_energy = torch.tensor(self.energy_df['Transformed_Energy'].iloc[idx], dtype=torch.float64, device=device)
        
        return input_tensor, read_tensor, target_energy

    # 加载数据集
train_dataset = CustomDataset('train-fix.h5', 'read_train.h5', 'energy_train.h5')
val_dataset = CustomDataset('val-fix.h5', 'read_val.h5', 'energy_val.h5')
# 数据集块数量
print(f"Train dataset has {len(train_dataset)} blocks.")#确认trainset的数量
print(f"Validation dataset has {len(val_dataset)} blocks.")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
val_blocks = [
    (input_tensor.to(device), read_tensor.to(device), target_energy.to(device))
    for input_tensor, read_tensor, target_energy in [val_dataset[i] for i in range(len(val_dataset))]]
# 设置验证集比例，假设选择20%的数据作为验证集
validation_size = int(0.5 * len(val_blocks))

# 随机选择验证集索引
#random.shuffle(val_blocks)  # 打乱数据顺序
val_data = val_blocks[:validation_size]  # 选择前20%作为验证集
cached_R = None
def compute_R(block, cache=True):#R的定义需要包含S、广义坐标（求导得到x、y、z方向力）、原子序号和环境原子序号
    R = block[:, 1:7].to(device)  # 直接提取需要的列
    R.requires_grad_()  # 设置需要计算梯度
    return R
def compute_T(embed_net, R):
    """
    支持批量输入的 compute_T。
    输入 R 的形状为 (batch_size, num_nodes, 6)。
    输出形状为 (batch_size, num_nodes, output_dim)。
    """
    embed_output = embed_net(R)  # embed_net 需要支持批量输入
    return embed_output

def compute_E_test(R):
    """
    支持批量输入的 compute_E_test。
    输入 R 的形状为 (batch_size, num_nodes, 6)。
    输出形状为 (batch_size, num_nodes * output_dim)。
    """
    # 所有输入 R 都使用 embed_net0
    embed_net = embed_net1
    
    # 计算 T
    T = compute_T(embed_net, R)  # T 的形状为 (batch_size, num_nodes, output_dim)
    
    # 将 T 展平为 (batch_size, num_nodes * output_dim)
    E = T.view(T.size(0), -1)  # 保持 batch_size 维度，展平其他维度
    
    return E
# 定义 Mollifier 函数
def mollifier(pos, sigma=mollifier_sigma):
    return torch.exp(-torch.norm(pos, dim=-1)**2 / (2 * sigma**2)) / (sigma * torch.sqrt(2 * torch.tensor(torch.pi)))
# 初始化嵌入网络和两个主网络
embed_net1 = EmbedNet(input_size=input_size_value, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_value).to(device)
#embed_net2 = EmbedNet(input_size=input_size_value, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_value).to(device)
#embed_net3 = EmbedNet(input_size=input_size_value, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_value).to(device)
#embed_net4 = EmbedNet(input_size=input_size_value, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_value).to(device)
#embed_net0 = EmbedNet(input_size=input_size_value, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_value).to(device)
#main_net1 = MainNet(input_size=10*max_atom , hidden_sizes=main_hidden_sizes1, dropout_rate=dropout_value).to(device)
#main_net2 = MainNet(input_size=embed_size * embed_size*4, hidden_sizes=main_hidden_sizes1, dropout_rate=dropout_value).to(device)
#main_net3 = MainNet(input_size=embed_size * embed_size*4, hidden_sizes=main_hidden_sizes1, dropout_rate=dropout_value).to(device)
#main_net4 = MainNet(input_size=embed_size * embed_size*4, hidden_sizes=main_hidden_sizes1, dropout_rate=dropout_value).to(device)
#main_net0 = MainNet2(input_size=embed_size * embed_size*4, hidden_sizes=main_hidden_sizes2, dropout_rate=dropout_value).to(device)#给虚原子的mainnet
#fit_net = MainNet2(input_size=42, hidden_sizes=main_hidden_sizes3, dropout_rate=dropout_value).to(device)#给权重函数的fit_net
#model = MainNet(input_size= 41 , hidden_sizes=main_hidden_sizes2, dropout_rate=dropout_value).to(device)
model = WeightedDynamicMLP(input_dim_weight, main_hidden_sizes4, 1,dropout_prob=0).to(device)
e3conv_layer = E3Conv(max_radius_main, number_of_basis_main, irreps_input_conv_main,irreps_output_conv_main, hidden_dim=emb_number_main).to(device)
e3conv_layer2 = E3Conv(max_radius_main, number_of_basis_main, irreps_input_conv_main_2,irreps_output_conv_main_2, hidden_dim=emb_number_main_2).to(device)
e3trans = E3_TransformerLayer(max_radius_main, number_of_basis_main, irreps_input_conv_main, irreps_query_main, irreps_key_main, irreps_output_conv_main_2, irreps_sh_transformer, hidden_dim_sh, emb_number_main_2).to(device)
optimizer1 = torch.optim.AdamW(
    list(embed_net1.parameters()) 
    + list(e3trans.parameters())
    ,
    lr=learning_rate,weight_decay=1e-6)
#scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.1, patience=patience_opim)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=patience_opim, gamma = gamma_value)

# 检查是否存在之前保存的模型文件
checkpoint_path = 'combined_model.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    embed_net1.load_state_dict(checkpoint['embed_net1_state_dict'])
    #main_net1.load_state_dict(checkpoint['main_net1'])
    model.load_state_dict(checkpoint['model_state_dict'])
    e3conv_layer.load_state_dict(checkpoint['e3conv_layer_state_dict'])
    e3conv_layer2.load_state_dict(checkpoint['e3conv_layer2_state_dict'])
    e3trans.load_state_dict(checkpoint['e3trans_state_dict'])
    #optimizer1.load_state_dict(checkpoint['optimizer1_state_dict'])
    #scheduler1.load_state_dict(checkpoint["scheduler_state_dict"])
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
    all_nets = [embed_net1,
            e3trans,
            model,e3trans]
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
        with torch.amp.autocast('cuda'):
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
                
                pos = read_tensor[:, [1, 2, 3]]
                pos.requires_grad = True
                
                #all_E = torch.zeros(len(dimensions), embedding_value, dtype=torch.float64, device=device)

                R_values = compute_R(input_tensor)  # 假设 compute_R 返回形状为 (num_samples, 6)

                R_reshaped = R_values.reshape(-1, max_atom, 6)

                # 如果需要，可以将 R_reshaped 转换为 float64 类型并移动到指定设备
                R_reshaped = R_reshaped.to(dtype=torch.float64, device=device)

                # 调用 compute_E_test
                E = compute_E_test(R_reshaped)  # 假设 compute_E_test 返回形状为 (len(dimensions), max_atom * output_dim)

                # 将 E 重塑为 (len(dimensions), embedding_value)
                E = E.view(len(dimensions), -1)  # 假设 embedding_value = max_atom * output_dim
                #all_E = E  # 直接使用 E 填充 all_E

                # 将 all_E 按行堆叠成一个 len(dimensions) x embedding_value 的张量
                E_cat = E

                # 进行后续的计算
                # E_conv = e3conv_layer(E_cat, pos)
                E_conv = e3trans(E_cat, pos).mean()
                #E_conv = E_conv.reshape(1,-1)
                #E_conv = model(E_conv)
                #E_total = E_conv.sum()
                E_mean = E_conv
                E_mean.backward(retain_graph=True)
                fx_pred_conv = train_dataset.restore_force(-pos.grad[:, 0]) / force_coefficient
                fy_pred_conv = train_dataset.restore_force(-pos.grad[:, 1]) / force_coefficient 
                fz_pred_conv = train_dataset.restore_force(-pos.grad[:, 2]) / force_coefficient 
                end_time_it = time.time()
                pos.grad.zero_()
                print(f"Total E_mean for this molecule: {train_dataset.restore_energy(E_mean)}")
                E_sum_all.append(E_mean)
                #print(E_sum_all)
                fx_pred_conv_batch = fx_pred_conv.to(device).view(-1)
                #print(f"froce_x:{fx_pred_conv_batch}")
                fy_pred_conv_batch = fy_pred_conv.to(device).view(-1)
                fz_pred_conv_batch = fz_pred_conv.to(device).view(-1)
            force_loss = (
                criterion(fx_pred_conv_batch, fx_ref.to(device).view(-1)) +
                criterion(fy_pred_conv_batch, fy_ref.to(device).view(-1)) +
                criterion(fz_pred_conv_batch, fz_ref.to(device).view(-1))) / 3
            force_rmse = ((
                criterion_2(fx_pred_conv_batch, fx_ref.to(device).view(-1)) +
                criterion_2(fy_pred_conv_batch, fy_ref.to(device).view(-1)) +
                criterion_2(fz_pred_conv_batch, fz_ref.to(device).view(-1))) / 3)
            # 计算 Mollifier 正则化项

            grad_energy = torch.autograd.grad(E_conv, pos, create_graph=True)[0]
            phi = mollifier(pos, sigma=mollifier_sigma)
            reg_loss = torch.sum(torch.norm(grad_energy, dim=-1)**2 * phi)
            print(reg_loss.item())
            # 总力的损失
            lambda_reg = lambda_reg_value / b # 正则化系数
            total_force_loss = force_loss + lambda_reg * reg_loss

            # 计算能量损失
            E_sum_tensor = torch.tensor(E_sum_all, device=device, requires_grad=True).view(-1)
            energy_loss = criterion(E_sum_tensor, target_energies)
            print(E_sum_tensor , target_energies)
            energy_rmse = train_dataset.restore_force(energy_loss ** 0.5)
            batch_energy_loss += energy_loss.item()

            # 总损失
            total_loss = (a * energy_loss + b * total_force_loss)

            # 使用 GradScaler 进行反向传播和优化
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=max_norm_value)
            scaler.step(optimizer1)
            scaler.update()
            # 学习率调整
            scheduler1.step()
            current_lr1 = scheduler1.get_last_lr()
            end_time_batch = time.time()
            print(f"Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, "
                f"Total Loss: {total_loss}, Energy RMSE:{energy_rmse}, Force Loss: {total_force_loss}, Force RMSE:{force_rmse}, "
                f"Learning Rate: {current_lr1[0]}",f"batch time: {end_time_batch - start_time_batch:.2f} seconds")
            total_energy_loss_val = 0.0
            total_force_loss_val = 0.0
            embed_net1.eval()
            e3trans.eval()
            model.eval()
            #with torch.no_grad():
            E_sum_all_val = []
            target_E_all_val = []
            for input_tensor, read_tensor, target_energy in val_data:  # 使用预加载的数据
                if input_tensor is None or read_tensor is None or target_energy is None:
                    continue  # 跳过空块
                input_tensor = input_tensor.to(device)
                read_tensor = read_tensor.to(device)
                pos_val = read_tensor[:, [1, 2, 3]]
                pos_val.requires_grad = True
                fx_ref_val = read_tensor[:, 5] * force_shift_value  # x 方向参考力
                fy_ref_val = read_tensor[:, 6] * force_shift_value  # y 方向参考力
                fz_ref_val = read_tensor[:, 7] * force_shift_value  # z 方向参考力
                # 获取所有维度的信息
                dimensions_val = input_tensor[:, 0].unique().tolist()

                # 预分配一个大张量，假设你知道最终结果的形状
                # 比如这里假设是 40 x 450 的矩阵
                #num_dimensions = len(dimensions_val)  # 维度的数量
                embedding_size = embedding_value  # 假设每个维度的嵌入大小是 450

                #all_E_val = torch.zeros(num_dimensions, embedding_size, dtype=torch.float64, device=device)

                R_values_val = compute_R(input_tensor)  # 假设 compute_R 可以一次性处理所有数据
                # 将 R_values 重塑为 (len(dimensions), max_atom, 6)
                R_reshaped_val = R_values_val.reshape(-1, max_atom, 6)
                R_reshaped_val = R_reshaped_val.to(dtype=torch.float64, device=device)


                # 调用 compute_E_test
                E_val = compute_E_test(R_reshaped_val)  # 假设 compute_E_test 返回形状为 (len(dimensions), max_atom * output_dim)

               # 将 E 重塑为 (len(dimensions), embedding_value)
                E_val = E_val.view(-1, embedding_value)  # 假设 embedding_value = max_atom * output_dim
                #all_E_val = E_val  # 直接使用 E 填充 all_E
                E_cat_val  = E_val
                # 进行后续计算
                # E_conv_val = e3conv_layer(E_cat_val, pos_val)
                E_conv_val = e3trans(E_cat_val, pos_val).mean()
                #E_conv_val = E_conv_val.reshape(1,-1)
                #E_conv_val = model(E_conv_val)
                #E_total_val = E_conv_val.sum()
                E_mean_val = E_conv_val
                E_mean_val.backward(retain_graph=True)
                E_sum_all_val.append(E_mean_val)
                target_E_all_val.append(target_energy)
                print(f"Total E_sum_val for this molecule: {val_dataset.restore_energy(E_mean_val.item())}")
                fx_pred_conv_val = val_dataset.restore_force(-pos_val.grad[:, 0]) / force_coefficient
                fy_pred_conv_val = val_dataset.restore_force(-pos_val.grad[:, 1]) / force_coefficient
                fz_pred_conv_val = val_dataset.restore_force(-pos_val.grad[:, 2]) / force_coefficient

                fx_pred_conv_batch_val = fx_pred_conv_val.to(device).view(-1)
                fy_pred_conv_batch_val = fy_pred_conv_val.to(device).view(-1)
                fz_pred_conv_batch_val = fz_pred_conv_val.to(device).view(-1)      
            fx_ref_val = fx_ref_val.to(device).view(-1)
            fy_ref_val = fy_ref_val.to(device).view(-1)
            fz_ref_val = fz_ref_val.to(device).view(-1)
            force_loss_val = ((
                criterion_2(fx_pred_conv_batch_val, fx_ref_val) +
                criterion_2(fx_pred_conv_batch_val, fy_ref_val) +
                criterion_2(fx_pred_conv_batch_val, fz_ref_val)) / 3)
            E_sum_val_tensor = torch.tensor(E_sum_all_val, device=device,requires_grad=True).view(-1)
            target_E_all_val_tensor = torch.tensor(target_E_all_val, device=device,requires_grad=True).view(-1)
            energy_loss_val = val_dataset.restore_force(criterion(E_sum_val_tensor, target_E_all_val_tensor)**0.5)
            total_energy_loss_val = energy_loss_val.item()
            total_force_loss_val = force_loss_val.item()
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
            embed_net1.train()
            e3trans.train()
            model.train()
            # 每 n个 epoch 保存一次模型
        if batch_count % 80 == 0:
            torch.save({
                'embed_net1_state_dict': embed_net1.state_dict(),
                'model_state_dict': model.state_dict(),
                'e3conv_layer_state_dict': e3conv_layer.state_dict(),
                'e3conv_layer2_state_dict': e3conv_layer2.state_dict(),
                'e3trans_state_dict': e3trans.state_dict(),
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
    epoch_energy_loss += batch_energy_loss / (batch_idx + 1)
    epoch_force_loss += batch_force_loss / (batch_idx + 1)
    loss_out_df = pd.DataFrame(loss_out)
    loss_out_df.to_csv(f'epoch_{epoch}_batch_count_{batch_count}_loss.csv', index=False)
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
        'model_state_dict': model.state_dict(),
        'e3conv_layer_state_dict': e3conv_layer.state_dict(),
        'e3conv_layer2_state_dict': e3conv_layer2.state_dict(),
        'e3trans_state_dict': e3trans.state_dict(),
        'optimizer1_state_dict': optimizer1.state_dict(),
        "scheduler_state_dict": scheduler1.state_dict(),
        "a": a, 
        "b": b, 
        "batch_count": batch_count,}, checkpoint_path)
result_df = pd.DataFrame(results)
result_df.to_csv('results.csv', index=False)