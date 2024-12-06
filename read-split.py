import pandas as pd
import numpy as np

file_path = 'data.xyz'  # 替换为你的数据文件路径
max_atom = 1  # 不用虚原子直接设为1

def extract_data_blocks(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()

    energy_list = []  # 存储归一化后的能量
    raw_energy_list = []  # 存储未归一化的能量
    data_blocks = []
    current_block = []
    energy = None
    previous_line = None  # 用于存储上一行

    # 定义一个包含所需元素符号的列表
    elements = ["C", "H", "O", "N"]  # 可以进一步扩展为整个周期表

    for line in data:
        line = line.strip()
        
        # 判断是否是能量行
        if line.startswith("Properties"):
            # 提取 energy 值，并确保为 float64
            energy = np.float64(line.split("energy=")[1].split()[0])
            raw_energy_list.append(energy)  # 存储原始能量
            
            # 如果上一行存在，获取上一行的第一列数值进行除法操作
            if previous_line is not None:
                first_column_value = np.float64(previous_line.split()[0])  # 获取上一行第一列数值
                # 计算 energy 除以上一行的第一列数值
                energy = energy / first_column_value

            energy_list.append(energy)  # 存储归一化后的能量

        elif any(line.startswith(element) for element in elements):  # 判断是否以元素符号开头
            # 提取分子的坐标和属性
            parts = line.split()
            if len(parts) >= 7:  # 确保至少有 7 列
                position = [
                    np.float64(parts[1]), np.float64(parts[2]), np.float64(parts[3]),
                    np.float64(parts[7]), np.float64(parts[4]), np.float64(parts[5]), np.float64(parts[6])
                ]  # x, y, z, A, Fx, Fy, Fz
                current_block.append(position)

        # 更新上一行内容
        previous_line = line
        
        # 用于判断数据块结束
        if line.isdigit():
            # 如果遇到空行，意味着一个数据块结束
            if current_block and energy is not None:
                data_blocks.append(current_block)
                current_block = []  # 清空当前块
                energy = None  # 重置能量

    # 处理最后一个数据块（如果没有以空行结束）
    if current_block and energy is not None:
        data_blocks.append(current_block)
    
    return data_blocks, energy_list, raw_energy_list


data_blocks, energy_list, raw_energy_list = extract_data_blocks(file_path)

# 确保能量和数据块是一一对应的
data_size = len(data_blocks)
indices = np.arange(data_size)
np.random.shuffle(indices)

train_size = int(0.996 * data_size)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# 拆分数据块和能量
train_data_blocks = [data_blocks[i] for i in train_indices]
train_energy_list = [energy_list[i] for i in train_indices]
train_raw_energy_list = [raw_energy_list[i] for i in train_indices]

val_data_blocks = [data_blocks[i] for i in val_indices]
val_energy_list = [energy_list[i] for i in val_indices]
val_raw_energy_list = [raw_energy_list[i] for i in val_indices]

# 保存归一化后的 energy 到 energy_train.csv 和 energy_val.csv
train_energy_df = pd.DataFrame(np.float64(train_energy_list), columns=['Energy'])
val_energy_df = pd.DataFrame(np.float64(val_energy_list), columns=['Energy'])

# 保存原始 energy 到 raw_energy_train.csv 和 raw_energy_val.csv
train_raw_energy_df = pd.DataFrame(np.float64(train_raw_energy_list), columns=['RawEnergy'])
val_raw_energy_df = pd.DataFrame(np.float64(val_raw_energy_list), columns=['RawEnergy'])

# 保存为 CSV
train_energy_df.to_csv('energy_train.csv', index=False)
val_energy_df.to_csv('energy_val.csv', index=False)
train_raw_energy_df.to_csv('raw_energy_train.csv', index=False)
val_raw_energy_df.to_csv('raw_energy_val.csv', index=False)

# 分别保存 energy 数据到 energy_train.h5 和 energy_val.h5
with pd.HDFStore('energy_train.h5') as store:
    store['train_energy'] = train_energy_df

with pd.HDFStore('energy_val.h5') as store:
    store['val_energy'] = val_energy_df

# 分别保存原始 energy 数据到 raw_energy_train.h5 和 raw_energy_val.h5
with pd.HDFStore('raw_energy_train.h5') as store:
    store['train_raw_energy'] = train_raw_energy_df

with pd.HDFStore('raw_energy_val.h5') as store:
    store['val_raw_energy'] = val_raw_energy_df

# 保存 x, y, z, A 到 read_train.csv 和 read_val.csv，每个数据块之间有一个空行
def save_data_to_csv_and_h5(data_blocks, csv_filename, h5_filename):
    all_data = []
    for block in data_blocks:
        # 如果数据块行数不足 42，则用虚原子补齐
        while len(block) < max_atom:
            x = np.random.uniform(50, 100)
            y = np.random.uniform(50, 100)
            z = np.random.uniform(50, 100)
            A = 0  # 固定为0
            Fx = 0
            Fy = 0
            Fz = 0
            block.append([np.float64(x), np.float64(y), np.float64(z), np.float64(A), np.float64(Fx), np.float64(Fy), np.float64(Fz)])

        for row_number, entry in enumerate(block, start=0):  # 增加行号，从0开始
            all_data.append([np.float64(row_number)] + entry)  # 在每行的开头加上行号，并转换为 float64
        all_data.append([np.float64(128128)])

    # 去掉最后一个多余的空行
    if all_data and all_data[-1] == [np.float64(128128)]:
        all_data.pop()

    data_df = pd.DataFrame(all_data, columns=['Dimension', 'x', 'y', 'z', 'A', 'Fx', 'Fy', 'Fz'])

    # 同时保存为 CSV 和 HDF5
    data_df.to_csv(csv_filename, index=False)

    with pd.HDFStore(h5_filename) as store:
        store['data'] = data_df

# 保存训练和测试数据到 CSV 和 HDF5
save_data_to_csv_and_h5(train_data_blocks, 'read_train.csv', 'read_train.h5')
save_data_to_csv_and_h5(val_data_blocks, 'read_val.csv', 'read_val.h5')
