import pandas as pd
import numpy as np

def euclidean_distance(row1, row2):
    """计算两个点之间的欧几里得距离"""
    return np.sqrt((row1[0] - row2[0])**2 + (row1[1] - row2[1])**2 + (row1[2] - row2[2])**2)

def s(rij, rcs, rc):
    """权重函数 s(rij) 的定义,这里参考是的deepmd原论文的形式,可以改成任何形式,我认为这个函数属于H2(Ω)空间即可"""
    if rij < rcs:
        return 1 / rij
    elif rcs <= rij < rc:
        return (1 / rij) * (0.5 * np.cos(np.pi * (rij - rcs) / (rc - rcs)) + 0.5)
    else:
        return 0

def process_chunk(data, rcs, rc):
    """处理数据块，计算每一行的距离"""
    results = []
    data = data.apply(pd.to_numeric, errors='coerce')

    num_rows = data.shape[0]

    for i in range(num_rows):
        x_i, y_i, z_i = data.iloc[i, 1], data.iloc[i, 2], data.iloc[i, 3]
        index = data.iloc[i, 0]
        A = data.iloc[i, 4]

        for j in range(num_rows):
            if i != j:  # 不计算自身的距离
                x_j, y_j, z_j = data.iloc[j, 1], data.iloc[j, 2], data.iloc[j, 3]
                distance = euclidean_distance([x_i, y_i, z_i], [x_j, y_j, z_j])
                S = s(distance, rcs, rc)
                Aenv = data.iloc[j, 4]  # 邻近原子的A值

                # 构造新的数据行，并添加名为 'append' 的列，值全为 0
                new_row = [
                    index,  # 添加第i行的浮点数
                    S,
                    S*x_i / distance if distance != 0 else 0,
                    S*y_i / distance if distance != 0 else 0,
                    S*z_i / distance if distance != 0 else 0,
                    A,
                    Aenv
                ]
                results.append(new_row)
    
    # 将结果转换为DataFrame，添加 'append' 列
    result_df = pd.DataFrame(results, columns=['Dimension', 'S', 'X/Distance', 'Y/Distance', 'Z/Distance', 'A', 'Aenv'])
    return result_df

def process_traindata(train_file, val_file, rcs=8.0, rc=9.0):
    """处理训练和测试数据文件"""
    for file, output_file, h5_file in zip([train_file, val_file], ['train.csv', 'val.csv'], ['train.h5', 'val.h5']):
        # 读取数据文件
        data = pd.read_hdf(file, header=0, skip_blank_lines=False)

        data_chunks = []
        current_chunk = []
        
        for index, row in data.iterrows():
            if 128128 in row.values:  # 判断行中是否含有128128
                if current_chunk:
                    data_chunks.append(pd.DataFrame(current_chunk, columns=data.columns))
                    current_chunk = []  # 清空当前数据块
            else:
                current_chunk.append(row.values)

        if current_chunk:  # 处理最后一个数据块
            data_chunks.append(pd.DataFrame(current_chunk, columns=data.columns))

        all_results = []
        for chunk in data_chunks:
            if not chunk.empty:
                result = process_chunk(chunk.dropna(), rcs, rc)
                if not result.empty:  # 只添加非空结果
                    all_results.append(result)
                    # 在每个数据块之间插入 128128串作为分隔
                    all_results.append(pd.DataFrame([[128128] * result.shape[1]], columns=result.columns))

        # 过滤掉空的DataFrame
        all_results = [df for df in all_results if not df.empty]

        # 合并所有结果
        final_results_with_stop = pd.concat(all_results, ignore_index=True, sort=False)

        # 保存为 CSV 文件
        final_results_with_stop.to_csv(output_file, index=False)

        # 保存为 HDF5 文件
        with pd.HDFStore(h5_file) as store:
            store['data'] = final_results_with_stop

process_traindata('read_train.h5', 'read_val.h5')
