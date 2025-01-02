import pandas as pd
import numpy as np

max_atom = 5  # 最大原子数

def process_data(input_csv_path, output_csv_path, output_h5_path):
    # 确保所有数据为 float64
    df = pd.read_csv(input_csv_path, dtype=np.float64)
    
    # 数据块分割：以 Dimension 等于 128128 为分割符
    blocks = []
    current_block = []
    
    for _, row in df.iterrows():
        if row['Dimension'] == 128128.0:  # 分割符 (确保是浮点数128128)
            if current_block:
                blocks.append(pd.DataFrame(current_block, columns=df.columns))
                current_block = []
        else:
            current_block.append(row.values.astype(np.float64))  # 强制类型为 float64
    
    # 如果最后一个块不为空，添加到 blocks
    if current_block:  
        blocks.append(pd.DataFrame(current_block, columns=df.columns))

    processed_blocks = []
    
    for block in blocks:
        # 按 Dimension 列分组
        grouped = block.groupby("Dimension")
        processed_group = []

        for dimension, group in grouped:
            if len(group) < max_atom:
                # 不足 max_atom 行，填充 0，第一列保持为 dimension，后面列填充 0
                padding = pd.DataFrame(
                    np.zeros((max_atom - len(group), len(df.columns) - 1), dtype=np.float64),  # 填充 0
                    columns=df.columns[1:]  # 从第二列开始填充 0
                )
                # 创建填充行的 DataFrame，第一列是当前的 dimension 值，后面的列是填充的 0
                padding_with_dimension = pd.DataFrame({
                    "Dimension": np.full((max_atom - len(group),), dimension, dtype=np.float64),  # 第一列为 dimension
                    **{col: padding[col].values for col in df.columns[1:]}  # 后面的列填充 0
                })
                group = pd.concat([group, padding_with_dimension], ignore_index=True)
            elif len(group) > max_atom:
                # 超过 max_atom 行，删除 Rij 最大的行
                group = group.sort_values(by="Rij", ascending=False).iloc[len(group) - max_atom:]
            processed_group.append(group)

        # 合并分组后的数据
        processed_block = pd.concat(processed_group, ignore_index=True)
        processed_blocks.append(processed_block)

    # 合并所有块
    result = pd.concat(processed_blocks, ignore_index=True)
    
    # 在每个块之间插入一个包含128128的分割符
    final_result = []
    for block in processed_blocks:
        final_result.append(block)
        # 插入分割符
        separator = pd.DataFrame(np.full((1, len(df.columns)), 128128.0, dtype=np.float64), columns=df.columns)
        final_result.append(separator)

    # 合并带分割符的所有块
    final_result = pd.concat(final_result, ignore_index=True)

    # 强制转换结果为 float64
    final_result = final_result.astype(np.float64)

    # 输出处理后的数据到 CSV 和 HDF5 文件
    final_result.to_csv(output_csv_path, index=False, float_format='%.16g')  # 输出为 CSV 文件
    final_result.to_hdf(output_h5_path, key='data', mode='w', format='table', complevel=9, complib='zlib')  # 输出为 HDF5 文件

# 处理 train-0.csv 和 val-0.csv
process_data('train.csv', 'train-fix.csv', 'train-fix.h5')
process_data('val.csv', 'val-fix.csv', 'val-fix.h5')
