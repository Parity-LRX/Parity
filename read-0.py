import pandas as pd

def process_csv_file(input_csv, output_h5, output_csv):
    # 读取CSV文件
    df = pd.read_csv(input_csv)
    
    # 删除第二列为0的行
    df_filtered = df[df.iloc[:, 1] != 0]
    
    # 保存过滤后的数据到新的CSV文件
    df_filtered.to_csv(output_csv, index=False)
    
    # 保存为 HDF5 文件
    with pd.HDFStore(output_h5) as store:
        store['data'] = df_filtered

# 处理 train.csv 和 val.csv 文件
process_csv_file('train.csv', 'train-0.h5', 'train-0.csv')
process_csv_file('val.csv', 'val-0.h5', 'val-0.csv')
