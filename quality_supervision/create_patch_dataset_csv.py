import os
import pandas as pd


# 定义图像目录和类别
data_dir = r'D:\data\Cohort\sample'   # the directory where the patch data that needs to be classified and predicted is located, so it needs to be modified according to the specific situation
csv_save_path = r'D:\data\Cohort'      # the patch of csv file

# 初始化空的数据列表
data = []

# 遍历目录中的每张图像
for img in os.listdir(data_dir):
    data.append((os.path.join(data_dir, img)))  # 将图像文件名和标签组成元组，并添加到标签列表中

# 将数据标签列表转换为DataFrame
df = pd.DataFrame(data, columns=['filename'])

# 保存图像文件名到CSV文件
df.to_csv(csv_save_path+'\\'+'test.csv', index=False)
