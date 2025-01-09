import os
import pandas as pd


# 定义图像目录和类别
data_dir = r'D:\yyt\data\BingLIproject\Cohort\sample'  # 这里是需要进行分类预测的图像数据所在目录，所以需要根据具体情况修改
csv_save_path = r'D:\yyt\data\BingLIproject\Cohort'  # csv文件保存路径

# 初始化空的数据列表
data = []

# 遍历目录中的每张图像
for img in os.listdir(data_dir):
    data.append((os.path.join(data_dir, img)))  # 将图像文件名和标签组成元组，并添加到标签列表中

# 将数据标签列表转换为DataFrame
df = pd.DataFrame(data, columns=['filename'])

# 保存图像文件名到CSV文件
df.to_csv(csv_save_path+'\\'+'test.csv', index=False)
