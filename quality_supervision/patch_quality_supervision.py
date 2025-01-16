import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from quality_supervision_model import SimpleCNN
from patch_reading import CustomDataset
from torch.utils.data import DataLoader
import os
import codecs
import csv
import shutil


os.environ["CUDA_VISIBLE_DEVICES"] = '7'  # 设置该程序可见的gpu.CUDA_VISIBLE_DEVICES表示当前可以被python环境程序检测到的显卡


def data_write_csv(file_name, data_list):  # file_name为写入CSV文件的路径，data_list为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=',', quotechar='\'', quoting=csv.QUOTE_MINIMAL)
    for dat in data_list:
        writer.writerow(dat)
    print("保存文件成功，处理结束")


def test(batch_size, test_csv_file, model_path, save_path):
    # 加载模型结构
    model = SimpleCNN()
    # 加载模型参数
    if torch.cuda.is_available():
        model = model.cuda()
    model_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_dict['model_state_dict'])

    test_dataset = CustomDataset(test_csv_file, transform=None, normalization=True)

    # 创建数据加载器
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=False)

    model.eval()
    idx = 0
    save_result = [['Img_filename', 'Label_predicted']]
    with torch.no_grad():
        for inputs in test_data_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # 函数返回两个tensor，第一个是每行的最大值；第二个是每行最大值的索引。

            preds = preds.detach().cpu().numpy()
            source_Img_path = test_data_loader.dataset.data.filename[idx]
            save_result.append([source_Img_path, preds])
            idx = idx+1

            image_files = os.path.basename(source_Img_path)  # 只取出图像文件名
            if preds == 0:  # 表示预测为delete，将该图像copy到预测为delete的文件夹
                target_path = os.path.join(save_path, 'predicted_delete', image_files)
                shutil.copy(source_Img_path, target_path)
            else:
                target_path = os.path.join(save_path, 'predicted_sample', image_files)
                shutil.copy(source_Img_path, target_path)

            print('The predicted lable of ' + '\'' + image_files + '\'' + ' is: {}'.format(preds))

    data_write_csv(save_path + '\\' + 'Predicted_Results.csv', save_result)


if __name__ == '__main__':
    # 加载CSV文件并创建数据集实例
    test_csv_file = r'D:\yyt\data\BingLIproject\Cohort\test.csv'
    model_path = r'D:\yyt\data\BingLIproject\Cohort\best_model.pth'
    save_path = r'D:\yyt\data\BingLIproject\Cohort'

    test(batch_size=1, test_csv_file=test_csv_file, model_path=model_path, save_path=save_path)

