import pandas as pd
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
from torchvision import transforms


# 创建自定义的数据集类
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, normalization=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.norm = normalization
        self.transform_ToTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img = Image.open(img_name)
        img = np.array(img)

        if self.norm:

            # 通道标准化图像数据
            channel_sum = np.zeros(3)
            channel_sum_squared = np.zeros(3)
            total_pixels = 0

            channel_sum += np.sum(img, axis=(0, 1))  # 累加通道像素值总和
            total_pixels += img.size / 3  # 计算每个通道总像素数

            # 计算通道均值和标准差
            mean = channel_sum / total_pixels
            channel_sum_squared += np.sum(np.square(img-mean), axis=(0, 1))  # 累加通道像素值平方总和
            std = np.sqrt(channel_sum_squared / total_pixels)

            transform = transforms.Normalize(mean, std)

            # img = self.transform_ToTensor(img)
            img = img.transpose(2, 0, 1)
            img = torch.tensor(img, dtype=torch.float32)
            img = transform(img)

        # if self.transform:
        #     img = self.transform(img)

        return img

