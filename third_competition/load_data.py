# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:57:45 2019

@author: Lenovo
"""

import os
import one_hot_encoding as ohe
from PIL import Image
import numpy as np
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
import fenge
count = 0


transform = T.Compose([
# =============================================================================
#     T.Grayscale(),
#     T.ToTensor(),
# =============================================================================
    T.Resize(224),  # 缩放图片，保持长宽比不变，最短边为 224 像素
    T.CenterCrop(224),  # 从图片中间切出 224 * 224 的图片 
    T.ToTensor(),  # 将Image转成Tensor，归一化至 [0.0, 1.0]
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 标准化至 [-1, 1]
])

class Yanzhengma():
    def __init__(self, root, transforms=None):
        images = os.listdir(root)
        self.images = [os.path.join(root, image) for image in images if image.endswith('.jpg')]
        self.transforms = transforms

    def __getitem__(self, index): # data[index] 就可以返回图像的数据和类别标签
        image_path = self.images[index] #找到序号为index图片的路径
        label = []
        target = image_path.split(os.sep)[-1].split('.jpg')[0] #根据路径中文件名，找到图片的类别，并对应到指定标签
        print(target)
        label.append(target)
        target = ohe.encode(target)
        pil_image = Image.open(image_path) #按照路径将图片加载
        img_list = fenge.get_crop_imgs(pil_image)
        fenge.save(img_list, target, count)
        #pil_image = pil_image.convert('RGB')
        #print(len(pil_image.split()))
        if self.transforms is not None:
            data = self.transforms(pil_image)
        else:
            image_array = np.asarray(pil_image)
            data = torch.from_numpy(image_array)
            
        return data, target,label 

    def __len__(self):
        return len(self.images)
    
    
def get_train_data_loader():
    dataset = Yanzhengma('D:/大三上/机器学习框架/third_competition/train/train', transforms=transform)
    return DataLoader(dataset, batch_size = 64, shuffle=True)

def get_test_data_loader():
    dataset = Yanzhengma('D:/大三上/机器学习框架/third_competition/test/test', transforms=transform)
    #for image,target,label in dataset:
    #    print(label)
    return DataLoader(dataset, batch_size=1, shuffle=True)
