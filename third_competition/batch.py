# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:56:39 2019

@author: Lenovo
"""

import os
import one_hot_encoding as ohe
from PIL import Image
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch.nn import functional as F


transform = T.Compose([
    T.Grayscale(),
    T.ToTensor(),
# =============================================================================
#     T.Resize(224),  # 缩放图片，保持长宽比不变，最短边为 224 像素
#     T.CenterCrop(224),  # 从图片中间切出 224 * 224 的图片 
#     T.ToTensor(),  # 将Image转成Tensor，归一化至 [0.0, 1.0]
#     T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 标准化至 [-1, 1]
# =============================================================================
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
        print(type(target))
        label.append(target)
        target = ohe.encode(target)
        pil_image = Image.open(image_path) #按照路径将图片加载
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
    dataset = Yanzhengma('D:\\大三上\\机器学习框架\\third_competition\\train\\train', transforms=transform)
    return DataLoader(dataset, batch_size=100, shuffle=True)

def get_test_data_loader():
    dataset = Yanzhengma('D:\\大三上\\机器学习框架\\third_competition\\test\\test', transforms=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear((150//8)*(30//8)*64, 1024),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU())
        self.rfc = nn.Sequential(
            nn.Linear(1024, 5*62),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.rfc(out)
        return out



def main():

    num_epochs = 50
    learning_rate = 0.001

    cnn = CNN()
    cnn.train()
    print('init net')
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the Model
    train_dataloader = get_train_data_loader()
    for epoch in range(num_epochs):
        for i, (images, target,label) in enumerate(train_dataloader):
            #images = images.transpose(np,(0,3,1,2))
            print(type(target))
            print(target)
            images = Variable(images)
            target = Variable(target.float())
            predict_labels = cnn(images)
            # print(predict_labels.type)
            # print(labels.type)
            loss = criterion(predict_labels, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print("epoch:", epoch, "step:", i, "loss:", loss.item())
            if (i+1) % 100 == 0:
                torch.save(cnn.state_dict(), "./model.pkl")   #current is model.pkl
                print("save model")
        print("epoch:", epoch, "step:", i, "loss:", loss.item())
    torch.save(cnn.state_dict(), "./50_model.pkl")   #current is model.pkl
    print("save last model")

if __name__ == '__main__':
    main()
    
    