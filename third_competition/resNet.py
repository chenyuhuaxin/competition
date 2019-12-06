# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:43:52 2019

@author: Lenovo
"""
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import os
import argparse
import torch.optim as optim
import one_hot_encoding as ohe
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import torchvision.models as models

# 超参数设置
EPOCH = 135  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128  # 批处理尺寸(batch_size)
LR = 0.1  # 学习率

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
#然后创建一个解析对象；然后向该对象中添加你要关注的命令行参数和选项，每一个add_argument方法对应一个你要关注的参数或选项；最后调用parse_args()方法进行解析；
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model18/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model18/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),  # 维度转化 由32x32x3  ->3x32x32
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # R,G,B每层的归一化用到的均值和方差     即参数为变换过程，而非最终结果。
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def read_path(root):
    path = []
    dir = os.listdir(root)
    for i in dir:
        images = os.listdir(root + '/' + i)
        r = root + '/' + i
        path.extend([(r + '/' + image) for image in images if image.endswith('.jpg')])
    return path

class Yanzhengma():
    def __init__(self, root, transforms=None):
        # dir = os.listdir(root)
        # for i in dir:
        #     images = os.listdir(root + '/' + i)
        #     r = root + '/' + i
        #     self.images = [(r +'/'+image) for image in images if image.endswith('.jpg')]
        #     print(self.images)
        self.images = root
        self.transforms = transforms

    def __getitem__(self, index):  # data[index] 就可以返回图像的数据和类别标签
        image_path = self.images[index]  # 找到序号为index图片的路径

        label = []
        target = int(image_path.split('&')[-1].split('.jpg')[0]) # 根据路径中文件名，找到图片的类别，并对应到指定标签

        label.append(target)
        #target = ohe.encode(target)
        pil_image = Image.open(image_path)  # 按照路径将图片加载

        # pil_image = pil_image.convert('RGB')
        # print(len(pil_image.split()))
        if self.transforms is not None:
            data = self.transforms(pil_image)
        else:
            image_array = np.asarray(pil_image)
            data = torch.from_numpy(image_array)

        return data, target, label

    def __len__(self):
        return len(self.images)


def get_train_data_loader():
    all_path = read_path('D:/大三上/机器学习框架/third_competition/picture')
    dataset = Yanzhengma(all_path, transforms=transform_train)
    print('end')
    return DataLoader(dataset, batch_size=64, shuffle=True)


def get_test_data_loader():
    dataset = Yanzhengma('D:/大三上/机器学习框架/third_competition/test/test', transforms=transform)
    # for image,target,label in dataset:
    #    print(label)
    return DataLoader(dataset, batch_size=1, shuffle=True)

def train():
    # 超参数设置
    EPOCH = 7 #135  # 遍历数据集次数
    pre_epoch = 0  # 定义已经遍历数据集的次数
    BATCH_SIZE = 128  # 批处理尺寸(batch_size)
    LR = 0.1  # 学习率

    net = models.resnet18().to(device)
    # 损失函数为交叉熵，多用于多分类问题,此标准将LogSoftMax和NLLLoss集成到一个类中。
    criterion = nn.CrossEntropyLoss()
    # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    # 定义遍历数据集的次数
    print("Start Training, Resnet-18!")
    trainloader = get_train_data_loader()
    for epoch in range(pre_epoch, EPOCH):  # 从先前次数开始训练
        print('\nEpoch: %d' % (epoch + 1))  # 输出当前次数
        net.train()  # 这两个函数只要适用于Dropout与BatchNormalization的网络，会影响到训练过程中这两者的参数
        # 运用net.train()时，训练时每个min - batch时都会根据情况进行上述两个参数的相应调整，所有BatchNormalization的训练和测试时的操作不同。
        sum_loss = 0.0  # 损失数量
        correct = 0.0  # 准确数量
        total = 0.0  # 总共数量
        for i, data in enumerate(trainloader, 0):
            image, target, label = data

            image = image.to(device)
            target = target.to(device)
            # images = Variable(image)
            # target = Variable(target)         #(torch.LongTensor(target))
            optimizer.zero_grad()
            outputs = net(image)
            #print(target.shape)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()  # 进行单次优化 (参数更新)
            if (i + 1) % 10 == 0:
                print("epoch:", epoch, "step:", i, "loss:", loss.item())
            if (i + 1) % 100 == 0:
                torch.save(net.state_dict(), "restNet11.pkl")  # current is model.pkl
                print("save model")
            print("epoch:", epoch, "step:", i, "loss:", loss.item())
            torch.save(net.state_dict(), "restNet11.pkl")  # current is model.pkl
            print("save last model")


train()