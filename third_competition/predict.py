# import os
# def read_path(root):
#     path = []
#     dir = os.listdir(root)
#     for i in dir:
#         images = os.listdir(root + '/' + i)
#         r = root + '/' + i
#         path.extend([(r + '/' + image) for image in images if image.endswith('.jpg')])
#     return path
#
# r = read_path('D:/大三上/机器学习框架/third_competition/picture')



import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import os
from collections import defaultdict
import argparse
import torch.optim as optim
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import torchvision.models as models
import pandas as pd


captcha_word = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        'q': 10, 'w': 11, 'e': 12, 'r': 13, 't': 14, 'y': 15, 'u': 16, 'i': 17, 'o': 18, 'p': 19, 'a': 20, 's': 21,
        'd': 22, 'f': 23, 'g': 24, 'h': 25, 'j': 26, 'k': 27, 'l': 28, 'z': 29, 'x': 30, 'c': 31, 'v': 32, 'b': 33,
        'n': 34, 'm': 35,
        'Q': 36, 'W': 37, 'E': 38, 'R': 39, 'T': 40, 'Y': 41, 'U': 42, 'I': 43, 'O': 44, 'P': 45, 'A': 46, 'S': 47,
        'D': 48, 'F': 49, 'G': 50, 'H': 51, 'J': 52, 'K': 53, 'L': 54, 'Z': 55, 'X': 56, 'C': 57, 'V': 58, 'B': 59,
        'N': 60, 'M': 61
    }
inverse_dic={}
for key,val in captcha_word.items():
    inverse_dic[val]=key

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
        target = image_path.split('.jpg')[0].split('/')[-1] # 根据路径中文件名，找到图片的类别，并对应到指定标签

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

def get_test_data_loader():
    all_path = read_path('D:/大三上/机器学习框架/third_competition/t_picture')
    dataset = Yanzhengma(all_path, transforms=transform_test)
    # for image,target,label in dataset:
    #    print(label)
    return DataLoader(dataset, batch_size=1, shuffle=True)


# 定义是否使用GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def predict():
    net = models.resnet18().to(device)
    net.eval()
    net.load_state_dict(torch.load('restNet11.pkl'))
    print("load resNet net.")

    test_dataloader = get_test_data_loader()
    result = {}#dict.fromkeys(range(20000), list(range(63,68)))
    
    for i, data in enumerate(test_dataloader):
        image, target, label = data
        #print(label)
        image = image.to(device)
        #target = target.to(device)
        outputs = net(image)
        _, predict = torch.max(outputs.data, 1)
        predict = predict.numpy()
        #print(predict[0])
        if int(label[0][0].split('_')[0]) not in result:
            result[int(label[0][0].split('_')[0])] = list(range(63,68))
            result[int(label[0][0].split('_')[0])][int(label[0][0].split('_')[-1])] = inverse_dic[predict[0]]
        else:
            result[int(label[0][0].split('_')[0])][int(label[0][0].split('_')[-1])] = inverse_dic[predict[0]]
            #result
        #result[int(label[0][0].split('_')[0])][int(label[0][0].split('_')[-1])] = inverse_dic[predict[0]]
        #pre.append(predict[0])
    #print(result)
        
    return result

result = predict() 

pre = [0 for x in range(20000)]
for i in range(20000):
    s = ''
    print(i)
    for j in range(5):
        print(j)
        s = s + result[i][j]
    #print(s)
    pre[i] = (s)
   
    
num = [x for x in range(20000)]
d = {'id':pd.Series(num), 'y':pd.Series(pre)}
result = pd.DataFrame(d)
result.to_csv("resNet1.csv", index=False)
for i in pre:
    print(i)