# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:15:42 2019

@author: Lenovo
"""

# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import pandas as pd
#import my_dataset
import batch
#from captcha_cnn_model import CNN
import load_data as data

def main():
    NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    ALL_CHAR_SET = NUMBER + ALPHABET
    ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
    MAX_CAPTCHA = 5
    
    cnn = batch.CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('model.pkl'))
    print("load cnn net.")

    test_dataloader = data.get_test_data_loader()

    pre = [0 for x in range(20000)]
    for i, (images, target,label) in enumerate(test_dataloader):
        image = images
        vimage = Variable(image)
        predict_label = cnn(vimage)

        c0 = ALL_CHAR_SET[np.argmax(predict_label[0, 0:ALL_CHAR_SET_LEN].data.numpy())]
        c1 = ALL_CHAR_SET[np.argmax(predict_label[0, ALL_CHAR_SET_LEN:2 * ALL_CHAR_SET_LEN].data.numpy())]
        c2 = ALL_CHAR_SET[np.argmax(predict_label[0, 2 * ALL_CHAR_SET_LEN:3 * ALL_CHAR_SET_LEN].data.numpy())]
        c3 = ALL_CHAR_SET[np.argmax(predict_label[0, 3 * ALL_CHAR_SET_LEN:4 * ALL_CHAR_SET_LEN].data.numpy())]
        c4 = ALL_CHAR_SET[np.argmax(predict_label[0, 4 * ALL_CHAR_SET_LEN:5 * ALL_CHAR_SET_LEN].data.numpy())]
        predict_label = '%s%s%s%s%s' % (c0, c1, c2, c3,c4)
        print(label[0][0])
        print('predict_lable = %s'%predict_label)
        
        pre[int(label[0][0])] = predict_label
        
# =============================================================================
#         true_label = one_hot_encoding.decode(labels.numpy()[0])
#         print('predict_lable = %s'%predict_label)
#         total += labels.size(0)
#         if(predict_label == true_label):
#             correct += 1
#         if(total%200==0):
#             print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
#     print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
# =============================================================================
        
    num = [x for x in range(20000)]
    pre = [str(x) for x in pre]
    d = {'id':pd.Series(num), 'y':pd.Series(pre)}
    result = pd.DataFrame(d)
    result.to_csv("xu1.csv", index=False)
if __name__ == '__main__':
    main()


