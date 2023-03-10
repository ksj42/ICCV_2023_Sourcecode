
import os
import natsort
import cv2
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.model_selection import train_test_split

from .config import *

dir_model = str(Path(os.getcwd()).parent)

NUM_PATIENTS=15

def png_to_numpy(path):
    img_size=512
    file_list = os.listdir(path)
    file_list_py = [file for file in file_list if file.endswith('.png')]
    file_list_py=natsort.natsorted(file_list_py)
    if file_list_py:
        sample=cv2.imread(path+file_list_py[0])
        if sample.shape[0] !=img_size or sample.shape[1] !=img_size:
            sample=np.zeros(img_size*img_size*3)
            sample=sample.reshape(1,img_size,img_size,3)
        else:
            sample=sample.reshape(1,sample.shape[0],sample.shape[1],sample.shape[2])
        npDICOM=np.array(sample)
        for i in range(len(file_list_py)-1):
            sample=cv2.imread(path+file_list_py[i+1])
            if sample.shape[0] !=img_size or sample.shape[1] !=img_size:
                continue
            else:
                sample=sample.reshape(1,sample.shape[0],sample.shape[1],sample.shape[2])
            npDICOM = np.concatenate((npDICOM, sample), axis=0)
        npDICOM=npDICOM.reshape(1,npDICOM.shape[0],npDICOM.shape[1],npDICOM.shape[2],npDICOM.shape[3])
        return npDICOM
    else:
        return False
    
def crop_image(x_train, y_train):
    x_train=x_train[:, :,106:394,:,1]
    y_train=y_train[:, :,106:394,:,1]
    return x_train, y_train

def fliplr_image(x_train, y_train):
    lst_x_train_fliplr_total=[]
    lst_y_train_fliplr_total=[]
    for i in range(x_train.shape[0]):
        lst_x_train_fliplr = []
        lst_y_train_fliplr = []
        for j in range(x_train.shape[1]):
            lst_x_train_fliplr.append(np.fliplr(x_train[i][j]))
            lst_y_train_fliplr.append(np.fliplr(y_train[i][j]))
        lst_x_train_fliplr_total.append(lst_x_train_fliplr)
        lst_y_train_fliplr_total.append(lst_y_train_fliplr)
    x_train_fliplr = np.array(lst_x_train_fliplr_total)
    y_train_fliplr = np.array(lst_y_train_fliplr_total)
    concate_x_train = np.concatenate((x_train, x_train_fliplr), axis=0)
    concate_y_train = np.concatenate((y_train, y_train_fliplr), axis=0)
    return concate_x_train, concate_y_train

def save_train_val_test(x_train, y_train, x_val, y_val, x_test, y_test):
    np.save(dir_model+r'/CT_3D_unet_dataset/total/x_train.npy', x_train)
    np.save(dir_model+r'/CT_3D_unet_dataset/total/y_train.npy', y_train)
    np.save(dir_model+r'/CT_3D_unet_dataset/total/x_val.npy', x_val)
    np.save(dir_model+r'/CT_3D_unet_dataset/total/y_val.npy', y_val)
    np.save(dir_model+r'/CT_3D_unet_dataset/total/x_test.npy', x_test)
    np.save(dir_model+r'/CT_3D_unet_dataset/total/y_test.npy', y_test)

def load_train_val_test():
    x_train = np.load(dir_model+r'/CT_3D_unet_dataset/total/x_train.npy')
    y_train = np.load(dir_model+r'/CT_3D_unet_dataset/total/y_train.npy')
    x_val = np.load(dir_model+r'/CT_3D_unet_dataset/total/x_val.npy')
    y_val = np.load(dir_model+r'/CT_3D_unet_dataset/total/y_val.npy')
    x_test = np.load(dir_model+r'/CT_3D_unet_dataset/total/x_test.npy')
    y_test = np.load(dir_model+r'/CT_3D_unet_dataset/total/y_test.npy')
    return x_train, y_train, x_val, y_val, x_test, y_test

def create_train_val_test(path):

    if os.path.exists(dir_model+r'/CT_3D_unet_dataset/total/x_train.npy'):
        return load_train_val_test()

    CT_img_path = path
    CT_img_list = os.listdir(CT_img_path)
    x_train_path=CT_img_path+'/'+CT_img_list[0]+'/x_train/'
    y_train_path=CT_img_path+'/'+CT_img_list[0]+'/y_train/'
    x_train_np=png_to_numpy(x_train_path)
    y_train_np=png_to_numpy(y_train_path)
    x_train=x_train_np
    y_train=y_train_np
    for i in range(1,len(CT_img_list)):
        if i==NUM_PATIENTS:
            break
        y_train_path=CT_img_path+'/'+CT_img_list[i]+'/y_train/'
        x_train_path=CT_img_path+'/'+CT_img_list[i]+'/x_train/'
        y_train_np=png_to_numpy(y_train_path)
        x_train_np=png_to_numpy(x_train_path)
        y_train = np.concatenate((y_train, y_train_np), axis=0) 
        x_train = np.concatenate((x_train, x_train_np), axis=0)
    x_train,y_train = crop_image(x_train, y_train)
    concate_x_train, concate_y_train = fliplr_image(x_train, y_train)
    x_train, x_test, y_train, y_test = train_test_split(concate_x_train, concate_y_train, test_size=0.15, shuffle=True, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.1, shuffle=True, random_state=42)
    save_train_val_test(x_train, y_train, x_val, y_val, x_test, y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test

def delete_no_image(x, y, mode):
    x_len = len(x)
    zero_inx = []
    for i in range(x_len):
        y_sum = np.sum(y[i])
        if y_sum==0:
            zero_inx.append(i)
    y = np.delete(y,zero_inx,0)
    x = np.delete(x,zero_inx,0)

    #save index for result's visualization
    np_zero_inx = np.array(zero_inx)
    np.save(dir_model+r'/CT_3D_unet_dataset/index/{}_inx.npy'.format(mode), np_zero_inx)
    return x, y

def dimension_reduction(x_train, y_train, x_val, y_val, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0]*x_train.shape[1], x_train.shape[2], x_train.shape[3])
    x_val = x_val.reshape(x_val.shape[0]*x_val.shape[1], x_val.shape[2], x_val.shape[3])
    x_test = x_test.reshape(x_test.shape[0]*x_test.shape[1], x_test.shape[2], x_test.shape[3])
    y_train = y_train.reshape(y_train.shape[0]*y_train.shape[1], y_train.shape[2], y_train.shape[3])
    y_val = y_val.reshape(y_val.shape[0]*y_val.shape[1], y_val.shape[2], y_val.shape[3])
    y_test = y_test.reshape(y_test.shape[0]*y_test.shape[1], y_test.shape[2], y_test.shape[3])
    return x_train, y_train, x_val, y_val, x_test, y_test

def additional_process_3d(x_train, y_train, x_val, y_val, x_test, y_test):
    x_train, y_train, x_val, y_val, x_test, y_test = dimension_reduction(x_train, y_train, x_val, y_val, x_test, y_test)
    x_train, y_train = delete_no_image(x_train, y_train, mode='train')
    x_val, y_val = delete_no_image(x_val, y_val, mode='val')
    x_test, y_test = delete_no_image(x_test, y_test, mode='test')
    return True

##
## 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

def model_save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

def model_load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    return net, optim, epoch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # 정규화
        label = label/255.0
        input = input/255.0

        # 이미지와 레이블의 차원 = 2일 경우(채널이 없을 경우, 흑백 이미지), 새로운 채널(축) 생성
        if label.ndim == 3:
            label = label[:, :, : , np.newaxis]
        if input.ndim == 3:
            input = input[:, :, :, np.newaxis]

        data = {'input': input, 'label': label}

        #print('In Dataset',data['input'].shape)
        
        # transform이 정의되어 있다면 transform을 거친 데이터를 불러옴
        if self.transform:
            data = self.transform(data)

        return data
    
# 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        #label = label.transpose((2, 0, 1)).astype(np.float32).contiguous()
        #input = input.transpose((2, 0, 1)).astype(np.float32)
        #print('ToTensor before', input.shape)
        label = label.transpose((3,0,1,2)).astype(np.float32)
        input = input.transpose((3,0,1,2)).astype(np.float32)
        #print('ToTensor after', input.shape)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()
    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma       
        return FocalTversky