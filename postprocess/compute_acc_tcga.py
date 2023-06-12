from PIL import Image
import numpy as np
from  matplotlib import pyplot as plt
import os
import sys
Image.MAX_IMAGE_PIXELS=None

def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
        return 1
    return (2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin))

path = '/data3/ian/dsmil-wsi/post_process/vit-1-0-th'

LUAD_file = '/data2/ian/LUAD.txt'
LUSC_file = '/data2/ian/LUSC.txt'
LUAD = []
LUSC = []

with open (LUAD_file, 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        ids ,f_name,_,_,_ = line.split('\t')
        if f_name == 'filename':
            continue
        LUAD.append(f_name[:23].strip('.'))
        
with open (LUSC_file, 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        ids ,f_name,_,_,_ = line.split('\t')
        if f_name == 'filename':
            continue
        LUSC.append(f_name[:23].strip('.'))
        
print(LUAD)
print(LUSC)

files = os.listdir(path)
acc_list = []
for f in files:
    file = os.path.join(path,f)
    sample = os.path.join(sample_path,f)
    print(file)
    
    img = Image.open(file).convert('L')
    img = np.array(img)

    try:
        sample_img = Image.open(sample).convert('L')
    except:
        binary = img.copy()
        print(np.unique(binary))
        binary = binary[binary>10]
        if len(binary) == 0:
            print('ACC : 1')
            acc_list.append(1)
        else:
            print('ACC : 0')
            acc_list.append(0)
        continue
        
    sample_img = np.array(sample_img)
    sample_img = np.where(sample_img>127, 1, 0)

    binary = img.copy()
    binary = binary[binary>10]
    print(np.unique(binary))
    if len(binary) == 0:
        print('ACC : 0')
        acc_list.append(0)
    else:
        print('ACC : 1')
        acc_list.append(1)

print(sum(acc_list)/len(acc_list))

