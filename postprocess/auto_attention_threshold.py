from PIL import Image
import numpy as np
from  matplotlib import pyplot as plt
import os
import json

Image.MAX_IMAGE_PIXELS=None

def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
        return 1
    return (2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin))

path =  '/data3/ian/dsmil-wsi/dsmil-wsi/05162023/LR_0.0008_BLOSS_0.9_POS_0_HEAD_2_BLOCK_2_LANDMARK_2048_ms_fusion_11_71/postprocess'
sample_path = '/data1/ian/C16_training_small/C16_test_mask/'
store_path = '/data3/ian/dsmil-wsi/dsmil-wsi/05162023/LR_0.0008_BLOSS_0.9_POS_0_HEAD_2_BLOCK_2_LANDMARK_2048_ms_fusion_11_71/segment'
files = os.listdir(path)
dsc_list = []
jsfile = open(os.path.join(path[:path.rfind('/')],'score.json') ,'w')
print(os.path.join(path[:path.rfind('/')],'score.json'))
score = {}

for f in files:
    file = os.path.join(path,f)
    sample = os.path.join(sample_path,f)
    print(file)
    
    img = Image.open(file).convert('L')
    img = np.array(img)

    try:
        sample_img = Image.open(sample).convert('L')
    except:
        continue
        binary = img.copy()
        print(np.unique(binary))
        binary = binary[binary>10]
        if len(binary) == 0:
            print('Max DSC : 1')
            dsc_list.append(1)
        else:
            print('Max DSC : 0')
            dsc_list.append(0)
        continue
        
    sample_img = np.array(sample_img)
    sample_img = np.where(sample_img>127, 1, 0)
    max_dsc = 0
    th = 0
    for i in range(0,61):
        binary = img.copy()
        binary = np.where(binary>i, 1, 0)
        dsc = single_dice_coef(binary, sample_img)
        print(dsc)
        
        if(dsc > max_dsc):
            max_dsc = dsc
            th = i
        
    binary = img.copy()
    binary = np.where(binary>th, 255, 0)
    binary = np.uint8(binary)
    
    binary = Image.fromarray(binary)

    print('Max DSC : ' , str(max_dsc))
    score[f] = max_dsc
    store_file = os.path.join(store_path,f)
    os.makedirs(store_path, exist_ok=True)

    binary.save(store_file)
    dsc_list.append(max_dsc)
    
json.dump(score, jsfile)
print(sum(dsc_list)/len(dsc_list))
