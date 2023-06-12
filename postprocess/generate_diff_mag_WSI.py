from PIL import Image
import numpy as np
from  matplotlib import pyplot as plt
import os
from scipy import ndimage as nd
import openslide

mask_path = "/data3/ian/dsmil-wsi/dsmil-wsi/05162023/LR_0.0008_BLOSS_0.9_POS_0_HEAD_2_BLOCK_2_LANDMARK_2048_ms_fusion_11_71/postprocess"
img_path = "/data1/ian/C16_training_small/C16_test_mask/"

store_path = "/data3/ian/dsmil-wsi/dsmil-wsi/05162023/LR_0.0008_BLOSS_0.9_POS_0_HEAD_2_BLOCK_2_LANDMARK_2048_ms_fusion_11_71/postprocess"
Image.MAX_IMAGE_PIXELS=None

files = os.listdir(mask_path)
for f in files:
    try:
        file = os.path.join(img_path,f)
        img = Image.open(file)
        w, h = img.size
                    
        mask_file = os.path.join(mask_path,f)
        mask = Image.open(mask_file)
        print(f)
        print(mask.size)
        mask = mask.resize((w, h))
        print(w,h)
        print(mask.size)
                                                    
        store_file = os.path.join(store_path,f)
        mask.save(store_file)
    except:
        print('Error',f)
