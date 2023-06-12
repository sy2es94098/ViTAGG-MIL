from PIL import Image
import numpy as np
from  matplotlib import pyplot as plt
import os
from scipy import ndimage as nd

Image.MAX_IMAGE_PIXELS=None

path = "/data3/ian/dsmil-wsi/dsmil-wsi/05162023/LR_0.0008_BLOSS_0.9_POS_0_HEAD_2_BLOCK_2_LANDMARK_2048_ms_fusion_11_71/output"
store_path = "/data3/ian/dsmil-wsi/dsmil-wsi/05162023/LR_0.0008_BLOSS_0.9_POS_0_HEAD_2_BLOCK_2_LANDMARK_2048_ms_fusion_11_71/postprocess"
os.makedirs(store_path,exist_ok=True)
files = os.listdir(path)
for f in files:
    file = os.path.join(path,f)
    img = Image.open(file)
    img = img.transpose(Image.Transpose.ROTATE_270)
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    
    store_file = os.path.join(store_path,f)
    img.save(store_file)
    
