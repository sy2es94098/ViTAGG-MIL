from PIL import Image
import numpy as np
from  matplotlib import pyplot as plt
import os
from scipy import ndimage as nd
import openslide

#path = "/data1/ian/C16/C16_training/WSI/1"
#path = "/data1/ian/patch/WSI/C16_WSI/1"
path = "/data1/ian/C16/C16_testing/WSI/0/"
store_path = "dsmil-wsi/with_pos_3/postprocess/"
files = os.listdir(path)

zoom_level = 0

for f in files:
    file = os.path.join(path,f)
    print(f)
    slide = openslide.open_slide(file)
    w, h = slide.level_dimensions[zoom_level]
    print(w,h)
    slide = slide.read_region((0, 0),zoom_level, (w, h))
    
    store_file = os.path.join(store_path,f[:-3]+'png')
    slide.save(store_file)
    print(store_file)
