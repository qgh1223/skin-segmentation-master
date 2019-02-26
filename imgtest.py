from keras.preprocessing.image import img_to_array,load_img
import os
import numpy as np
import matplotlib.pyplot as plt
dir='G:/BaiduNetdiskDownload/ISBI2016_ISIC_Part1_Training_GroundTruth/'
mask=img_to_array(load_img(dir+os.listdir(dir)[0]))
mask/=255
print(np.unique(mask))
plt.imshow(mask)
print(mask)