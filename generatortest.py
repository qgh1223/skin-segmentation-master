import os
from keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.preprocessing.image import img_to_array,load_img

IMG_DIR='G:/BaiduNetdiskDownload/ISBI2016_ISIC_Part1_Training_GroundTruth/'

img=img_to_array(load_img(IMG_DIR+os.listdir(IMG_DIR)[0],target_size=(256,256)))
imgdatagenerator=mask_generator=ImageDataGenerator(featurewise_center=True,
                                                   featurewise_std_normalization=True,
                                                   rotation_range=90.,
                                                   width_shift_range=0.1,
                                                   height_shift_range=0.1,
                                                   zoom_range=0.2)
gen=imgdatagenerator.flow(img)
