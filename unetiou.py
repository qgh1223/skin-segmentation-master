from keras.models import load_model
from keras.preprocessing.image import img_to_array,load_img
import keras.backend as K
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from model.resunet import UResNet

model=UResNet((256,256,3),16,2,block_num=5,scse=True)
model.load_weights('unet_model_50h_scse.h5')
IMG_ROW=IMG_COL=256
BASE_DIR='D:/YeJQ/skinmask/'
IMG_DIR=BASE_DIR+'ISBI2016_ISIC_Part1_Training_Data/'
MASK_DIR=BASE_DIR+'ISBI2016_ISIC_Part1_Training_GroundTruth/'
imglist=np.zeros((len(os.listdir(MASK_DIR)),IMG_ROW,IMG_COL,3))
masklist=np.zeros((len(os.listdir(MASK_DIR)),IMG_ROW,IMG_COL),dtype=np.int32)
imgpathlist1=[]
maskpathlist=os.listdir(MASK_DIR)
for i,maskpath in enumerate(maskpathlist):
    maskpatharr=maskpath.split('_')
    imgpath=IMG_DIR+maskpatharr[0]+'_'+maskpatharr[1]+'.jpg'

    img=img_to_array(load_img(imgpath,
                              target_size=(IMG_ROW,IMG_COL)))
    imglist[i]=img
    mask=np.array(img_to_array(load_img(MASK_DIR+maskpath,grayscale=True,
                                target_size=(IMG_ROW,IMG_COL)))/255)

    masklist[i]=np.squeeze(mask)
print(masklist.shape)
masklist=tf.constant(masklist)
predictresult=model.predict(imglist)
predictmasklist=tf.argmax(predictresult,axis=3)
print(predictmasklist.shape)
iou,conf_mat=tf.metrics.mean_iou(masklist,predictmasklist,num_classes=2)
sess = tf.Session()
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())

sess.run(conf_mat)
res = sess.run(iou)

print(res)
