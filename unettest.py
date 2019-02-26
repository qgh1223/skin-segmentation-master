from keras.models import load_model
from keras.preprocessing.image import img_to_array,load_img
import keras.backend as K
import os
import numpy as np
import matplotlib.pyplot as plt
from model.resunet import UResNet

def dice_coef(y_true,y_pred):
    y_true_f=K.flatten(y_true)
    y_pred_f=K.flatten(y_pred)
    intersection=K.sum(y_true_f*y_pred_f)
    return (2.0*intersection)/(K.sum(y_true_f)+K.sum(y_pred_f))

def dice_coef_loss(y_true,y_pred):
    return -dice_coef(y_true,y_pred)
model=UResNet((256,256,3),16,2,block_num=5,scse=True)

model.load_weights('unet_model_50h_scse.h5')
BASE_DIR='D:/YeJQ/skinmask/'
IMG_DIR=BASE_DIR+'ISBI2016_ISIC_Part1_Training_Data/'
MASK_DIR=BASE_DIR+'ISBI2016_ISIC_Part1_Training_GroundTruth/'
imgpathlist=os.listdir(IMG_DIR)
maskpathlist=os.listdir(MASK_DIR)
test_size=50
rndidlist=np.random.randint(0,len(imgpathlist),test_size)
imglist=np.zeros((test_size,256,256,3))
masklist=np.zeros((test_size,256,256,3))
for i,rndid in enumerate(rndidlist):
    imgpath=IMG_DIR+imgpathlist[rndid]
    maskpath=MASK_DIR+maskpathlist[rndid]
    img=img_to_array(load_img(imgpath,target_size=(256,256)))
    mask=img_to_array(load_img(maskpath,target_size=(256,256)))
    imglist[i]=img
    masklist[i]=mask
predictresult=model.predict(imglist)
predictmask=np.zeros((test_size,256,256))
for i in range(test_size):
    for j in range(256):
        for k in range(256):
            index1=np.argmax(predictresult[i][j][k])
            predictmask[i][j][k]=index1
predictmask=np.logical_not(predictmask)
for i in range(test_size):
    plt.subplot(121)
    plt.imshow(masklist[i])
    plt.subplot(122)
    plt.imshow(predictmask[i])
    plt.savefig('result/'+str(i)+'.jpg')