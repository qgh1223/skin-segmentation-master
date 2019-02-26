
import os
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,TensorBoard
import keras.backend as K
import tensorflow as tf
from model.deeplab import deeplabv3_plus
from imagegenerator import data_gen
BASE_DIR='D:/YeJQ/skinmask/'
IMG_DIR=BASE_DIR+'ISBI2016_ISIC_Part1_Training_Data/'
MASK_DIR=BASE_DIR+'ISBI2016_ISIC_Part1_Training_GroundTruth/'
train_imgpathlist,valid_imgpathlist,train_maskpathlist,valid_maskpathlist=train_test_split(os.listdir(IMG_DIR),
                                                                                           os.listdir(MASK_DIR),
                                                                                           test_size=0.1)
print(train_maskpathlist)
IMG_ROW=IMG_COL=256
def dice_coef(y_true,y_pred):
    y_true_f=K.flatten(y_true)
    y_pred_f=K.flatten(y_pred)
    intersection=K.sum(y_true_f*y_pred_f)
    return (2.0*intersection+1.0)/(K.sum(y_true_f)+K.sum(y_pred_f)+1.0)

def dice_coef_loss(y_true,y_pred):
    return -dice_coef(y_true,y_pred)
model=deeplabv3_plus(IMG_ROW,IMG_COL,COLORNUM=1)
model.compile(optimizer='adam',loss='binary_crossentropy',
              metrics=[dice_coef])

modelpath='deeplab_model.h5'

callbacks=[
    ReduceLROnPlateau(monitor='val_loss',patience=15,verbose=1,
                      min_lr=1e-6),
    ModelCheckpoint(modelpath,monitor='val_loss',save_best_only=True,verbose=1),
    TensorBoard(log_dir='logs/'),
]
model.fit_generator(data_gen(IMG_ROW,IMG_COL,IMG_DIR,MASK_DIR,train_imgpathlist,train_maskpathlist,batch_size=10),
                    steps_per_epoch=100,
                    epochs=200,
                    validation_data=data_gen(IMG_ROW,IMG_COL,IMG_DIR,MASK_DIR,valid_imgpathlist,valid_maskpathlist,batch_size=5),
                    validation_steps=10,
                    callbacks=callbacks)
