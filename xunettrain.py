from keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,TensorBoard,LearningRateScheduler
import keras.backend as K
from keras.optimizers import Adam
import tensorflow as tf
from model.resunet import UResNet
from imagegenerator import ImgGenerator

from model.loss import my_iou_metric_2,lovasz_loss
BASE_DIR='D:/YeJQ/skinmask/'
IMG_DIR=BASE_DIR+'ISBI2016_ISIC_Part1_Training_Data/'
MASK_DIR=BASE_DIR+'ISBI2016_ISIC_Part1_Training_GroundTruth/'
training_epochs=600
lr_base = 0.001
lr_power = 0.9

#print(train_maskpathlist)
IMG_ROW=IMG_COL=256

def dice_coef(y_true,y_pred):
    y_true_f=K.flatten(y_true)
    y_pred_f=K.flatten(y_pred)
    intersection=K.sum(y_true_f*y_pred_f)
    return (2.0*intersection)/(K.sum(y_true_f)+K.sum(y_pred_f))

def dice_coef_loss(y_true,y_pred):
    return -dice_coef(y_true,y_pred)


model=UResNet((256,256,3),16,2,block_num=23,scse=True)


model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=[dice_coef])


modelpath='unet_model_xt_50h.h5'
if(os.path.exists(modelpath)):
    model.load_weights(modelpath)


def lr_scheduler(epoch, mode='progressive_drops'):
    '''if lr_dict.has_key(epoch):
        lr = lr_dict[epoch]
        print 'lr: %f' % lr'''

    # drops as progression proceeds, good for sgd
    if epoch > 0.9 * training_epochs:
        lr = 3e-5
    elif epoch > 0.75 * training_epochs:
        lr = 5e-5
    elif epoch > 0.6 *training_epochs:
        lr = 1e-4
    elif epoch > 0.5 * training_epochs:
        lr = 3e-4
    elif epoch > 0.3*training_epochs:
        lr = 5e-4
    elif epoch > 0.2*training_epochs:
        lr = 1e-3
    else:
        lr = 5e-3

    print('lr: %f' % lr)

    return lr
callbacks=[
    #ReduceLROnPlateau(monitor='val_dice_coef',mode = 'max',patience=15,verbose=1,min_lr=1e-6),
    ModelCheckpoint(modelpath,monitor='val_dice_coef',mode='max',
                    save_best_only=True,verbose=1),
    TensorBoard(log_dir='logs/'),
    LearningRateScheduler(lr_scheduler)
]
imggen=ImgGenerator(IMG_ROW,IMG_COL,IMG_DIR,MASK_DIR)
imglist,masklist=imggen.img_mask_arr()
train_imglist,valid_imglist,train_masklist,valid_masklist=train_test_split(imglist,masklist,
                                                                           test_size=0.1)
print(train_imglist.shape)
data_gen_args = dict(rotation_range=40.,
                     width_shift_range=0.3,
                     height_shift_range=0.3,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     fill_mode='nearest'
                     )

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
imggenerator=ImgGenerator(IMG_ROW,IMG_COL,IMG_DIR,MASK_DIR)

model.fit_generator(imggenerator.data_gen(train_imglist,train_masklist,image_datagen,
                                          augment=True,batch_size=12),
                    steps_per_epoch=100,
                    epochs=training_epochs,
                    validation_data=imggenerator.data_gen(valid_imglist,valid_masklist,image_datagen,
                                                          augment=False,
                                                          batch_size=5),
                    validation_steps=10,
                    callbacks=callbacks)
