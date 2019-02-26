from skimage.io import imread,imshow
from skimage.transform import resize
import numpy as np
from keras.preprocessing.image import img_to_array,load_img
import os
class ImgGenerator:
    def __init__(self,IMG_ROW,IMG_COL,
                 IMG_DIR,MASK_DIR):
        self.IMG_ROW=IMG_ROW
        self.IMG_COL=IMG_COL
        self.IMG_DIR=IMG_DIR
        self.MASK_DIR=MASK_DIR

    def img_mask_arr(self):
        imglist=[]
        masklist=[]
        maskpathlist=os.listdir(self.MASK_DIR)
        for i,maskpath in enumerate(maskpathlist):
            maskpatharr=maskpath.split('_')
            imgpath=self.IMG_DIR+maskpatharr[0]+'_'+maskpatharr[1]+'.jpg'

            img=img_to_array(load_img(imgpath,
                            target_size=(self.IMG_ROW,self.IMG_COL)))
            imglist.append(img)
            mask=np.zeros((self.IMG_ROW,self.IMG_COL,2))
            mask1=np.zeros((self.IMG_ROW,self.IMG_COL,1))
            mask_=img_to_array(load_img(self.MASK_DIR+maskpath,grayscale=True,
                                      target_size=(self.IMG_ROW,self.IMG_COL)))/255
            mask[:,:,0]=np.logical_or(mask1,mask_==0)[:,:,0]
            mask[:,:,1]=np.logical_or(mask1,mask_==1)[:,:,0]

            masklist.append(mask)
        return np.asarray(imglist),np.asarray(masklist)

    def data_gen(self,imglist,masklist,imagedatagenerator,
                 augment,
                 gen_batch_size=10,
                 batch_size=15):
        while(True):
            imglist1=np.zeros((batch_size,self.IMG_ROW,self.IMG_COL,3))
            masklist1=np.zeros((batch_size,self.IMG_ROW,self.IMG_COL,2))
            rndidlist=np.random.randint(0,len(imglist1),batch_size)
            for i,rndid in enumerate(rndidlist):
                imglist1[i]=imglist[rndid]
                masklist1[i]=masklist[rndid]
            if(augment==True):
                imglist2=np.zeros((gen_batch_size,self.IMG_ROW,self.IMG_COL,3))
                masklist2=np.zeros((gen_batch_size,self.IMG_ROW,self.IMG_COL,2))
                seed=1337
                imggen=imagedatagenerator.flow(imglist1,batch_size=gen_batch_size,
                                           seed=seed)
                for i,img1 in enumerate(next(imggen)):
                    imglist2[i]=img1
                for i in range(2):
                    maskgen=imagedatagenerator.flow(masklist1[:,:,:,i].reshape(batch_size,self.IMG_ROW,self.IMG_COL,1),
                                                batch_size=gen_batch_size,
                                            seed=seed)
                    for j,mask1 in enumerate(next(maskgen)):
                        #print(mask1.shape)
                        masklist2[j,:,:,i]=mask1[:,:,0]
                yield (imglist2,masklist2)
            else:
                yield (imglist1,masklist1)
