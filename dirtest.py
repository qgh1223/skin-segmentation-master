import os

IMG_DIR='ISIC-2017_Training_Data/'
MASK_DIR='ISIC-2017_Training_Part1_GroundTruth/'
imgpathlist=os.listdir(IMG_DIR)
maskpathlist=os.listdir(MASK_DIR)
imgpathlist1=[]
num=0
for i,maskpath in enumerate(maskpathlist):
    maskpatharr=maskpath.split('_')
    imgpath=IMG_DIR+maskpatharr[0]+'_'+maskpatharr[1]+'.jpg'
    print(os.path.exists(imgpath))
    if(not os.path.exists(imgpath)):
        num+=1
print(num)
