import scipy.io as sio
import numpy as np

class SceneCutter(object):

    def readDivImages(self):
        imagesMat = sio.loadmat('input/image_div.mat')
        images = imagesMat['IMAGES_div']
        return(images)
        
    def getInput(self,patchsize):
        images = self.__Images
        pictNbr = np.random.randint(0,10)
        length,width = images.shape[0:2]
        xPos = np.random.randint(0,length-patchsize)
        yPos = np.random.randint(0,width-patchsize)
        inputPatch = images[xPos:xPos+patchsize,yPos:yPos+patchsize,:,pictNbr]
        maxVal = np.max(images[:,:,:,pictNbr])
        return(inputPatch,maxVal)

    def __init__(self):
        self.__Images = self.readDivImages()
        print('read Images')
