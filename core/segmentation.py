from segnet import *

from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, AveragePooling2D, MaxPooling2D, UpSampling2D, Input, Reshape
from keras import backend as K
from keras.optimizers import Nadam, Adam, SGD

from keras.metrics import categorical_accuracy, binary_accuracy
#from keras_contrib.losses import jaccard

import tensorflow as tf

class Segmentation:

    def __init__(self):
        pass

    def LBPC_Segmentation(self, _gray):
    
        #lbp filter
        lbp_image = describe(_gray)
        
        #2. Gaussian Blur
        blur = cv2.GaussianBlur(lbp_image, (21,21), 0, 3)
        
        #2. Subtract orignal gray image from lbp, blurred image
        subtracted = np.zeros((lbp_image.shape[0], lbp_image.shape[1]), np.uint8)
        subtracted = _gray - blur 
        
        #3. K-means cluster into 2 colours
        cluster = subtracted.reshape((_gray.shape[0] * _gray.shape[1], 1))
        
        km = KMeans(2)
        labels = km.fit_predict(cluster)
        quant = km.cluster_centers_.astype("uint8")[labels]
        
        final = quant.reshape((_gray.shape[0], _gray.shape[1]))
        thresh = cv2.threshold(quant, 127, 255, cv2.THRESH_BINARY)
        
        return final

    def dullRazor():

        #Gray scale
        grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
        #Black hat filter
        kernel = cv2.getStructuringElement(1,(9,9)) 
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
        #Gaussian filter
        bhg= cv2.GaussianBlur(blackhat,(3,3),cv2.BORDER_DEFAULT)
        #Binary thresholding (MASK)
        ret,mask = cv2.threshold(bhg,10,255,cv2.THRESH_BINARY)
        #Replace pixels of the mask
        dst = cv2.inpaint(img,mask,6,cv2.INPAINT_TELEA)

    #Load the pre-trained model, predict and return the results
    def segNet(img):

        img = [img] #Prediction requires an array of images

        #Load a pre=trained model of SegNet
        model = Segnet.getModelSegnet((192, 256, 3))

        model.compile(optimizer= Adam(learning_rate=0.001), 
              loss = 'binary_crossentropy')

        model.load_weights('../models/skin_lesion.h5')
        
        prediction = model.predict(img, batch_size=16)
        
        return prediction[0]

    #Add a joint technique of SegNet and LBPC/Otsu
    def combinedSegment():
        pass
