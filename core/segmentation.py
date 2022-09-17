

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