import numpy as np
import cv2
import os
import xlrd
import math
from utils.cv import *
from PIL import Image, ImageTk

import xlwt

class PH2:

    path_images = []

    workbook = []
    sheet = []
    
    def __init__(self):
        #Load Excel file

        path_xl = 'D:\\Datasets\PH2Dataset\PH2_dataset.xlsx'
        path_images = 'D:\\Datasets\PH2Dataset\PH2 Dataset images'
        
        self.workbook = xlrd.open_workbook(path_xl)
        self.sheet = self.workbook.sheet_by_index(0)

        self.path_images = path_images
    
    def image_path(self, index):
        return os.path.join(self.path_images, index + '\\' + index +  '_Dermoscopic_Image\\' + index + '.bmp')

    def mask_path(self, index):
        return os.path.join(self.path_images, index + '\\' + index +  '_lesion\\' + index + '_lesion.bmp')

    def load_images_tk(self):
        images = []

        #13 being starting point of names in the ph2 spreadsheet
        for i in range(13, self.sheet.nrows):
            index = self.sheet.cell_value(i, 0)

            path = self.image_path(index)
            img_element = Image.open(path)

            img_element.thumbnail((100, 100))

            images.append(img_element)

        return images

    def load_images(self):
        images = []

        #13 being starting point of names in the ph2 spreadsheet
        for i in range(13, self.sheet.nrows):
            index = self.sheet.cell_value(i, 0)

            path = self.image_path(index)
            img_element = load_img_cv(path)
            
            images.append(img_element)

        return images

    def load_masks(self):
        masks = []

        #13 being starting point of names in the ph2 spreadsheet
        for i in range(13, self.sheet.nrows):
            index = self.sheet.cell_value(i, 0)

            path = self.mask_path(index)
            mask_element = load_img_cv(path)
            
            masks.append(mask_element)

        return masks

    def load_diagnosis(self):
        diagnoses = []

        #List of clinical diagnoses
        for j in range(13, self.sheet.nrows):
            for i in range(2, 5):
                if self.sheet.cell_value(j, i) == 'X':
                   diagnoses.append(self.sheet.cell_value(12, i)) #Gets name of diagnoses (Common Nevus, Atypical Nevus, Melanoma)
                   #diagnoses.append(i - 2)
                   #print(self.sheet.cell_value(12, i))

        return diagnoses

    def load_asymmetry(self):
        asymmetry = []

        #List of clinical diagnoses
        for j in range(13, self.sheet.nrows):
            asymmetry.append(self.sheet.cell_value(j, 5)) #Gets name of diagnoses (Common Nevus, Atypical Nevus, Melanoma)

        return asymmetry
    
    #Gets colour values for each lesion from PH2 xl file
    def load_colours(self):
        colours = []
    
        #List of clinical diagnoses
        for j in range(13, self.sheet.nrows):
            new = []
            for i in range(11, 17):
                if self.sheet.cell_value(j, i) == 'X':
                   #new.append(self.sheet.cell_value(12, i)) #Gets name of colour
                   new.append(i-11) #Numerical value
    
            colours.append(new)
    
        return colours

        workbook.save('output.xls')

class PH2_extend:

    path_images = []
    workbook = []
    sheet = []

    def __init__(self, path_xl, path_images):
        #Load Excel file
        self.workbook = xlrd.open_workbook(path_xl)
        self.sheet = self.workbook.sheet_by_index(0)

        self.path_images = path_images

    def save_glcm():
        size = 7

        for i in range(13, sheet.nrows):
            path_img = os.path.join(self.path_samples, sheet.cell_value(i, 0) + '\\' + sheet.cell_value(i, 0) +  '_Dermoscopic_Image\\' + sheet.cell_value(i, 0) + '.bmp')
            path_glcm = os.path.join(path_samples, sheet.cell_value(i, 0) + '\\' + sheet.cell_value(i, 0) +  '_glcm\\')
            
            if not os.path.isdir(path_glcm):
                os.makedirs(path_glcm)

            img = image.load_img(path_img)
            image_array = image.img_to_array(img, dtype="uint8")/255.0

            gray = rgb2gray(image_array)
            gray *= 255
            img_int = gray.astype(np.uint8)
            glcm_cor = gray.astype(np.uint8)
            glcm_dis = gray.astype(np.uint8)

            for x in range(0, img_int.shape[0] - size):
                for y in range(0, img_int.shape[1] - size):

                    patch = img_int[x:x+size, 
                                    y:y+size]

                    if patch.shape[0] > 0:
                        glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
                        glcm_cor[x,y] = greycoprops(glcm, 'correlation')[0, 0]
                        glcm_dis[x,y] = greycoprops(glcm, 'dissimilarity')[0, 0]

            cv2.imwrite(path_glcm + sheet.cell_value(i, 0) + '_glcm_cor.bmp', glcm_cor)
            cv2.imwrite(path_glcm + sheet.cell_value(i, 0) + '_glcm_dis.bmp', glcm_dis)

    #Converts skin lesion image to an LBP texture before saving the image in the PH2 dataset file
    def save_lbp():
       
        lbp = LocalBinaryPatterns(0, 0)
    
        #Image and mask samples
        for i in range(13, sheet.nrows):
    
            path_img = os.path.join(path_samples, sheet.cell_value(i, 0) + '\\' + sheet.cell_value(i, 0) +  '_Dermoscopic_Image\\' + sheet.cell_value(i, 0) + '.bmp')
            path_lbp = os.path.join(path_samples, sheet.cell_value(i, 0) + '\\' + sheet.cell_value(i, 0) +  '_lbp\\')
            
            if not os.path.isdir(path_lbp):
                os.makedirs(path_lbp)
    
            img = image.load_img(path_img)
            image_array = image.img_to_array(img, dtype="uint8")/255.0
    
            #Apply LBP filter
            lbp_image = lbp.describe(rgb2gray(image_array))
    
            cv2.imwrite(path_lbp + sheet.cell_value(i, 0) + '_lbp.bmp', lbp_image)

    def expert_masks():
        pass

