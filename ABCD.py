
from tkinter import *
import tkinter.messagebox
from tkinter import filedialog
import customtkinter
from PIL import Image, ImageTk

import numpy as np
import cv2
from core.asymmetry import Asymmetry
from core.border import Border
from core.colour import Colour
from core.segmentation import *
from core.bayesian import *
from utils.draw import *
from utils.plot import *


from utils.cv import *
from utils.plot import draw_svm_boundries
from utils.xlrd import *

from dataset.load_PH2 import PH2

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
import pickle
from joblib import dump, load

import matplotlib.pyplot as plt

import os
import csv


def count_white_pixels(image):
    return np.sum(image > 10) / 255

def csv_to_array(path):
    return np.genfromtxt (path, delimiter=",", dtype=str)

def array_to_csv(array, path):
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(array)

def white_pixels(img):
    pixel_count = count_white_pixels(img)
    if pixel_count > 0:
        return 1
    else:
        return 0


class ABCD_Rules:

    #Path to dataset on main PC
    #folder_path = 'D:\\Datasets\\ISIC_2018\\Resized\\Training\\'
    
    #path to dataset on laptop
    folder_path = 'C:\\Users\\el295904\\Dataset\\ISIC_2018\\Resized\\'
    
    csv_path = os.path.join(os.getcwd(), 'data\\ISIC_2017_GroundTruth.csv')
    save_path = os.path.join(os.getcwd(), 'data\\ISIC_2017_generated.csv')
    model_path = os.path.join(os.getcwd(), 'models\\')

    segmentation = Segmentation()
    asymmetry = Asymmetry()
    #border = Border()
    colour = Colour()

    structure = ['Diagnosis', 'Asymmetry', 'Globules', 'Milia', 'Negative', 'Pigment', 'Streaks']

    '''
    Train bayesian network based on pre-compiled csv file
    Model will be saved in the future
    '''
    def train_network(self):
        
        '''
        Uncaption when training new Bayesian network, needs 'csv_path'
        file with file names in first column, see file.
        '''
        #self.generateMetdata()

        #Train bayesian network based on the provided metadata
        self.Bf = bayesianFusion(self.save_path)


    #file_path is a csv with columns containing filenames for testing
    def generateMetdata(self):

        array = csv_to_array(self.csv_path)
        img_array = []

        #Load all images from csv file into array that can be used with segnet
        for i in range(1, len(array)):
            
            lesion_path = self.folder_path + 'removed/' + array[i][0] + '.jpg'

            img = Image.open(lesion_path)
            img = np.array(img)
            #img = img[np.newaxis, ...] #Prediction requires 

            #Segment
            img_array.append(img)
        
        img_array = np.array(img_array)
        
        masks, masked = self.getSegmentation(img_array, 'skin_lesion.h5')


        for i in range(0, len(masks)):
            
            #draw_image(masked[i])

            #Gets asymmetry of image, needs img, mask, and masked images
            dataH, dataV, locH, locV, asymmetry = self.asymmetry.run(img_array[i], masked[i], masks[i])

            array[i+1][2] = asymmetry
            #print("Finished image: " + i + " out of " + len(masks))

            #White, red, light_brown, dark brown, blue-gray and black (0 not present, 1 present)
            colours = [0, 0, 0, 0, 0, 0]
            number_colours = self.colour.run(img_array[i], masked[i], masks[i])

            number = 0

            #Label that the colour exists in order of white, red, light_brown, dark brown, blue-gray and black
            for j in range(0, len(number_colours)):
                colours[number_colours[j]] = 1
                if number_colours[j] > 0:
                    number += 1

            array[i+1][3] = number

            '''
            for j in range(0, len(colours)):
                array[i+1][j+3] = colours[j]
            '''
            

            
            '''
            CSV is already populated with dermoscopic structures.
            Will be replaced with automatic detection in the future.
            '''
            
            save = array
            #Removing headers and filenames for bayesian network training
            np.delete(save, 0, 0) #Delete headers
            save[:,1:] #Remove column 0 containing all the names        

            array_to_csv(save, self.save_path)


    '''
    Using path to an image of size 192 x 256 by passing a (1, 192, 256, 3)
    into SegNet. Passes out a (1, 192, 256), and is transformed back into
    a (192, 256, 3). Also generates a mask.
    '''
    def getSegmentation(self, images, filename):
        mask_array = []
        masked_array = []
        
        #Initalize SegNet
        masks = self.segmentation.segNet(images, self.model_path + filename)
        
        threshmask = []

        #Convert float32 to uint8 and convert single pixel value to tuple
        masks *= 255
        masks = masks.astype(np.uint8)

        #Threshold to make images either black or white and no inbetween
        for m in masks:
            ret, threshold = cv2.threshold(m, 15, 255, cv2.THRESH_BINARY)
            threshmask.append(threshold)

        threshmask = np.array(threshmask)
        

        for i in range(0, len(threshmask)):
            
            #Convert into an RGB mask
            rgb_mask = cv2.cvtColor(threshmask[i], cv2.COLOR_GRAY2BGR)
            
            mask_array.append(rgb_mask)

            masked = apply_mask_cv(images[i], rgb_mask)

            #draw_image(masked)

            #Apply a mask using OpenCV
            masked_array.append(masked)
            
            
        return mask_array, masked_array

    '''
    Gets metadata on the loaded image using the interface.
    Populates the interface (interface.py) to allow for
    changes before predictImage() is called.
    '''
    def analyseImage(self, file_path):

        variables = [] #Order of asymmetry, globules, milia, negative, network, and streaks
        
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        #Load image using Pillow, loading using OpenCV produces BGR values
        img = Image.open(file_path)
        img = np.array(img)
        img = img[np.newaxis, ...] #Prediction requires an array of image

        #Image loaded as BGR, changes it to RGB
        #img = img[:,:,::-1]

        #plt.imshow(img)
        #plt.show()

        #Segment
        masks, masked = self.getSegmentation(img, 'skin_lesion.h5')

        #Asymmetry
        dataH, dataV, asymmetry = self.asymmetry.run(img[0], masked[0], masks[0])

        
        variables.append(asymmetry)

        #Border
        #border = self.border.run(masks[0])

        #Colour
        number_colours = self.colour.run(img[0], masked[0], masks[0])

        #White, red, light_brown, dark brown, blue-gray and black (0 not present, 1 present)
        colours = [0, 0, 0, 0, 0, 0]
        
        #Label that the colour exists in order of white, red, light_brown, dark brown, blue-gray and black
        for i in range(0, len(number_colours)):
            colours[number_colours[i]] = 1

        for c in colours:
            variables.append(c)

        #Numpy function to pass CSV file into an array
        array = csv_to_array(self.csv_path)

        #Dermoscopic structure
        dermo_list = ['globules','milia_like_cyst','negative_network', 'pigment_network','streaks']

        '''
        Get dermoscopic structures, currently on whether there 
        is an image available in the ISIC 2017 dataset.
        Will be replaced with automatic detection in the future.
        '''
        for i in range(1, len(array)):

            if(array[i][0] == file_name):
                
                for j in range(0, len(dermo_list)):

                    dermo = cv2.imread(self.folder_path + 'Training_Dermoscopic\\' + array[i][0] + '_attribute_' + dermo_list[j] + '.png')
                    
                    if dermo is not None:

                        variables.append(white_pixels(dermo))

                    else:
                        print("Error: No image data found on " + file_name)

        #Pigment Network
        p_masks, p_masked = self.getSegmentation(img, 'network.h5')

        variables[4] = white_pixels(p_masks[0])

        return masks[0], p_masks[0], variables

    '''
    Variables are values in the interface, that are saved and used for
    predition. Call analyseImage() first before predictImage().
    '''
    def predictImage(self, variables):
        
        NAMES = 'Asymmetry', 'Globules', 'Milia', 'Negative', 'Pigment', 'Streaks'

        #evidence={'Asymmetry': _a, 'Globules':_g, 'Milia':_m, 'Negative':_n, 'Pigment':_p, 'Streaks':_s}

        evidence = {}

        for i in range(0, len(variables)):
            if variables[i] != -1: #-1 means "Don't know" and to exclude it
                evidence[NAMES[i]] = variables[i] #Dictionary of variable

        #Predict based on the variables provided        
        weights = self.Bf.predict(evidence)
        
        #Remove each and re-predict to find out which features matter most
        

        return weights


