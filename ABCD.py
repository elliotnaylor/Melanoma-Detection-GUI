
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
    folder_path = 'D:\\Datasets\\ISIC_2018\\Resized\\Training\\'
    
    #path to dataset on laptop
    #folder_path = 'C:\\Users\\el295904\\Dataset\\ISIC_2018\\Resized\\'
    
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

            img = cv2.imread(lesion_path)

            #Segment
            img_array.append(img)
        
        img_array = np.array(img_array)
        
        masks, masked = self.getSegmentation(img_array)
        
        for i in range(0, len(masks)):
            #Gets asymmetry of image, needs img, mask, and masked images
            dataH, dataV, asymmetry = self.asymmetry.run(img_array[i], masked[i], masks[i])

            array[i+1][2] = asymmetry
            print("Finished image: " + i + " out of " + len(masks))


        '''
        CSV is already populated with dermoscopic structures.
        Will be replaced with automatic detection in the future.
        '''

        #Removing headers and filenames for bayesian network training
        np.delete(array, 0, 0) #Delete headers
        array[:,1:] #Remove column 0 containing all the names        

        array_to_csv(array, self.save_path)


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
        
        #Convert float32 to uint8 and convert single pixel value to tuple
        masks *= 255
        masks = masks.astype(np.uint8)

        for i in range(0, len(masks)):

            #Convert into an RGB mask
            rgb_mask = cv2.cvtColor(masks[i], cv2.COLOR_GRAY2BGR)
            
            mask_array.append(rgb_mask)

            #Apply a mask using OpenCV
            masked_array.append(apply_mask_cv(images[i], rgb_mask))
            
        return mask_array, masked_array
    

    '''
    Gets metadata on the loaded image using the interface.
    Populates the interface (interface.py) to allow for
    changes before predictImage() is called.
    '''
    def analyseImage(self, file_path):

        variables = [] #Order of asymmetry, globules, milia, negative, network, and streaks
        
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        #Load image using Pillow, loading using OpenCV produces different RGB values
        img = Image.open(file_path)
        img = np.array(img)
        img = img[np.newaxis, ...] #Prediction requires an array of image

        #Segment
        masks, masked = self.getSegmentation(img, 'skin_lesion.h5')

        #Asymmetry
        dataH, dataV, asymmetry = self.asymmetry.run(img[0], masked[0], masks[0])

        variables.append(asymmetry)

        #Border

        #Colour
        position, number_colours = self.colour.run(img[0], masked[0], masks[0])

        #Dermoscopic structure
        dermo_list = ['globules','milia_like_cyst','negative_network', 'pigment_network','streaks']

        #Numpy function to pass CSV file into an array
        array = csv_to_array(self.csv_path)

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
        
        weights = self.Bf.predict(
            variables[0], #Asymmetry
            variables[1], #Globules
            variables[2], #Milia
            variables[3], #Negative 
            variables[4], #Network
            variables[5]) #Streaked
        
        
        return weights


