
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

from load_PH2 import PH2

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
    return np.sum(image == 255) / 255

def csv_to_array(path):
    return np.genfromtxt (path, delimiter=",", dtype=str)

def array_to_csv(array, path):
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(array)


class ABCD_Rules:

    folder_path = 'D:/Datasets/ISIC_2018/Resized/Training/'
    csv_path = 'C:/Users/scary/Documents/GitHub/Melanoma-detection-GUI/data/ISIC_2017_GroundTruth_Complete.csv'
    save_path = 'D:/Datasets/ISIC_2018/ISIC_2017_generated.csv'
    
    asymmetry = Asymmetry()
    #border = Border()
    #colour = Colour()

    structure = ['Diagnosis', 'Asymmetry', 'Globules', 'Milia', 'Negative', 'Pigment', 'Streaks']

    #Train bayesian network based on pre-compiled csv file
    #Model will be saved in the future
    def train_network(self):
        
        #self.generateMetdata()

        #Train bayesian network based on the provided metadata
        self.Bf = bayesianFusion(self.save_path)

    def getSegmentation(self, images):
        mask_array = []
        masked_array = []

        masks = Segmentation.segNet(images)
        
        #Convert float32 to uint8 and convert single pixel value to tuple
        masks = masks*255
        masks = masks.astype(np.uint8)

        for i in range(0, len(masks)):

            rgb_mask = cv2.cvtColor(masks[i], cv2.COLOR_GRAY2BGR)
            
            mask_array.append(rgb_mask)

            #Apply a mask using OpenCV
            masked_array.append(apply_mask_cv(images[i], rgb_mask))

        return mask_array, masked_array

    #Used to get metadata on the loaded image using the interface
    #User can then modfiy values before predictImage() is called
    def analyseImage(self, file_path):

        variables = []

        dermo_list = ['globules','milia_like_cyst','negative_network','pigment_network','streaks']

        array = csv_to_array(self.csv_path)

        file_name = os.path.splitext(os.path.basename(file_path))[0]

        for i in range(1, len(array)):

            if(array[i][0] == file_name):
                
                img = cv2.imread(file_path)

                img = img[np.newaxis, ...] #Prediction requires an array of images

                #Segment
                masks, masked = self.getSegmentation(img)

                #Asymmetry
                dataH, dataV, asymmetry = self.asymmetry.run(img[0], masked[0], masks[0])

                variables.append(asymmetry)

                '''
                Get dermoscopic structures, currently on whether there is an
                image available in the ISIC 2017 dataset.
                Will be replaced with automatic detection in the future
                '''
                for j in range(0, len(dermo_list)):

                    dermo = cv2.imread(self.folder_path + 'Training_Dermoscopic/' + array[i][0] + '_attribute_' + dermo_list[j] + '.png')

                    dermo_num = count_white_pixels(dermo)

                    if dermo is not None:
                        if dermo_num > 0: #If there is is a mask of the dermoscopic feature
                            variables.append(1)
                        else:
                            variables.append(0)
                    else:
                        print("Error: No image data found on " + file_name)

        return variables



    #file_path is a csv with coloums containing filenames for testing
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
            #Asymmetry
            dataH, dataV, asymmetry = self.asymmetry.run(img_array[i], masked[i], masks[i])

            array[i+1][2] = asymmetry
            print("Finished image: " + i + " out of " + len(masks))

        '''
        CSV is already populated with dermoscopic structures
        Will be replaced with automatic detection in the future
        '''

        #Removing headers and filenames for bayesian network training
        #np.delete(array, 0, 0) #Delete headers
        #array[:,1:] #Remove column 0 containing all the names        

        array_to_csv(array, self.save_path)

    #"Variables" is currently from the metadata, but will be auto-generated in the future

    def predictImage(self, variables):

        '''
        self.train_network()
        
        #Get segmentation mask of image using SegNet
        mask, masked = self.getSegmentation(image)

        #Run asymmetry
        dataH, dataV, asymmetry = self.asymmetry.run(image, masked, mask)

        variables.append(asymmetry)
        '''
        

        #Train Bayesian Network
        #self.trainNetwork()

        weights = self.Bf.predict(
            variables[0], #Asymmetry
            variables[1], #Globules
            variables[2], #Milia
            variables[3], #Negative 
            variables[4], #Network
            variables[5]) #Streaked
        
        
        return weights


