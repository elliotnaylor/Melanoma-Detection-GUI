
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
    csv_path = 'D:/Datasets/ISIC_2018/ISIC_2017_GroundTruth.csv'
    save_path = 'D:/Datasets/ISIC_2018/ISIC_2017_Complicated.csv'
    
    asymmetry = Asymmetry()
    border = Border()
    colour = Colour()

    structure = ['Diagnosis', 'Asymmetry', 'Globules', 'Milia', 'Negative', 'Pigment', 'Streaks']

    #Train bayesian network based on pre-compiled csv file
    #Model will be saved in the future
    def train_network(self):

        #Train bayesian network based on the provided metadata
        self.Bf = bayesianFusion(self.csv_path)

    def getSegmentation(image):
        mask = Segmentation.segNet(image)

        #Image might need converting?
        
        #Apply a mask using OpenCV
        element = apply_mask_cv(image, mask)
        masked = element

        return mask, masked

    #file_path is a csv with coloums containing filenames for testing
    def generateMetdata(self):
        
        dermo_list = ['globules','milia_like_cyst','negative_network','pigment_network','streaks']

        array = csv_to_array(self.csv_path)

        for i in range(1, len(array)):


            lesion_path = self.folder_path + 'Training_Skin_lesion/' + array[i][0] + '.jpg'
            img = cv2.imread(lesion_path)

            #Segment
            mask, masked = self.segmenation(img)

            #Asymmetry
            dataH, dataV, asymmetry = self.asymmetry(img, masked, mask)
            array[i][2] = asymmetry

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
                        array[i][j+3] = 1
                    else:
                        array[i][j+3] = 0
                else:
                    #Remove row
                    #array[i][j+3] = '-'
                    np.delete(array, i, 0) #Delete row where no data is found

        #Removing headers and filenames for bayesian network training
        np.delete(array, 0, 0) #Delete headers
        array[:,1:] #Remove column 0 containing all the names        

        array_to_csv(array, self.save_path)

    #"Variables" is currently from the metadata, but will be auto-generated in the future

    def predictImage(self, image, variables):
        
        self.generateMetdata()
        self.train_network()

        #Get segmentation mask of image using SegNet
        mask, masked = self.segmenation(image)

        #Run asymmetry
        dataH, dataV, asymmetry = self.asymmetry(image, masked, mask)

        variables.append(asymmetry)

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


