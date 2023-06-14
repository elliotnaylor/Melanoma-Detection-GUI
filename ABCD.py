
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

class ABCD_Rules():

    metadata_path = 'D:/Datasets/ISIC_2018/ISIC_2017_GroundTruth_complete5.csv'
    
    asymmetry = Asymmetry()
    border = Border()
    colour = Colour()

    def trainNetwork(self):
        #Train bayesian network based on the provided metadata
        self.Bf = bayesianFusion(self.metadata_path)

    def segmentation(image):
        mask = Segmentation.segNet(image)

        #Image might need converting?
        
        #Apply a mask using OpenCV
        element = apply_mask_cv(image, mask)
        masked = element

        return mask, masked

    def generateMetdata():
        pass

    #"Variables" is currently from the metadata, but will be auto-generated in the future
    def predictImage(self, image, variables):

        #Get segmentation mask of image using SegNet
        mask, masked = self.segmenation(image)

        #Run asymmetry
        dataH, dataV, asymmetry = self.asymmetry(image, masked, mask)

        variables.append(asymmetry)

        #Train Bayesian Network
        self.trainNetwork()

        weights = self.Bf.predict(
            variables[0], #Globules
            variables[1], #Milia
            variables[2], #Negative
            variables[3], #Network
            variables[4], #Streaked
            variables[5]) #Dermoscopic Structures (amount)

        return weights



