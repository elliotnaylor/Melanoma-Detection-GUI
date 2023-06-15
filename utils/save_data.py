import numpy as np
import csv
import os
import cv2

from PIL import Image, ImageTk

import numpy as np
import cv2
from core.asymmetry import Asymmetry
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

import matplotlib.pyplot as plt

import os

def count_white_pixels(image):
    return np.sum(image == 255) / 255


path_folder = 'D:/Datasets/ISIC_2018/Resized/Training/'
path = 'D:/Datasets/ISIC_2018/ISIC_2017_GroundTruth.csv'
path_save = 'D:/Datasets/ISIC_2018/ISIC_2017_Complicated.csv'

dermo_list = ['globules','milia_like_cyst','negative_network','pigment_network','streaks']

def csv_to_array(path):
    return np.genfromtxt (path, delimiter=",", dtype=str)

def array_to_csv(array, path):
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(array)



def findDermoscopic():
    array = csv_to_array(path)
    for i in range(1, len(array)):
        seg_path = path_folder + 'Training_Segmentation/' + array[i][0] + '_segmentation.png'
        seg = cv2.imread(seg_path)

        seg_num = count_white_pixels(seg)

        for j in range(0, len(dermo_list)):

            dermo = cv2.imread(path_folder + 'Training_Dermoscopic/' + array[i][0] + '_attribute_' + dermo_list[j] + '.png')

            dermo_num = count_white_pixels(dermo)

            percentage = dermo_num / seg_num

            if dermo is not None:
                array[i][j+3] = round(percentage, 2)
            else:
                array[i][j+3] = '-'

    array_to_csv(array, path_save)
