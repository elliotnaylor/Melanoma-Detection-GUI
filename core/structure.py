
import numpy as np
import cv2
import math

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from skimage import segmentation, color, feature
from skimage import measure
from skimage.segmentation import mark_boundaries

from scipy.spatial import ConvexHull

import mahotas as mh

#Dermoscopic structure
class Dermoscopic_Structure:

    def __init__():
        pass

    def pigmentNetwork():
        BANK = 10
        
        #Parameters
        size = 21
        sigma = 2
        phi = 0
        lam = 5
        gamma = 8
        theta = 1
        
        alpha = 0.5
        beta = 1.0 - alpha
        
        kernel = cv2.getGaborKernel((size, size), sigma, 0+math.pi/2, lam, gamma, theta+math.pi/2, cv2.CV_32F)
        combined = cv2.filter2D(img_gray, -1, kernel)
        
        for i in range(1, BANK):
            kernel = cv2.getGaborKernel((size, size), sigma, (i* theta)+math.pi/2, lam, gamma, phi+math.pi/2, cv2.CV_32F)
            #kernel /= math.sqrt((kernel * kernel).sum())
            
            filtered = cv2.filter2D(img_gray, -1, kernel)
        
            #for x in range(0, combined.shape[1]):
            #   for y in range(0, combined.shape[0]):
            
            combined += filtered
            
            #cv2.addWeighted(filtered, alpha, combined, beta, 0.0)
        
        thing, thresh = cv2.threshold(combined, 100, 255, cv2.THRESH_TOZERO)