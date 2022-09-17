import numpy as np
import cv2
import math
import skimage

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.spatial import ConvexHull

import mahotas as mh

class Border:

    radius = 21

    def __init__():
        pass

    def number_boxes(self, border, size):

        number = np.add.reduceat(
            np.add.reduceat(border, np.arange(0, border.shape[0], size), axis=0),
                                    np.arange(0, border.shape[1], size), axis=1)

        return len(np.where((number > 0) & number < size*size))[0]

    def fractual_box(self, border, threshold=60):
        
        #Create binary image
        border = (border < threshold)

        #Minimal dimension
        minimum = min(border.shape)

        #Exponent
        n = 2**np.floor(np.log(minimum)/np.log(2))
        n = int(np.log(n)/np.log(2))

        sizes = 2**np.arange(n, 1, -1)

        #Find amount of generated boxes for caluclating the score
        count = []
        for size in sizes:
            counts.append(self.number_boxes(border, size))

        #Fit log sizes with log counts to calculate the total score
        #Higher the value means more fractal in nature (tight corners)
        coeff = np.polyfit(np.log(sizes), np.log(counts), 1)

        return -coeff[0]
        
    def zernike_moments():
        #https://peerj.com/articles/cs-268/

        #1. Canny segmentation of mask for caluckating fractal score
        edges = feature.canny(mask, sigma=3)

        #2. Calculate box-counting fractal dimension (Requires gray scale)
        fractal_score = self.fractual_box(edges)

        #3. Generate zernick moments based on segmentation area
        z_moments = mh.features.zernike_moments(mask, self.radius)

        #4. caluclate convexity using convex hull and controus
        #Find contour of mask
        contours, hierarchy = cv2.findContours(mask, 2, 1)
        count = contours[0]

        convex_hull = cv2.convexHull(count, returnPoints = False)
        defects = cv2.convexityDefects(count, convex_hull)

        convexity = 0
        convex_locations = []

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i,0]
            far = tuple(count[f][0])
            convex_locations.append(far)
            #cv2.circle(mask,far,5,[0,0,255],-1)
           
        if(len(convex_locations) > 0):
            convexity = 1

        #5. Combine features into a array of 27 vector (1 = fractal, 1 = convexity, 25 = moments)
        vectors = []

        vectors.append(fractal_score)
        vectors.append(convexity)

        for m in z_moments:
            vectors.append(m)

        #6. CNN consisting of two layers

    def run(self, mask):
        mask = cv2.cvtcolor(mask, cv2.COLOR_BGR2GRAY)