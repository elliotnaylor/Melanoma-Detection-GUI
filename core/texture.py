 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import math
import xlrd

class LocalBinaryPatterns:

    _powers = [1, 2, 4, 8, 16, 32, 64, 128]
    _positions = [[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0]]

    def __init__(self, numPoints, radius):
        #Not implemented just yet
        self.numPoints = numPoints 
        self.radius = radius 

    def getPixel(self, image, center, x, y):
        value = 0
        try:
            if image[x][y] >= center:
                value = 1
        except:
            pass
        return value

    def describe(self, image):
        print("LBP filter being applied")
        
        width = image.shape[0]
        height = image.shape[1]

        output = np.zeros((width,height), np.uint8)

        for i in range(0, width):
            for j in range(0, height):
                
                #Generate binary list
                samples = []
                for k in range(len(self._positions)):
                    pos = self._positions[k]
                    samples.append(self.getPixel(image, image[i][j], i+pos[0], j+pos[1]))

                #Convert binary list to decimal
                v = 0
                for l in range(len(samples)):
                    v += samples[l] * self._powers[l]
                
                #If v power is the power of 2 or 0
                if v != 0 and math.log(v, 2).is_integer():
                    output[i][j] = 0
                else:
                    output[i][j] = v

        return output