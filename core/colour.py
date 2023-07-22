import numpy as np
import cv2
import math
import skimage

from utils.cv import *
from utils.draw import *
from utils.plot import *
from utils.math import *

from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans

class Colour:

    #Starting colour ranges for white, black, light brown, dark brown, red and blue gray in LAB
    #https://www.researchgate.net/publication/342223285_Towards_the_automatic_detection_of_skin_lesion_shape_asymmetry_color_variegation_and_diameter_in_dermoscopic_images#pfd

    image = []
    colours = []
    THRESH = 5

    #Starting colour ranges (LAB) in order of white, red, light_brown, dark brown, blue-gray and black (OPENCV STORES LAB DIFFERENTLY)
    #0 > L > 100 ⇒ OpenCV range = L*255/100 (1 > L > 255)
    #-127 > a > 127 ⇒ OpenCV range = a + 128 (1 > a > 255)
    #-127 > b > 127 ⇒ OpenCV range = b + 128 (1 > b > 255)
    colour_ranges = ((173.9865, 139.43, 149.62), (99.807, 140.9, 125.21), (153.816, 155.21, 172.48), (106.794, 139.44, 132.4), (95.9055, 132.88, 113.366), (77.2395, 136.06, 128.34))
    
    #NAMES = 'white', 'red', 'light_brown', 'dark_brown', 'blue_gray', 'black'

    #white, red, light_brown, dark_brown, blue_gray, black
    NAMES = 0, 1, 2, 3, 4, 5

    colour_min = [(255, 255, 255), (255, 0, 0), (141, 106, 81), (48, 33, 27), (0, 134, 139), (1, 0, 0)]

    colour_max = [(255, 255, 255), (255, 0, 0), (255, 139, 54), (166, 93, 31), (0, 134, 139), (144, 72, 60)]


    '''
    Original LAB colours

    black_min = (0.06, 0.27, 0.1)
    black_max = (39.91, 30.23, 22.1)
    dark_brown_min = (14.32, 6.85, 6.96)
    dark_brown_max = (47.57, 27.14, 46.81)
    light_brown_min = (47.94, 11.89, 19.86)
    light_brown_max = (71.65, 44.81, 64.78)
    white = (100, 0, 0)
    red = (54.29, 80.81, 69.89)
    blue_gray = (50.28, -30.14, -11.96)
    '''

    colour_names = ('#f6ceaf', '#db4545', '#b66b0f', '#5e3e25', '#5e6889', '#250000') #Colours in hex for pyplot

    def __init__(self):
        pass


    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        

    def histogram3d(self, image):
        pass


    def getColourRanges(self):
        locations = []
        colour_loc = []
        
        closest_colours = []

        X_colours = []
        Y_colours = []
        Z_colours = []

        number_colour = []
        colour_labels = []

        colour_dict = []

        #Gets all the possible locations within the colour ranges
        for i in range(len(self.colour_min)):
            
            colour_dict.append((i, self.colour_min[i])) 
            colour_dict.append((i, self.colour_max[i]))

            for l in range(self.colour_min[i][0], self.colour_max[i][0], 10):
                for a in range(self.colour_min[i][1], self.colour_max[i][1], 10):
                    for b in range(self.colour_min[i][2], self.colour_max[i][2], 10):
                        colour_dict.append((i, (l, a, b)))  

        #lab = cv2.cvtColor(self.image, cv2.COLOR_RGB2LAB)

        #Store locations of the lesion that are not masked
        for x in range(0, self.image.shape[0], 5):
            for y in range(0, self.image.shape[1], 5):

                #Remove superpixels that are black
                value = self.image[x, y][0] + self.image[x, y][1] + self.image[x, y][2]

                if value > 10 and x < self.image.shape[0] and y < self.image.shape[1]: #regions returns some values outside of the size of the image
                    locations.append((x, y))
        
        #Kmeans to find the seven most dominant colours
        #k-means to locate the colour centroid of the colour values
        km = KMeans(6) #All six colours counting black (mask)
        km.fit(locations)
        labels = km.fit_predict(locations)
        centroids = km.cluster_centers_

        #1. Find the location of the closest colour to the 7 lesion colours for k-means starting locations
        #Removing colour areas outside the threshold (anomalous data)
        for l in centroids:
            score = 999
            col = -1
            
            for c in range(0, len(colour_dict)):
                dis = euclidean3d(self.image[l[0], l[1]], colour_dict[c][1])

                #If below threshold store value in array
                if(dis < score and dis < self.THRESH):
                    score = dis
                    col = colour_dict[c][0]

            '''
            #Find closest colour location
            for c in range(0, self.colours):
                
                dis = euclidean3d(lab[l[0], l[1]], self.colour_ranges[c])
                
                #If below threshold store value in array
                if(dis < score and dis < thresh):
                    score = dis
                    col = c
            '''
            
            #2. Store areas asccociated with the closest colour (stored as -1 if no colour is in range)
            if col > -1:
                closest_colours.append((self.image[l][0], self.image[l][1], self.image[l][2]))
                number_colour.append(col)

        closest_colours = np.array(closest_colours)
        number_colour = np.array(number_colour)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for i in range(0, len(number_colour)):
            ax.scatter(closest_colours[i][0], closest_colours[i][1], closest_colours[i][2], c=self.colour_names[number_colour[i]], marker='o')
        #plt.show()

        return closest_colours, number_colour


    def run(self, _images, _masked_images, _masks, _colours = 6):

        #draw_image(_masked_images[i])
    
        self.colours = _colours

        self.image = cv2.GaussianBlur(_masked_images, (11, 11), 0)
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
    
        pos, col = self.getColourRanges() #Pos is array of [(x, y)] [(l,a,b)]
    
        


        '''
        #Removes locations of pixels that aren't associated with one of the 6 colours
        for i in range(0, len(col)):
            locations.append((pos[i][0], pos[i][1], pos[i][2]))
            colours.append(col[i])
        '''

        return pos, col


    '''
    Colour pallete is already pre-defined in 'colour_ranges', but the 
    below code can be used to generate one using the Ph2 dataset
    '''
    def generateArea(self):
        
        #Find colour pallete of an image 
        for i in range(0, 200):
            #k-means to locate the colour centroid of the colour values
            km = KMeans(7) #All six colours counting black (mask)
            km.fit(locations)
            labels = km.fit_predict(locations)
            centroids = km.cluster_centers_

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            locations = np.array(locations)

            ax.scatter(locations[:,0], locations[:,1], locations[:,2], c=labels, marker='o')
            plt.show()

            colour = []
            t = 10

            #Compare euclidean distance of centroids from k-means to find which colour is the closest
            #Note: Positions should be a colour range instead, but just like this for testing
            for cen in centroids:
                score = 999
                col = -1

                for l in range(0, len(locations)):

                    dis = euclidean3d(locations[l], cen[0])
                
                #If below threshold store value in array
                    if(dis < score and dis < t):
                        score = dis
                        col = colours[i]

                if col > -1:
                    colour.append(col)
