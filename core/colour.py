import numpy as np
import cv2
import math
import skimage

from utils.cv import *
from utils.draw import *
from utils.plot import *
from utils.math import *

from scipy.spatial import ConvexHull
from scipy.spatial import distance
from sklearn.cluster import KMeans
from skimage import io, color

class Colour:

    #Starting colour ranges for white, black, light brown, dark brown, red and blue gray in LAB
    #https://www.researchgate.net/publication/342223285_Towards_the_automatic_detection_of_skin_lesion_shape_asymmetry_color_variegation_and_diameter_in_dermoscopic_images#pfd

    colours = []
    THRESH = 50

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
    thresh_names = 0, 1, 4
    thresh_colours = [(255, 255, 255), (255, 0, 0), (0, 134, 139)]

    range_names = 2, 3, 5
    range_colours_min = [(141, 106, 81), (49, 33, 27), (1, 0, 0)]
    range_colours_max = [(255, 139, 54), (166, 93, 31), (144, 72, 60)]
    '''

    #Red was originally 54.29, 80.81, 69.89
    #New red is 45.71, 67.67, 50.18

    thresh_names = 0, 1, 4
    thresh_colours = [(100, 0, 0), (54.29, 80.81, 69.89), (50.28, -30.14, -11.96)]

    range_names = 2, 3, 5
    range_colours_min = [(47.94, 11.89, 19.86), (14.32, 6.85, 6.96), (0.06, 0.27, 0.10)]
    range_colours_max = [(71.65, 44.81, 64.78), (47.57, 27.14, 46.81), (39.91, 30.23, 22.10)]

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


    def getColourRanges(self, lab):
        locations = []
        colours = []
        
        closest_colours = []

        X_colours = []
        Y_colours = []
        Z_colours = []

        number_colour = []
        colour_labels = []

        colour_dict = []

        #Gets all the possible locations within the colour ranges
        '''
        for i in range(len(self.colour_min)):
            
            colour_dict.append((i, self.colour_min[i])) 
            colour_dict.append((i, self.colour_max[i]))

            for l in range(self.colour_min[i][0], self.colour_max[i][0], 10):
                for a in range(self.colour_min[i][1], self.colour_max[i][1], 10):
                    for b in range(self.colour_min[i][2], self.colour_max[i][2], 10):
                        colour_dict.append((i, (l, a, b)))  
        '''
        

        #lab = cv2.cvtColor(self.image, cv2.COLOR_RGB2LAB)

        '''
        #Store skin lesion areas that are not black
        for x in range(0, self.image.shape[0]):
            for y in range(0, self.image.shape[1]):

                #Remove superpixels that are black
                #value = self.image[x, y][0] + self.image[x, y][1] + self.image[x, y][2]


                if x < self.image.shape[0] and y < self.image.shape[1]: #regions returns some values outside of the size of the image
                    locations.append((x, y))
                    colours.append((self.image[x, y][0], self.image[x, y][1], self.image[x, y][2]))
        '''
        
        #Flatten image for k-means
        img = lab.reshape((lab.shape[0] * lab.shape[1], lab.shape[2]))

        #1. Find the location of the closest colour to the 7 lesion colours for k-means starting locations
        #Removing colour areas outside the threshold (anomalous data)
        
        km = KMeans(7) #All six colours counting black (mask)
        km.fit(img)
        labels = km.fit_predict(img)
        avg_colours = km.cluster_centers_

        label_index = np.arange(0,len(np.unique(labels)) + 1)

        (hist, _) = np.histogram(labels, bins = label_index)

        hist = hist.astype("float")
        hist /= hist.sum()
        hist

        hist_bar = np.zeros((50, 300, 3), dtype = "uint8")

        rgb = color.lab2rgb(lab)
        rgb_colours = color.lab2rgb(avg_colours)

        rgb_colours *= 255

        startX = 0
        for (percent, colour) in zip(hist,  rgb_colours): 
            endX = startX + (percent * 300) # to match grid
            cv2.rectangle(hist_bar, (int(startX), 0), (int(endX), 50),
            colour.astype("uint8").tolist(), -1)
            startX = endX


        #plt.figure(figsize=(15,15))
        #plt.subplot(121)
        #plt.imshow(rgb)
        #plt.subplot(122)
        #plt.imshow(hist_bar)
        #plt.show()

        new_avg = []

        #Remove the mask calculated from the
        for i in range(0, len(avg_colours)):

            value = avg_colours[i][0] + avg_colours[i][1] + avg_colours[i][2]

            if value >= 5:
                new_avg.append(avg_colours[i])


        for k in new_avg:
            score = 999
            col = -1
            
            for c in range(0, len(self.thresh_names)):
                dis = distance.minkowski(k, self.thresh_colours[c], 3)
                #dis = euclidean3d(k, colour_dict[c][1]) #self.image[l[0], l[1]], colour_dict[c][1])

                #If below threshold store value in array
                if(dis < score and dis < self.THRESH):
                    score = dis
                    col = self.thresh_names[c]

            #2. Store areas asccociated with the closest colour (stored as -1 if no colour is in range)
            if col > -1:
                #closest_colours.append((self.image[l][0], self.image[l][1], self.image[l][2]))
                number_colour.append(col)


            for c in range(0, len(self.range_names)):

                #If below threshold store value in array
                if(all(k > self.range_colours_min[c]) and all(k < self.range_colours_max[c])):

                    number_colour.append(self.range_names[c])

        #closest_colours = np.array(closest_colours)
        number_colour = np.array(number_colour)
    
        
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for i in range(0, len(number_colour)):
            ax.scatter(closest_colours[i][0], closest_colours[i][1], closest_colours[i][2], c=self.colour_names[number_colour[i]], marker='o')
        #plt.show()
        '''
        
        return number_colour


    def run(self, _image, _masked_image, _mask, _colours = 6):
    
        #Image loaded as BGR, changes it to RGB
        #img = img.astype("float32") / 255
         
        lab = color.rgb2lab(_masked_image)

        #draw_image(_masked_image)
        
        self.colours = _colours

        col = self.getColourRanges(lab) #Pos is array of [(x, y)] [(l,a,b)]
    
        return col