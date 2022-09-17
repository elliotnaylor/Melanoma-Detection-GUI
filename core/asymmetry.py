import numpy as np
import cv2
import math
from skimage import segmentation, color, measure

from scipy.spatial import ConvexHull

from utils.cv import *
from utils.draw import *
from utils.math import *
from utils.numpy import *
from utils.plot import *

import mahotas

class Asymmetry:

    DISTANCE_THRESH = 40
    THRESH = 20
    score = 100 #Starting score

    #https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python

    def superpixel_centroids(self, image, mask, number_segments):
        locations = []

        #Find area of skin lesion, change 'n_segments' for less with larger skin lesion and more with smaller (equal segments)
        

        #Use SLIC to generate superpixels and comparecolour on either side of the lesion
        labels = segmentation.slic(image, compactness=20, n_segments=number_segments) #1500) #Should scale with the size of the lesion
        superpixels = color.label2rgb(labels, image, kind = 'avg', bg_label = 0) #Average colour of superPixels
        regions = measure.regionprops(labels)

        #Capture centroid of each superpixel, storing all points within the lesion area into a mask
        for props in regions:
            x, y = props.centroid

            x = int(round(x))
            y = int(round(y))

            #Check which centroids are wihtin the area of the mask
            if mask[x, y] > 0 and y < superpixels.shape[0] and x < superpixels.shape[1]: #regions returns some values outside of the size of the image
                locations.append((y, x))
        
        return superpixels, locations

    def split_positions(self, locations, center, axis):
        locations1 = []
        locations2 = []

        #Seperate positions of lesion into two sets of halves for colour symmetry
        for l in locations:
            if l[axis] < center[axis]:
                locations1.append(l)
            elif l[axis] > center[axis]:
                locations2.append(l)

        return locations1, locations2

    def moments(self, _image):
        moments = cv2.moments(_image)
        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])) #Gets center
        phi = 0.5*math.atan(2*moments["m11"]/(moments["m20"]-moments["m02"])) #Gets rotation

        return center, phi

    def difference(self, difference, thresh):
        over = 0
        for v in difference:
            if v >= thresh:
                over += 1

        asymmetry = 0
        if over > len(difference) / 2:
            asymmetry = 1

        return asymmetry

    def shapeSymmetry(self, image, center):
        #Crop image into four halves relating to vertical and horizontal lines
        h1 = image[0:center[1], 0:image.shape[1]]
        h2 = image[center[1]:image.shape[0], 0:image.shape[1]]
        v1 = image[0:image.shape[0], 0:center[0]]
        v2 = image[0:image.shape[0], center[0]:image.shape[1]]

        #Crop images for the sizes to match
        if h1.shape[0] > h2.shape[0]:
             h1 = h1[h1.shape[0] - h2.shape[0]:center[1], 0:h1.shape[1]]
        else:
             h2 = h2[0:h1.shape[0], 0:h1.shape[1]]
        
        if v1.shape[1] > v2.shape[1]:
             v1 = v1[0:v1.shape[0], v1.shape[1] - v2.shape[1]:center[0]]
        else:
             v2 = v2[0:v1.shape[0], 0:v1.shape[1]] 

        #Flip images
        h2 = cv2.flip(h2, 0)
        v2 = cv2.flip(v2, 1)

        #Subtract sides
        h_sub = cv2.add(h1-h2, h2-h1)
        v_sub = cv2.add(v1-v2, v2-v1)

        #draw_image(h2)
        #draw_image(v2)
        #draw_image(h_sub)
        #draw_image(v_sub)

        #Mirror results of images and combine (Only visual)
        #h_symm = Image.new('GREY', h_sub.shape[0], h_sub.shape[1]*2)
        #h_symm.paste(h_sub, (0,0))
        #h_symm.paste(cv2.flip(h_sub, 0), (0, h_sub.shape[1]))

        #Add translation method to move the center point
        area = np.sum(image == 255) / 255

        h_White = np.sum(h_sub == 255) / 255
        v_White = np.sum(v_sub == 255) / 255
        
        h_score = h_White / area
        v_score = v_White / area

        return h_score, v_score

    #Pair the closest positions in the array and compare their colour
    def colourSymmetry(self, image, center, locations_1, locations_2, axis):
        difference = []
        positions = []
        index = 0

        for i, h_1 in enumerate(locations_1):
            is_opposite = False
            smallest = 999

            #Distance between center and current position
            distance = center[axis] - h_1[axis]
            position = center[axis] + distance

            for j, h_2 in enumerate(locations_2):

                #Euclidean distance comparing new position and positions on other side of lesion
                if axis == 0:
                    euclidean = euclidean2d(position - h_2[0], h_1[1] - h_2[1])
                else:
                    euclidean = euclidean2d(h_1[0] - h_2[0], position - h_2[1])

                #Find the closest on the opposite side
                if euclidean <= self.DISTANCE_THRESH and euclidean < smallest:
                    is_opposite = True
                    smallest = euclidean
                    h = h_2
                    index = j

            if is_opposite:
                locations_2.pop(index) # Prevent an area from being chosen twice
                
                #Measuring the euclidean distance of the 2 colours to find which colours are a noticable difference
                #JND = square_root((L_b1 - L_b2)^2 + (A_b1 - A_b2)^2 + (B_b1 - B_b2)^2)
                dis = euclidean3d(image[h[1], h[0]], image[h_1[1], h_1[0]])

                # < 3 = not visible, 3<*< = 6 barely perceptible, > 6 perceptable (visible to the human eye)
                difference.append(dis)
                positions.append((h, h_1))

                #temp = image
                #cv2.circle(temp, h, 5,[0,0,255],-1)
                #cv2.circle(temp, h_1, 5,[0,255,255],-1)
                #draw_image(temp)


        return difference, positions

    def TextureSymmetry():
        pass
        #Bag of features
        #Bhattacharyya distance

    def run(self, _image, _masked_image, _mask):

        bgr = _masked_image
        gray = bgr2gray_cv(_masked_image)
        gray_mask = bgr2gray_cv(_mask)

        #Center of mass
        center, phi = self.moments(gray)

        image_rotation = rotate_bound(gray, math.degrees(-phi)) #cv2.warpAffine(gray, rotation_matrix, gray.shape[1::-1], flags=cv2.INTER_LINEAR)
        mask_rotation = rotate_bound(gray_mask,  math.degrees(-phi)) #cv2.warpAffine(gray_mask, rotation_matrix, gray_mask.shape[1::-1], flags=cv2.INTER_LINEAR)
        colour_rotation = rotate_bound(bgr, math.degrees(-phi))

        #Recalculate center of mass from the newly rotated image (rotate_bound function resizes image, making it difficult to translate position)
        center, phi = self.moments(mask_rotation)
        
        h_shape, v_shape = self.shapeSymmetry(mask_rotation, center)

        lab = bgr2lab_cv(colour_rotation)

        segment_number = 1500

        #Create superpixels and get the centroids of each within the area of the mask
        pixels, locations = self.superpixel_centroids(lab, mask_rotation, segment_number)

        #split the positions along the 
        h_loc1, h_loc2 = self.split_positions(locations, center, 0)
        v_loc1, v_loc2 = self.split_positions(locations, center, 1)

        h_colour, h_pos = self.colourSymmetry(pixels, center, h_loc1, h_loc2, 0) #Horizontal
        v_colour, v_pos = self.colourSymmetry(pixels, center, v_loc1, v_loc2, 1) #Vertical
        
        asymmetry = 0
        asymmetry += self.difference(h_colour, self.THRESH)
        asymmetry += self.difference(v_colour, self.THRESH)

        #draw_image(lab)
        #draw_comparison(h_colour, v_colour, self.THRESH)

        return h_colour, v_colour, asymmetry