from load_PH2 import PH2
import cv2

from utils.cv import *

#Read PH2 data set and assign a score for asymmetry and colour, display in graph
ph2 = PH2()

images = ph2.load_images()
masks = ph2.load_masks()


def get_images_comparison(_lesion_Index):

    masked = []
    for i in range(0, len(images)):
    
        element = apply_mask_cv(images[i], masks[i])
        masked.append(element)

    diagnoses_lesion = ph2.load_diagnosis()
    diagnoses_symmetry = ph2.load_asymmetry()
    diagnoses_colours = ph2.load_colours()

    similarity_index = []
    similarity_image = []

    for i in range (0, len(diagnoses_symmetry)):
        if(i != _lesion_Index and diagnoses_symmetry[6] == diagnoses_symmetry[i]):
            similarity_index.append(i) #Saves index of images
            similarity_image.append(images[i])
            
    

    return similarity_image 