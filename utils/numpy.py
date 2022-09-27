from utils.constants import * 
import numpy as np

def rgb2gray(image):
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

def apply_mask_np(image, mask):
    np.multiply(image[i], mask[i])