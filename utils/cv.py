from utils.constants import * 
import numpy as np
import cv2

def load_img_cv(path):
    bgr = cv2.imread(path)
    resized = cv2.resize(bgr, IMG_SIZE_CV)

    return resized

def load_gray_cv(path):
    bgr = cv2.imread(path)
    resized = cv2.resize(bgr, IMG_SIZE_CV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    return gray

def apply_mask_cv(image, mask):
    masked = cv2.bitwise_and(image, mask)

    return masked

def bgr2lab_cv(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    return lab

def bgr2gray_cv(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray

def crop_mask(mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    coordinates = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coordinates)

    return x, y, w, h