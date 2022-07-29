from utils.constants import * 
from keras.preprocessing import image

def load_img_k(path):
    return image.load_img(path, target_size=IMG_SIZE)

def to_array_k(image):
    return image.img_to_array(image, dtype="uint8")/255.0