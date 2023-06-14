
from tkinter import *
import tkinter.messagebox
from tkinter import filedialog
import customtkinter
from PIL import Image, ImageTk

import numpy as np
import cv2
from core.asymmetry import Asymmetry
from utils.cv import *
from utils.plot import draw_svm_boundries
from utils.xlrd import *

from load_PH2 import PH2

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
import pickle
from joblib import dump, load

import matplotlib.pyplot as plt

import os

class ABCD_Rules:

    def segmentation():


    def asymmetry():
        a = Asymmetry

    def asymmetry():
        a = Asymmetry