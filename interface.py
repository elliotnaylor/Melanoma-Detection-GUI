import os

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image 

from core.bayesian import *
from utils.plot import *

import csv

from ABCD import ABCD_Rules

import matplotlib.pyplot as plt
from io import BytesIO

def csv_to_array(path):
    return np.genfromtxt (path, delimiter=",", dtype=str)

class MainApplication(tk.Tk, ABCD_Rules) :

    tk_image = []
    loaded_image = []
    loaded_mask = []

    IMG_SHAPE = (700, 700)
    
    def prepareImage(self, img):
        img.thumbnail(self.IMG_SHAPE)
        return ImageTk.PhotoImage(img)
    
    def plt2tk(self):
        buffer = BytesIO()
        plt.savefig(buffer,format='png')
        image = Image.open(buffer)

        return image

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.train_network()

        # Adding a title to the window
        self.wm_title("Melanoma Detector")

        # creating a frame and assigning it to container
        container = tk.Frame(self, height=400, width=600)

        # specifying the region where the frame is packed in root
        container.pack(side="top", fill="both", expand=True)

        DROP_DOWN_OPTIONS = [-1, 0, 1]
        DROP_DOWN_OPTIONS_WORDS = ["Don't know", "No", "Yes"]

        MASK_DROP_DOWN = ['Segmentation', 'Asymmetry', 'Pigment']

        # configuring the location of the container using grid
        container.grid_rowconfigure(2, weight=1)
        container.grid_columnconfigure(2, weight=1)
        
        info_frame =tk.LabelFrame(container, text="Information")
        info_frame.grid(row=0,rowspan=3, column=0, padx=20, pady=10)
        
        # Saving User Info
        abcd_info_frame =tk.LabelFrame(info_frame, text="ABCD Rules", highlightthickness=0, borderwidth = 0)
        abcd_info_frame.grid(row= 0, column=0, padx=20, pady=10, sticky="W")

        asymmetry_name_label = tk.Label(abcd_info_frame, text="Asymmetry")
        asymmetry_name_label.grid(row=0, column=0, padx=10, pady=5)
        self.asymmetry_name_entry = tk.Entry(abcd_info_frame)
        self.asymmetry_name_entry.grid(row=0, column=1, padx=10, pady=5)

        border_name_label = tk.Label(abcd_info_frame, text="Border")
        border_name_label.grid(row=1, column=0, padx=10, pady=5)
        border_name_entry = tk.Entry(abcd_info_frame)
        border_name_entry.grid(row=1, column=1, padx=10, pady=5)

        colour_name_label = tk.Label(abcd_info_frame, text="Colour")
        colour_name_label.grid(row=2, column=0, padx=10, pady=5)
        colour_name_entry = tk.Entry(abcd_info_frame)
        colour_name_entry.grid(row=2, column=1, padx=10, pady=5)

        dermo_name_label = tk.Label(abcd_info_frame, text="Dermoscopic")
        dermo_name_label.grid(row=3, column=0, padx=10, pady=5)
        self.dermo_name_entry = tk.Entry(abcd_info_frame, state="disabled")
        self.dermo_name_entry.grid(row=3, column=1, padx=10, pady=5, sticky="W")

        #Section on dermoscopic structures
        structures_info_frame =tk.LabelFrame(info_frame, text="Dermoscopic Structures", highlightthickness=0, borderwidth = 0)
        structures_info_frame.grid(row= 1, column=0, padx=20, pady=10)

        network_name_label = tk.Label(structures_info_frame, text="Pigment Network")
        network_name_label.grid(row=0, column=0, padx=10, pady=5)
        self.network_combo = ttk.Combobox(structures_info_frame, values=DROP_DOWN_OPTIONS)
        self.network_combo.grid(row=0, column=1, padx=10, pady=5)
        
        negative_network_name_label = tk.Label(structures_info_frame, text="Negative Network")
        negative_network_name_label.grid(row=1, column=0, padx=10, pady=5)
        self.negative_network_combo = ttk.Combobox(structures_info_frame, values=DROP_DOWN_OPTIONS)
        self.negative_network_combo.grid(row=1, column=1, padx=10, pady=5)
        
        streaked_name_label = tk.Label(structures_info_frame, text="Branched Streaks")
        streaked_name_label.grid(row=2, column=0, padx=10, pady=5)
        self.streaked_combo = ttk.Combobox(structures_info_frame, values=DROP_DOWN_OPTIONS)
        self.streaked_combo.grid(row=2, column=1, padx=10, pady=5)

        milia_name_label = tk.Label(structures_info_frame, text="Milia-Like Cysts")
        milia_name_label.grid(row=3, column=0, padx=10, pady=5)
        self.milia_combo = ttk.Combobox(structures_info_frame, values=DROP_DOWN_OPTIONS)
        self.milia_combo.grid(row=3, column=1, padx=10, pady=5)

        globules_name_label = tk.Label(structures_info_frame, text="Globules")
        globules_name_label.grid(row=4, column=0, padx=10, pady=5)
        self.globules_combo = ttk.Combobox(structures_info_frame, values=DROP_DOWN_OPTIONS)
        self.globules_combo.grid(row=4, column=1, padx=10, pady=5)

        #Section on extra patient information including DOB, gender, and location of lesion
        patient_info_frame =tk.LabelFrame(info_frame, text="Patient information", highlightthickness=0, borderwidth = 0)
        patient_info_frame.grid(row= 2, column=0, padx=20, pady=10, sticky="W")

        dob_name_label = tk.Label(patient_info_frame, text="Date of Birth")
        dob_name_label.grid(row=0, column=0, padx=20, pady=5)
        dob_name_entry = tk.Entry(patient_info_frame)
        dob_name_entry.grid(row=0, column=1, padx=10, pady=5)

        gender_name_label = tk.Label(patient_info_frame, text="Sex")
        gender_name_label.grid(row=1, column=0, padx=20, pady=5)
        gender_name_entry = tk.Entry(patient_info_frame)
        gender_name_entry.grid(row=1, column=1, padx=10, pady=5)

        location_name_label = tk.Label(patient_info_frame, text="Body Location")
        location_name_label.grid(row=2, column=0, padx=20, pady=5)
        location_name_entry = tk.Entry(patient_info_frame)
        location_name_entry.grid(row=2, column=1, padx=10, pady=5)

        buttons_frame =tk.LabelFrame(info_frame, text="buttons", highlightthickness=0, borderwidth = 0)
        buttons_frame.grid(row=3, column=0, padx=20, pady=10, sticky="W")

        #Loads dataset, will be removed in future update
        init_button = tk.Button(buttons_frame, text="Run", command=lambda : self.run())
        init_button.grid(row=0, column=0, sticky="news", padx=20, pady=5)

        load_button = tk.Button(buttons_frame, text="load Image", command=lambda : self.load_image())
        load_button.grid(row=0, column=1, sticky="news", padx=20, pady=5)

        lesion_info_frame = tk.LabelFrame(container, text="Patient information")
        lesion_info_frame.grid(row=0, column=1, padx=20, pady=10, rowspan=3, columnspan=2)

        lesion_name_label = tk.Label(lesion_info_frame, text="No image loaded")
        lesion_name_label.grid(row=0, column=0, padx=0, pady=0, columnspan=2, rowspan=2)

        self.lesion_name_label = tk.Label(lesion_info_frame, text="No image loaded")
        self.lesion_name_label.grid(row=0, column=0, padx=0, pady=0, columnspan=2, rowspan=2)

        #self.seg_name_label = tk.Label(lesion_info_frame, text="No image loaded")
        #self.seg_name_label.grid(row=2, column=0, padx=20, pady=5)

        #self.masks_combo = ttk.Combobox(lesion_info_frame, values=MASK_DROP_DOWN)
        #self.masks_combo.grid(row=3, column=0, padx=10, pady=5)

        #more_info_button = tk.Button(lesion_info_frame, text="show", command=lambda : self.load_mask())
        #more_info_button.grid(row = 3, column = 1, padx=10, pady=5)

    def run(self):
        print("Run pressed")

        variables = [
            int(self.asymmetry_name_entry.get()),
            int(self.globules_combo.get()),
            int(self.milia_combo.get()),
            int(self.negative_network_combo.get()),
            int(self.network_combo.get()),
            int(self.streaked_combo.get()) 
        ]

        weights = self.predictImage(variables)
        
        data = {'Benign Naevi':weights[0], 'Seborriec':weights[1], 'Melanoma':weights[2]}        

        courses = list(data.keys())
        values = list(data.values())
        
        f, axarr = plt.subplots(2,2)
        
        axarr[0,0].imshow(self.loaded_image)
        axarr[1,0].imshow(self.loaded_mask)
        axarr[0,1].barh(courses, values)

        display_img = self.plt2tk()
        
        resized_image = display_img.resize((display_img.width, display_img.height))
        combined_image = self.prepareImage(resized_image)
        
        self.lesion_name_label.configure(image=combined_image)
    

    def load_image(self):
        print("Load_Image pressed")

        #Open dialog box and look for png, jpg, and bmp images
        filepath = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("Image file",
                                            "*.png ? *.jpg ? *.bmp"),
                                          ("all files", "*.*")))
        
        #Analyse image and return masks
        mask, p_mask, variables = self.analyseImage(filepath)
        
        #Save masks and lablelled images for displaying
        self.loaded_image = Image.open(filepath)
        self.loaded_mask = Image.fromarray(mask)
        p_mask = Image.fromarray(p_mask)

        #Set boxes to the values automatically detected using analyseImage()
        self.asymmetry_name_entry.delete(0, tk.END)
        self.asymmetry_name_entry.insert(0, variables[0])

        self.globules_combo.current(variables[1]+1)
        self.milia_combo.current(variables[2]+1)
        self.negative_network_combo.current(variables[3]+1)
        self.network_combo.current(variables[4]+1)
        self.streaked_combo.current(variables[5]+1)
        
        #Display images in 'lesion_name_label' relating to combobox value
        
        self.run()
        
        
        

    #Checks combobox value and displays the corrisponding image
    def load_mask(self):
        value = self.masks_combo.get()
        image = self.tk_mask
        
        print('Showing image ' + value + ' in load_mask()')

        #Checks value of combobox and gets the relevant image
        if value == 'Segmentation':
            image = self.tk_mask
        elif value == 'Pigment':
            image = self.tk_pigment

        return image
