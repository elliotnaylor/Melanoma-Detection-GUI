import os

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image 

from core.bayesian import *
from utils.plot import *

import csv

import matplotlib.pyplot as plt
from io import BytesIO

def csv_to_array(path):
    return np.genfromtxt (path, delimiter=",", dtype=str)

class MainApplication(tk.Tk) :

    image_main = []
    image_graph = []

    path = 'D:/Datasets/ISIC_2018/ISIC_2017_GroundTruth_complete5.csv'
    path_data = 'D:/Datasets/ISIC_2018/ISIC_2017_GroundTruth_complete.csv'
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        # Adding a title to the window
        self.wm_title("Test Application")


        self.Bf = bayesianFusion(self.path)


        # creating a frame and assigning it to container
        container = tk.Frame(self, height=400, width=600)
        # specifying the region where the frame is packed in root
        container.pack(side="top", fill="both", expand=True)

        DROP_DOWN_OPTIONS = [0, 1]

        # configuring the location of the container using grid
        container.grid_rowconfigure(2, weight=1)
        container.grid_columnconfigure(2, weight=1)
        
        info_frame =tk.LabelFrame(container, text="Information")
        info_frame.grid(row=0,rowspan=3, column=0, padx=20, pady=20)

        # Saving User Info
        abcd_info_frame =tk.LabelFrame(info_frame, text="ABCD Rules")
        abcd_info_frame.grid(row= 0, column=0, padx=20, pady=20)

        asymmetry_name_label = tk.Label(abcd_info_frame, text="Asymmetry")
        asymmetry_name_label.grid(row=0, column=0, padx=10, pady=10)
        asymmetry_name_entry = tk.Entry(abcd_info_frame)
        asymmetry_name_entry.grid(row=0, column=1, padx=10, pady=10)

        border_name_label = tk.Label(abcd_info_frame, text="Border")
        border_name_label.grid(row=1, column=0, padx=10, pady=10)
        border_name_entry = tk.Entry(abcd_info_frame)
        border_name_entry.grid(row=1, column=1, padx=10, pady=10)

        colour_name_label = tk.Label(abcd_info_frame, text="Colour")
        colour_name_label.grid(row=2, column=0, padx=10, pady=10)
        colour_name_entry = tk.Entry(abcd_info_frame)
        colour_name_entry.grid(row=2, column=1, padx=10, pady=10)


        dermo_name_label = tk.Label(abcd_info_frame, text="Dermoscopic")
        dermo_name_label.grid(row=3, column=0, padx=10, pady=10)
        self.dermo_name_entry = tk.Entry(abcd_info_frame, state="disabled")
        self.dermo_name_entry.grid(row=3, column=1, padx=10, pady=10)

        #Section on dermoscopic structures
        structures_info_frame =tk.LabelFrame(info_frame, text="Dermoscopic Structures")
        structures_info_frame.grid(row= 1, column=0, padx=20, pady=20)

        network_name_label = tk.Label(structures_info_frame, text="Pigment Network")
        network_name_label.grid(row=0, column=0, padx=10, pady=10)
        self.network_combo = ttk.Combobox(structures_info_frame, values=DROP_DOWN_OPTIONS)
        self.network_combo.grid(row=0, column=1, padx=10, pady=10)
        
        negative_network_name_label = tk.Label(structures_info_frame, text="Negative Network")
        negative_network_name_label.grid(row=1, column=0, padx=10, pady=10)
        self.negative_network_combo = ttk.Combobox(structures_info_frame, values=DROP_DOWN_OPTIONS)
        self.negative_network_combo.grid(row=1, column=1, padx=10, pady=10)
        
        streaked_name_label = tk.Label(structures_info_frame, text="Branched Streaks")
        streaked_name_label.grid(row=2, column=0, padx=10, pady=10)
        self.streaked_combo = ttk.Combobox(structures_info_frame, values=DROP_DOWN_OPTIONS)
        self.streaked_combo.grid(row=2, column=1, padx=10, pady=10)

        milia_name_label = tk.Label(structures_info_frame, text="Milia-Like Cysts")
        milia_name_label.grid(row=3, column=0, padx=10, pady=10)
        self.milia_combo = ttk.Combobox(structures_info_frame, values=DROP_DOWN_OPTIONS)
        self.milia_combo.grid(row=3, column=1, padx=10, pady=10)

        globules_name_label = tk.Label(structures_info_frame, text="Globules")
        globules_name_label.grid(row=4, column=0, padx=10, pady=10)
        self.globules_combo = ttk.Combobox(structures_info_frame, values=DROP_DOWN_OPTIONS)
        self.globules_combo.grid(row=4, column=1, padx=10, pady=10)

        #Section on extra patient information including DOB, gender, and location of lesion
        patient_info_frame =tk.LabelFrame(info_frame, text="Patient information")
        patient_info_frame.grid(row= 2, column=0, padx=20, pady=20)

        dob_name_label = tk.Label(patient_info_frame, text="Date of Birth")
        dob_name_label.grid(row=0, column=0, padx=20, pady=10)
        dob_name_entry = tk.Entry(patient_info_frame)
        dob_name_entry.grid(row=0, column=1, padx=10, pady=10)

        gender_name_label = tk.Label(patient_info_frame, text="Gender")
        gender_name_label.grid(row=1, column=0, padx=20, pady=10)
        gender_name_entry = tk.Entry(patient_info_frame)
        gender_name_entry.grid(row=1, column=1, padx=10, pady=10)

        location_name_label = tk.Label(patient_info_frame, text="Body Location")
        location_name_label.grid(row=2, column=0, padx=20, pady=10)
        location_name_entry = tk.Entry(patient_info_frame)
        location_name_entry.grid(row=2, column=1, padx=10, pady=10)


        buttons_frame =tk.LabelFrame(container, text="buttons")
        buttons_frame.grid(row=0, column=1, padx=20, pady=20)

        #Loads dataset, will be removed in future update
        init_button = tk.Button(buttons_frame, text="Initalise", command=lambda : self.run())
        init_button.grid(row=0, column=0, sticky="news", padx=20, pady=10)

        load_button = tk.Button(buttons_frame, text="load Image", command=lambda : self.load_image())
        load_button.grid(row=0, column=1, sticky="news", padx=20, pady=10)

        lesion_info_frame = tk.LabelFrame(container, text="Patient information")
        lesion_info_frame.grid(row=1, column=1, padx=0, pady=20)
        lesion_name_label = tk.Label(lesion_info_frame, width=40, height=20, text="No data loaded")
        lesion_name_label.grid(row=1, column=0, padx=20, pady=10)

        self.lesion_name_label = tk.Label(lesion_info_frame, text="No image loaded")
        self.lesion_name_label.grid(row=1, column=0, padx=20, pady=10)

        bayesian_info_frame = tk.LabelFrame(container, text="Bayesian")
        bayesian_info_frame.grid(row=1, column=2, padx=20, pady=20)
        bayesian_name_label = tk.Label(bayesian_info_frame, width=40, height=20, text="No data loaded")
        bayesian_name_label.grid(row=1, column=0, padx=20, pady=10)

        self.bayesian_name_label = tk.Label(bayesian_info_frame, text="No data loaded")
        self.bayesian_name_label.grid(row=1, column=0, padx=20, pady=10)
    
    def run(self):
        print("Run pressed")

        globules = int(self.globules_combo.get())
        milia = int(self.milia_combo.get())
        negative = int(self.negative_network_combo.get())
        network = int(self.network_combo.get())
        streaked = int(self.streaked_combo.get())
        dermo = globules + milia + negative + network + streaked

        if dermo > 4:
            dermo = 4

        weights = self.Bf.predict(
            globules, 
            milia, 
            negative, 
            network, 
            streaked, 
            dermo)

        data = {'Benign Naevi':weights[0], 'Seborriec':weights[1], 'Melanoma':weights[2]}        

        courses = list(data.keys())
        values = list(data.values())

        #fig = plt.figure(figsize = (10, 10))

        plt.bar(courses, values)

        buffer = BytesIO()
        plt.savefig(buffer,format='png')
        image = Image.open(buffer)

        image.thumbnail((360, 360))
        
        self.image_graph = ImageTk.PhotoImage(image)

        self.bayesian_name_label.configure(image = self.image_graph)

        plt.clf()




    def load_image(self):

        #Load image from file location
        print("Load_Image pressed")

        filepath = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("Image file",
                                                        "*.png ? *.jpg ? *.bmp"),
                                                       ("all files",
                                                "*.*")))
        image = Image.open(filepath)

        image.thumbnail((360, 360))

        #Update tje lesion label with the image
        self.image_main = ImageTk.PhotoImage(image)
        self.lesion_name_label.configure(image=self.image_main)

        #Load variables from image

        filename = os.path.basename(filepath)

        filename = filename.split('.', 1)[0]

        array = csv_to_array(self.path_data)

        for i in range(0, len(array)):
            if filename == array[i][0]:

                self.globules_combo.current(array[i][4])
                self.milia_combo.current(array[i][5])
                self.negative_network_combo.current(array[i][6])
                self.network_combo.current(array[i][7])
                self.streaked_combo.current(array[i][8])

                self.dermo_name_entry.configure(state="normal")
                self.dermo_name_entry.delete(0, tk.END)
                self.dermo_name_entry.insert(0, array[i][9])
                self.dermo_name_entry.configure(state="disabled")
                self.run()
            






if __name__ == "__main__":
    testObj = MainApplication()
    testObj.mainloop()