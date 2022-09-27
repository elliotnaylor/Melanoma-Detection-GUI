from tkinter import *
import tkinter.messagebox
from tkinter import filedialog
import customtkinter
from PIL import Image, ImageTk

import numpy as np
import cv2
from core.asymmetry import Asymmetry
from utils.cv import *
from utils.xlrd import *

from load_PH2 import PH2
from score import get_images_comparison

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

import matplotlib.pyplot as plt

import os

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"

class App(customtkinter.CTk):

    WIDTH = 780
    HEIGHT = 520

    IMG_SIZE = (360,360)

    PATH = os.getcwd()

    ph2 = PH2()
    asymmetry = Asymmetry()

    def __init__(self):
        super().__init__()

        self.title("Melanoma Detector")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed

        # ============ create two frames ============

        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = customtkinter.CTkFrame(master=self,
                                                 width=180,
                                                 corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        # ============ frame_left ============

        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize=10)   # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(5, weight=1)  # empty row as spacing
        self.frame_left.grid_rowconfigure(8, minsize=20)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        self.label_1 = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Melanoma Detector",
                                              text_font=("Roboto Medium", -16))  # font name and size in px
        self.label_1.grid(row=1, column=0, pady=10, padx=10)

        self.button_1 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Load Image",
                                                command=self.button_event_load)
        self.button_1.grid(row=2, column=0, pady=10, padx=20)

        self.button_2 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Run",
                                                command=self.button_event_svm)
        self.button_2.grid(row=3, column=0, pady=10, padx=20)

        self.button_3 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Pictures",
                                                command=self.button_event_run)
        self.button_3.grid(row=5, column=0, pady=10, padx=20)

        self.label_mode = customtkinter.CTkLabel(master=self.frame_left, text="Appearance Mode:")
        self.label_mode.grid(row=9, column=0, pady=0, padx=20, sticky="w")

        self.optionmenu_1 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                        values=["Light", "Dark", "System"],
                                                        command=self.change_appearance_mode)
        self.optionmenu_1.grid(row=10, column=0, pady=10, padx=20, sticky="w")

        # ============ frame_right ============

        # configure grid layout (3x7)
        self.frame_right.rowconfigure((0, 1, 2, 3), weight=1)
        self.frame_right.rowconfigure(7, weight=10)
        self.frame_right.columnconfigure((0, 2), weight=1)
        self.frame_right.columnconfigure(2, weight=0)

        self.frame_info = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_info.grid(row=0, column=0, columnspan=8, rowspan=3, pady=20, padx=20, sticky="nsew")

        # ============ frame_info ============

        image = Image.open(self.PATH + "/images/IMD004.bmp")
        image.thumbnail(self.IMG_SIZE)
        self.image_main = ImageTk.PhotoImage(image)
        
        # configure grid layout (1x1)
        self.frame_info.rowconfigure(0, weight=1)
        self.frame_info.columnconfigure(0, weight=1)

        self.label_image = tkinter.Label(master=self, image=self.image_main)

        self.label_info_1 = customtkinter.CTkLabel(master=self.frame_info, image=self.image_main,
                                                   justify=tkinter.LEFT
                                                   )
        self.label_info_1.grid(column=0, row=0, sticky="nwe", padx=15, pady=15)

        self.progressbar = customtkinter.CTkProgressBar(master=self.frame_info)
        self.progressbar.grid(row=1, column=0, sticky="ew", padx=15, pady=15)

        # ============ frame_right ============

        self.radio_var = tkinter.IntVar(value=0)

        self.check_box_1 = customtkinter.CTkRadioButton(master=self.frame_right,
                                                     variable=self.radio_var,
                                                     value=0,
                                                     text="None")
        self.check_box_1.grid(row=6, column=1, pady=10, padx=20, sticky="w")

        self.check_box_2 = customtkinter.CTkRadioButton(master=self.frame_right,
                                                     variable=self.radio_var,
                                                     value=1,
                                                     text="Asymmetry")
        self.check_box_2.grid(row=6, column=2, pady=10, padx=20, sticky="w")

        self.check_box_3 = customtkinter.CTkRadioButton(master=self.frame_right,
                                                     variable=self.radio_var,
                                                     value=2,
                                                     text="Border")
        self.check_box_3.grid(row=6, column=3, pady=10, padx=20, sticky="w")

        self.check_box_4 = customtkinter.CTkRadioButton(master=self.frame_right,
                                                     variable=self.radio_var,
                                                     value=3,
                                                     text="Colour")
        self.check_box_4.grid(row=6, column=4, pady=10, padx=20, sticky="w")

        self.check_box_5 = customtkinter.CTkRadioButton(master=self.frame_right,
                                                     variable=self.radio_var,
                                                     value=4,
                                                     text="Structure")
        self.check_box_5.grid(row=6, column=5, pady=10, padx=20, sticky="w")

    def button_event_load(self):
        global select
        global original
        global image_main

        select = filedialog.askopenfilename(initialdir = "*", filetypes = [('Image files', '*.png'),('Image files', '*.jpg'),('Image files', '*.bmp')])

        if not select:
            return
        print(select)

        try:
            original = cv2.imread(select)

            im = Image.open(select)
            im.thumbnail(self.IMG_SIZE)
            tkimage = ImageTk.PhotoImage(im)

            img_rgb = np.array(im)

            hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            self.image_main=tkimage
            self.label_info_1.configure(image=tkimage)
        except:
            return

    def button_event_run(self):
        print("button pressed")

        images_tk = self.ph2.load_images_tk()
        images = self.ph2.load_images()
        masks = self.ph2.load_masks()

        #image_2 = Image.open(self.PATH + "/images/IMD004.bmp")
        #image_2.thumbnail((100, 100))

        masked = []
        for i in range(0, len(images)):
            element = apply_mask_cv(images[i], masks[i])
            masked.append(element)

        
        
        self.img_similar = []

        #images = get_images_comparison(6)
        #Similar skin lesions display, add to "img_similar" to show results
        self.img_similar.append(ImageTk.PhotoImage(images_tk[10]))


        data = []
        dataX = []
        dataY = []
        #run()
        
        for i in range(150, 200):
            dataH, dataV, asymmetry = self.asymmetry.run(images[i], masked[i], masks[i])
            data.append(asymmetry)
            print(asymmetry)
            dataX.append(dataH)
            dataY.append(dataV)

        book = xlwt.Workbook()
        save_output2d(dataX, 'Horizontal', book)
        save_output2d(dataY, 'Vertical', book)
        save_output1d(data, 'Results', book)
        book.save('output.xls')

        for i in range(0, len(self.img_similar)):

            self.label_image_2 = tkinter.Label(master=self, image=self.img_similar[i])

            self.label_info_2 = customtkinter.CTkLabel(master=self.frame_right, image=self.img_similar[i],
                                                        justify=tkinter.LEFT)

            self.label_info_2.grid(column=i+1, row=8, sticky="nwe", padx=5, pady=5)

    def button_event_svm(self):
        path = os.path.join(os.getcwd() + '/SVM test data.xls')
        x, y = self.ph2.load_test_data(path, 1)
        #x2, y2 = self.ph2.load_test_data(path, 2)
        clf = make_pipeline(StandardScaler(), SVC(kernel = 'rbf', gamma='auto'))
        model = clf.fit(x, y)
    
        #clf.predict(x2, y2)
        
        print(accuracy_score(y, clf.predict(x)))
    

        x = np.array(x)

        x_sep, y_sep = [], []
        for i in range(len(x)):
            if y[i] != 3:
                y_sep.append(x[i][0])
                x_sep.append(x[i][1])

        x0, x1 = x[:, 0], x[:, 1]

        fig, ax = plt.subplots()
        
        disp = DecisionBoundaryDisplay.from_estimator(
            clf,
            x,
            response_method="predict",
            cmap=plt.cm.coolwarm,
            alpha=0.8,
            ax=ax
        )

        ax.scatter(x0, x1, c=y, cmap=plt.cm.coolwarm, s=5, edgecolors='k', linewidth=0.5)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title('SVM with RBF Kernel for Colour Asymmetry Detection')
        ax.set_xlabel('Size of Skin Lesion (100 / sample_size * i)')
        ax.set_ylabel('Colour difference (3D euclidean distance)')
        plt.show()

    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def on_closing(self, event=0):
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()

