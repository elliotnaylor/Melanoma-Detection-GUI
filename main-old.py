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

from dataset.load_PH2 import PH2

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
import pickle
from joblib import dump, load

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


    images = ph2.load_images()
    masks = ph2.load_masks()

    masked = []

    for i in range(0, 187):
        masked.append(apply_mask_cv(images[i], masks[i]))

    model = []

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
        self.frame_right.grid(row=0, column=1, sticky="nswe")

        # ============ frame_left ============

        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize=10)   # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(6, weight=1)  # empty row as spacing
        self.frame_left.grid_rowconfigure(8, minsize=20)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        self.label_1 = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Melanoma Detector",
                                              text_font=("Roboto Medium", -16))  # font name and size in px
        self.label_1.grid(row=1, column=0, pady=10, padx=10)

        self.button_1 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Load Image",
                                                command=self.button_event_load)
        self.button_1.grid(row=2, column=0, pady=20, padx=20)

        self.button_2 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Asymmetry",
                                                command=self.load_asymmetry)
        self.button_2.grid(row=3, column=0, pady=10)
        
        self.button_3= customtkinter.CTkButton(master=self.frame_left,
                                                text="Border",
                                                command=self.load_border)
        self.button_3.grid(row=4, column=0, pady=10)

        self.button_4= customtkinter.CTkButton(master=self.frame_left,
                                                text="Colour",
                                                command=self.load_colour)
        self.button_4.grid(row=5, column=0, pady=10)

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
        self.frame_info.grid(row=0, column=0, columnspan=8, rowspan=3, sticky="nsew")

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

        #self.button_event_run()
        self.button_event_load_model()

        # create scrollable textbox
        tk_textbox = tkinter.Text(self, highlightthickness=0)
        tk_textbox.grid(row=1, column=2, sticky="nsew")

        # create CTk scrollbar
        ctk_textbox_scrollbar = customtkinter.CTkScrollbar(self, command=tk_textbox.yview)
        ctk_textbox_scrollbar.grid(row=1, column=3, sticky="ns")


    def load_asymmetry(self):

        self.asymmetry.run(self.images[0], self.masked[0], self.masks[0])
    
        

    def load_border(self):
        pass

    def load_colour(self):
        pass

    def load_similarity(self):
        pass
    
    #Currently loads image from file
    #Should instead load from index in dataset
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

    def button_asymmetry(self):
        print("Asymmetry pressed")

        v_data, h_data = [], []

        #1. Segment skin lesion
        images = self.ph2.load_images()
        masks = self.ph2.load_masks()

        #2. Apply mask to skin lesion
        for i in range(0, 200):
            masked = apply_mask_cv(images[i], masks[i])

            dataH, dataV, asymmetry = self.asymmetry.run(images[i], masked, masks[i])

            dataV.sort()
            dataH.sort() #Sorts in ascending order

            vertical = []
            horizontal = []

        #4. Add Y value based on number of values
            for i in range(0, len(dataV)):

                vertical.append([100 / len(dataV) * i, dataV[i]])

            for i in range(0, len(dataH)):

                horizontal.append([100 / len(dataH) * i, dataH[i]])

            #4. predict using model
            v_prd = self.model.predict(vertical)
            h_prd = self.model.predict(horizontal)

            vertical_sym, horizontal_sym = 0,0
        
            for v in v_prd:
                if v == 0:
                    vertical_sym += 1
                elif v == 2:
                    vertical_sym -= 1

            for h in h_prd:
                if h == 0:
                    horizontal_sym += 1
                elif h == 2:
                    horizontal_sym -= 1

            v_data.append(vertical_sym)
            h_data.append(horizontal_sym)

        book = xlwt.Workbook()
        save_output1d(v_data, 'Vertical', book)
        save_output1d(h_data, 'Horizontal', book)
        book.save('output_final.xls')
    
        
    def button_event_run(self):
        print("Run pressed")

        images_tk = self.ph2.load_images_tk()
        images = self.ph2.load_images()
        masks = self.ph2.load_masks()

        #image_2 = Image.open(self.PATH + "/images/IMD004.bmp")
        #image_2.thumbnail((100, 100))

        masked = []
        for i in range(0, len(masks)):
            element = apply_mask_cv(images[i], masks[i])
            masked.append(element)
        
        self.img_similar = []

        #Similar skin lesions display, add to "img_similar" to show results
        self.img_similar.append(ImageTk.PhotoImage(images_tk[10]))

        data = []
        dataX = []
        dataY = []
        #run()
        
        for i in range(0, 200):
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

    #Train an SVM with pre-processed data saved into an .xls file (Saves processing time)
    def button_event_load_model(self): 

        path = os.path.join(os.getcwd() + '/output.xls')#'/SVM test data.xls')
        path_test = os.path.join(os.getcwd() + '/output_test.xls')#'/SVM test data.xls')

        index, xtrain, ytrain = self.ph2.load_test_data(path)
        index, xtest, ytest = self.ph2.load_test_data(path_test)

        newxtrain = []
        newytrain = []
        for i in range(0, len(xtrain)):
            if xtrain[i][0] >= 95:
                newxtrain.append([xtrain[i][0], xtrain[i][1]])
                newytrain.append(ytrain[i])


        clf = make_pipeline(StandardScaler(), SVC(kernel = 'rbf', gamma='auto'))
        self.model = clf.fit(newxtrain, newytrain)

        print('Training accuracy: ' + str(accuracy_score(newytrain, clf.predict(newxtrain))))
        #print('Validation accuracy: ' + str(accuracy_score(ytest, clf.predict(xtest))))

        newxtrain = np.array(newxtrain)
        #xtest = np.array(xtest)
        
        draw_svm_boundries(clf, newxtrain, newytrain)
        #draw_svm_boundries(clf, xtest, ytest)

        #clf = make_pipeline(StandardScaler(), SVC(kernel = 'rbf', gamma='auto'))
        #self.model = clf.fit(xtrain, ytrain)

        #print('Training accuracy: ' + str(accuracy_score(ytrain, clf.predict(xtrain))))
        #print('Validation accuracy: ' + str(accuracy_score(ytest, clf.predict(xtest))))
        
        with open('model.pkl', 'rb') as f:
            clf2 = pickle.load(f)

        #Load SVM model
        xtrain = np.array(xtrain)
        xtest = np.array(xtest)

        #draw_svm_boundries(clf2, xtrain, ytrain)
        #draw_svm_boundries(clf2, xtest, ytest)

    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def on_closing(self, event=0):
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()

    