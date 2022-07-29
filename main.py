from tkinter import *
import tkinter.messagebox
from tkinter import filedialog
import customtkinter
from PIL import Image, ImageTk

import numpy as np
import cv2

from load_PH2 import PH2
from score import get_images_comparison

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(customtkinter.CTk, PH2):

    WIDTH = 780
    HEIGHT = 520

    IMG_SIZE = (360,360)

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
                                                command=self.button_event)
        self.button_2.grid(row=3, column=0, pady=10, padx=20)

        self.button_3 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Pictures",
                                                command=self.button_event)
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
        PATH = "C:/Users/scrib/Documents/GitHub/melanoma-gui"

        image = Image.open(PATH + "/images/IMD004.bmp")
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


        self.img_similar = []

        PATH = "C:/Users/scrib/Documents/GitHub/melanoma-gui"

        image_2 = Image.open(PATH + "/images/IMD004.bmp")
        image_2.thumbnail((100, 100))
        
        #Similar skin lesions display, add to "img_similar" to show results
        self.img_similar.append(ImageTk.PhotoImage(image_2))
        self.img_similar.append(ImageTk.PhotoImage(image_2))
        self.img_similar.append(ImageTk.PhotoImage(image_2))

        for i in range(0, len(self.img_similar)):

            self.label_image_2 = tkinter.Label(master=self, image=self.img_similar[i])

            self.label_info_2 = customtkinter.CTkLabel(master=self.frame_right, image=self.img_similar[i],
                                                        justify=tkinter.LEFT)

            self.label_info_2.grid(column=i+1, row=8, sticky="nwe", padx=15, pady=15)

        #image_list = []
        #image_grid = []

        #images = Image.open(PATH + "/images/IMD004.bmp")
        #images.thumbnail((100, 100))
        #images_tk = ImageTk.PhotoImage(images)

        # configure grid layout (1x1)
        #self.frame_info.rowconfigure(0, weight=1)
        #self.frame_info.columnconfigure(0, weight=1)

        #label_image_2 = tkinter.Label(master=self, image=self.image_main)

        #img_grid = customtkinter.CTkLabel(master=self.frame_right, image=images_tk, justify=tkinter.LEFT)

        #img_grid.grid(row=8, column=1, columnspan=5, pady=20, padx=20, sticky="nwe")
        #for i in range(5):
            #pass
            
            #images.append(ImageTk.PhotoImage(image_list[i]))

            #image_grid.append(customtkinter.CTkLabel(master=self.frame_right, image=images_tk, justify=tkinter.LEFT))  # font name and size in px

            #img_grid.grid(row=8, column=i, columnspan=2, pady=20, padx=20, sticky="we")


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
            self.label_info_1.config(image=tkimage)
        except:
            return

    def button_event(self):
        print("button pressed")
        images = get_images_comparison(6)
        pass

    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def on_closing(self, event=0):
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()