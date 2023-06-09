import tkinter as tk
from tkinter import ttk

class interface(tk.Tk):
    
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        # Adding a title to the window
        self.wm_title("Test Application")

        # creating a frame and assigning it to container
        container = tk.Frame(self, height=400, width=600)
        # specifying the region where the frame is packed in root
        container.pack(side="top", fill="both", expand=True)

        DROP_DOWN_OPTIONS = ["Yes", "No", "Don't know"]

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

        

        dermo_name_entry = tk.Entry(abcd_info_frame)
        dermo_name_entry.grid(row=3, column=1, padx=10, pady=10)

        #Section on dermoscopic structures
        structures_info_frame =tk.LabelFrame(info_frame, text="Dermoscopic Structures")
        structures_info_frame.grid(row= 1, column=0, padx=20, pady=20)

        network_name_label = tk.Label(structures_info_frame, text="Pigment Network")
        network_name_label.grid(row=0, column=0, padx=10, pady=10)
        network_combo = ttk.Combobox(structures_info_frame, values=DROP_DOWN_OPTIONS)
        network_combo.grid(row=0, column=1, padx=10, pady=10)
        
        negative_network_name_label = tk.Label(structures_info_frame, text="Negative Network")
        negative_network_name_label.grid(row=1, column=0, padx=10, pady=10)
        negative_network_combo = ttk.Combobox(structures_info_frame, values=DROP_DOWN_OPTIONS)
        negative_network_combo.grid(row=1, column=1, padx=10, pady=10)
        
        streaked_name_label = tk.Label(structures_info_frame, text="Branched Streaks")
        streaked_name_label.grid(row=2, column=0, padx=10, pady=10)
        streaked_combo = ttk.Combobox(structures_info_frame, values=DROP_DOWN_OPTIONS)
        streaked_combo.grid(row=2, column=1, padx=10, pady=10)

        milia_name_label = tk.Label(structures_info_frame, text="Milia-Like Cysts")
        milia_name_label.grid(row=3, column=0, padx=10, pady=10)
        milia_combo = ttk.Combobox(structures_info_frame, values=DROP_DOWN_OPTIONS)
        milia_combo.grid(row=3, column=1, padx=10, pady=10)

        globules_name_label = tk.Label(structures_info_frame, text="Globules")
        globules_name_label.grid(row=4, column=0, padx=10, pady=10)
        globules_combo = ttk.Combobox(structures_info_frame, values=DROP_DOWN_OPTIONS)
        globules_combo.grid(row=4, column=1, padx=10, pady=10)

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
        init_button = tk.Button(buttons_frame, text="Initalise", command= self.run)
        init_button.grid(row=0, column=0, sticky="news", padx=20, pady=10)

        load_button = tk.Button(buttons_frame, text="load Image", command= self.run)
        load_button.grid(row=0, column=1, sticky="news", padx=20, pady=10)

        lesion_info_frame = tk.LabelFrame(container, text="Patient information")
        lesion_info_frame.grid(row=1, column=1, padx=20, pady=20)
        lesion_name_label = tk.Label(lesion_info_frame, width=40, height=20, text="No image loaded")
        lesion_name_label.grid(row=1, column=0, padx=20, pady=10)

        bayesian_info_frame = tk.LabelFrame(container, text="Bayesian")
        bayesian_info_frame.grid(row=1, column=2, padx=20, pady=20)
        bayesian_name_label = tk.Label(bayesian_info_frame, width=40, height=20, text="No data loaded")
        bayesian_name_label.grid(row=1, column=0, padx=20, pady=10)




    def run():
        pass


if __name__ == "__main__":
    testObj = interface()
    testObj.mainloop()