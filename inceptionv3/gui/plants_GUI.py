# Working GUI!!!

import numpy as np
from keras import layers
import keras.backend as K
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img, img_to_array
from plants_predict import get_image, create_model # IMP: NEED THIS FILE!!!

from tkinter import *
from tkinter import ttk
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2

class Root(Tk):
    # 'self' refers to the Tkinter object ("this" in C++)
    
    def __init__(self): # Constructor
        super(Root, self).__init__()
        # Initialize Tkinter parameters
        self.title("Plant Disease Classifer!")
        self.minsize(400, 400)  
        self.configure(background = '#8ee283')
        self.label0=tk.Label(self, text="Welcome to the Plant Disease Classifier!", background='#8ee283', font=18)
        self.label0.text = "Welcome to the Plant Disease Classifier!"
        self.label0.place(x=100, y=50, anchor='center')
        self.label0.pack()
        self.button()
        weights_path = "Documents/toda_okura_inceptionv3.h5"
        global model, labels
        model = create_model(weights_path, 224, 224)
        labels = self.get_plant_labels('plant_labels.txt')
        
    def get_plant_labels(self, path):
        with open(path, 'r') as f:
            labels = f.readlines()
            return labels

    def button(self):
        self.button = ttk.Button(self, text="Upload image!", command=self.display_image)
        self.button.place(x=160, y=200, anchor='center')
        self.button.pack()
        
    def get_predictions(self, filename):
        test_image = get_image(filename, 224, 224)
        predictions = model.predict(test_image[np.newaxis,:])
        max_pred_idx = np.argmax(predictions)
        max_pred = str(predictions[0][max_pred_idx] * 100)
        print("Prediction: ", labels[max_pred_idx][:-1])
        prediction = "Prediction: " + labels[max_pred_idx][:-1] + ""
        accuracy = "Accuracy: " + max_pred[:5] + "%"
        
        self.label1=tk.Label(self, text=prediction, background='#8ee283')
        self.label1.text = prediction
        self.label1.place(x=200, y=330, anchor='center', height=20, width=400)
        
        self.label2=tk.Label(self, text=accuracy, background='#8ee283')
        self.label2.text = accuracy
        self.label2.place(x=200, y=350, anchor='center', height=20, width=400)

    def display_image(self):
        filename = filedialog.askopenfilename()
        if(filename != ""):
            print('Selected:', filename)
            my_img = cv2.imread(filename)
            b,g,r = cv2.split(my_img)
            my_img = cv2.merge((r,g,b))
            my_img = cv2.resize(my_img, (224, 224), interpolation = cv2.INTER_LINEAR) # Resize image
            img = ImageTk.PhotoImage(Image.fromarray(my_img))
            self.label=tk.Label(root, image=img)
            self.label.image=img
            self.label.place(x=200, y=185, anchor='center')
            self.get_predictions(filename)
        else: print('Error: Could not retrieve file!')

root = Root()
root.mainloop()