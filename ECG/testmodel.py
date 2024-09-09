import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox

class_name = {5: 'Ventricular ectopic beat', 2: 'Unknown beat', 3: 'Normal beat', 1: 'Myocardial infarction ', 0: 'Supraventricular ectopic beat', 4: 'Fusion beat'}

# Load the trained model
model = tf.keras.models.load_model('vgg16_best_model.keras')

# Function to prepare the image for prediction
def prepare_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Function to load an image
def load_image():
    global image_path, img_label, loaded_image
    image_path = filedialog.askopenfilename()

    if not image_path:
        return
    try:
        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror("Error", "Cannot open the selected image file.")
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        loaded_image = ImageTk.PhotoImage(image_pil)
        img_label.configure(image=loaded_image)
        img_label.image = loaded_image  # Prevent garbage collection of the image
    except Exception as e:
        print(e)

# Function to predict the class of the image
def predict_image():
    if not image_path:
        messagebox.showerror("Error", "Please load an image first.")
        return
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "Cannot open the selected image file.")
        return
    prepared_image = prepare_image(image)
    predictions = model.predict(prepared_image)
    predicted_class = np.argmax(predictions, axis=1)
    result_label.configure(text=f'Predicted class: {class_name[predicted_class[0]]}')

# Setting up the customtkinter interface
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Image Classification")
root.geometry("600x400")  # Set window size (width x height)
root.grid_columnconfigure(0, weight=0)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0,weight=0)
root.grid_rowconfigure(1,weight=0)
root.grid_rowconfigure(2,weight=0)

# Button to load the image
load_button = ctk.CTkButton(root, text="Load Image", command=load_image)
load_button.grid(row=0,column=0,sticky="w")

# Label to display the image
img_label = ctk.CTkLabel(root,text="",height=300)
img_label.grid(row=0,column=1,sticky="w",padx=20)

# Button to predict the image class
predict_button = ctk.CTkButton(root, text="Predict", command=predict_image)
predict_button.grid(row=1,column=0)

# Label to display the result
result_label = ctk.CTkLabel(root, text="Predicted class: ")
result_label.grid(row=2,column=0)

# Run the interface
root.mainloop()
