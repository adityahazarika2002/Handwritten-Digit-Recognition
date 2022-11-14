import tkinter as tk
import numpy as np
from tkinter import *
from PIL import Image, ImageOps
from pathlib import Path
from numpy import asarray
from recognition import predict
from add_data import add_input
from keras_preprocessing import image

cwd = Path.cwd()

main = tk.Tk()
main.title('Digit Recognition')

main.geometry("380x280")
main.minsize(480, 280)
main.maxsize(480, 280)

main.configure(bg='grey')

def prediction(canvas,fileName):
    # save postscipt image 
    canvas.postscript(file='images/postscript.eps') 
    # use PIL to convert to PNG 
    img = Image.open('images/postscript.eps')
    img.save(fileName + '.jpeg', 'jpeg')
    img = image.load_img(fileName + '.jpeg', 'jpeg', color_mode='grayscale', target_size=(28, 28))
    img = ImageOps.invert(img)
    img.save(fileName + '.jpeg', 'jpeg')
    img = image.load_img(fileName + '.jpeg', 'jpeg', color_mode='grayscale', target_size=(28, 28))

    global img_data
    data = asarray(img)
    img_data = list(img.getdata())
    img_data = np.array(img_data)
    img_data = img_data.reshape(-1, 784)
    output = predict(data)
    return output

def add_user_data(entry):
    add_input(img_data, entry)
    

def get_xy(event):
    global lasx, lasy
    lasx, lasy = event.x, event.y

def draw_canvas(event):
    global lasx, lasy
    canvas.create_line((lasx, lasy, event.x, event.y), fill='black', width=25)
    lasx, lasy = event.x, event.y

def clear_canvas():
    canvas.delete("all")

def update_text():
   label.configure(text = f'Predicted : {prediction(canvas, f"images/img")}')

canvas = tk.Canvas(main, cursor="dot", width=280, height=280)
canvas.place(x=0, y=0)
canvas.bind("<Button-1>", get_xy)
canvas.bind("<B1-Motion>", draw_canvas)

label = tk.Label(main, bg = "gray",text = f'Predicted : {prediction(canvas, f"images/img")}')
label.place(x=293, y=5)

label1 = tk.Label(main, bg = "gray",text = "Annotate : ")
label1.place(x=293, y=45)

entry=Entry(main, width=20, textvariable="annotation")
entry.insert(0,"0")
entry.place(x=360, y=45, width=15)

button = tk.Button(main, text='Predict', width =7, command=lambda:[prediction(canvas, f"images/img"), update_text()])
button.place(x=293, y=250)

button1 = tk.Button(main, text='Insert', width =7, command=add_user_data(int(entry.get())))
button1.place(x=353, y=250)

button2 = tk.Button(main, text='Reset', width =7, command=clear_canvas)
button2.place(x=413, y=250)



main.mainloop()