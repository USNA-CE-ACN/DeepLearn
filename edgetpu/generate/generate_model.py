#!/usr/bin/python3
import sys
from tkinter import *

import numpy as np
import keras
import tensorflow as tf
import tensorflow_datasets as tfds



def donothing():
   filewin = Toplevel(window)
   button = Button(filewin, text="Do nothing button")
   button.pack()

window = Tk()
greeting = Label(text="Hello")

menubar = Menu(window)

filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Open", command=donothing)
filemenu.add_command(label="Load", command=donothing)
filemenu.add_separator()
filemenu.add_command(label="Generate", command=donothing)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=window.quit)

transformmenu = Menu(menubar, tearoff=0)
transformmenu.add_command(label="Add", command=donothing)
transformmenu.add_command(label="Remove", command=donothing)
transformmenu.add_separator()
transformmenu.add_command(label="Width", command=donothing)
transformmenu.add_separator()
transformmenu.add_command(label="Replicate", command=donothing)

menubar.add_cascade(label="File", menu=filemenu)
menubar.add_cascade(label="Transform", menu=transformmenu)

window.config(menu=menubar)

window.mainloop()

