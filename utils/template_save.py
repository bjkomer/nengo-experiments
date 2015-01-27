# Browse through images saved to a pickle file. Pick out images to save as
# template images for use in template matching

import cPickle as pickle
import sys
from Tkinter import *
#import Image
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import cv2

if len(sys.argv) > 1:
  filename = str(sys.argv[1])
else:
  print("Need to specify file name")
  exit()

video = pickle.load( open( filename, "rb" ) )
cur_index = 0 # Index of currently displayed image

rec_im = np.zeros( (128, 128), dtype='uint8' )

fig = plt.figure(1)
ax = fig.add_subplot(121)
ax.set_title("DVS Image")
display = ax.imshow( rec_im, cmap=plt.cm.gray, vmin=0, vmax=127 )

ax_template = fig.add_subplot(122)
ax_template.set_title("Template")
template_display = ax_template.imshow( rec_im, cmap=plt.cm.gray, vmin=0, vmax=127 )

fig.show()
display.axes.figure.canvas.draw()
template_display.axes.figure.canvas.draw()

master = Tk()

master.wm_title("Template Saving")

def get_template():
  global cur_index
  tlx = int(top_left_x.get())
  tly = int(top_left_y.get())
  w = int(width.get())
  h = int(height.get())

  # FIXME: might need to multiply by two because images seem to be saved with
  # 256 as the max value rather than 128
  #return video[cur_index][tly:tly+h,tlx:tlx+w]*2

  return video[cur_index][tly:tly+h,tlx:tlx+w]

def save():
  name = entry.get()
  misc.imsave(name+'.png', get_template())

def rectangle(i=0):
  global cur_index
  tlx = int(top_left_x.get())
  tly = int(top_left_y.get())
  w = int(width.get())
  h = int(height.get())

  top_left = (tlx, tly)
  bottom_right = (tlx + w, tly + h)

  rec_im = video[cur_index].copy()
  cv2.rectangle(rec_im, top_left, bottom_right, 127, 1)
  template_display.set_data(rec_im)
  template_display.axes.figure.canvas.draw()

def slider_move( index ):
  global cur_index
  cur_index = int(index)
  display.set_data(video[cur_index])
  display.axes.figure.canvas.draw()

  rectangle()

# Choose which image from the video to be displayed
slider = Scale(master, from_=0, to=len(video)-1, command=slider_move)
slider.grid(row=0, column=0)

# Press to save image
button = Button(master, text="Save", command=save)
button.grid(row=0, column=1)

# Name of image file to be saved
entry = Entry(master)
entry.grid(row=0, column=2)

# Controls for the bounding rectangle
"""
top_left_x = Entry(master)
top_left_x.grid(row=1, column=0)
top_left_x.insert(0, "0")

top_left_y = Entry(master)
top_left_y.grid(row=1, column=1)
top_left_y.insert(0, "0")

width = Entry(master)
width.grid(row=1, column=2)
width.insert(0, "10")

height = Entry(master)
height.grid(row=1, column=3)
height.insert(0, "10")
"""
top_left_x = Scale(master, from_=0, to=127, command=rectangle)
top_left_x.grid(row=1, column=0)

top_left_y = Scale(master, from_=0, to=127, command=rectangle)
top_left_y.grid(row=1, column=1)

width = Scale(master, from_=0, to=127, command=rectangle)
width.grid(row=1, column=2)

height = Scale(master, from_=0, to=127, command=rectangle)
height.grid(row=1, column=3)

show_rect = Button(master, text="Update Rectange", command=rectangle)
show_rect.grid(row=1, column=4)

master.mainloop()
