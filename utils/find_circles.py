# Finds and displays circles detected in live spiking camera feed
import vrep
import numpy as np
import time
import ctypes
import matplotlib.pyplot as plt
import cv2

match_threshold=0.20

empty_str=""
raw_bytes = (ctypes.c_ubyte * len(empty_str)).from_buffer_copy(empty_str) 

fig = plt.figure(1)
ax = fig.add_subplot(121)
ax.set_title("DVS Image")
image = np.zeros( (128, 128), dtype='uint8' )
display = ax.imshow( image, cmap=plt.cm.gray, vmin=0, vmax = 127 )

ax_detection = fig.add_subplot(122)
ax_detection.set_title("Detected Point")
detection_display = ax_detection.imshow( image, cmap=plt.cm.gray, vmin=0, vmax = 127 )


fig.show()
display.axes.figure.canvas.draw()

def find_circles( image ):
  # Copy image so the circle is drawn on a new image
  img = image.copy()
  #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  #circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100)
  #circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100, minRadius=1)
  #circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, 300, 50, 1)
  #circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, 20, param1=50,
  #                           param2=30, minRadius=0, maxRadius=0)
  #circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, 20, param1=10,
  #                           param2=10, minRadius=0, maxRadius=0)
  circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, 100, param1=25,
                             param2=18, minRadius=6, maxRadius=50)

  if circles is not None:
    print( "Found circles, yay!" )
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
      # draw the circle in the output image, then draw a rectangle
      # corresponding to the center of the circle
      cv2.circle(img, (x, y), r, (0, 255, 0), 4)
      cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
  
  # Update display
  detection_display.set_data(img)
  detection_display.axes.figure.canvas.draw()

cid = vrep.simxStart('127.0.0.1',20002,True,True,5000,5)

if cid != -1:
  print ('Connected to V-REP remote API server, client id: %s' % cid)
  vrep.simxStartSimulation( cid, vrep.simx_opmode_oneshot )
else:
  print ('Failed connecting to V-REP remote API server')
  exit(1)

while True:
  err, data = vrep.simxGetStringSignal(cid, "dataFromThisTimeStep",
                                       vrep.simx_opmode_oneshot)
  err = vrep.simxSetStringSignal(cid, "dataFromThisTimeStep", raw_bytes,
                                 vrep.simx_opmode_oneshot_wait)
  l = len(data)
  for i in range( int(l/4) ):
    b = list(bytearray(data[i*4:i*4+4]))
    x_coord = b[0]
    y_coord = b[1]
    if x_coord > 128:
      x_coord -= 128
      polarity = 1
    else:
      polarity = 0
    timestamp = (b[3] * 256) + b[2]
    
    # Distinguish between different polarities
    image[127-y_coord][127-x_coord] = polarity*127

    # Only caring about a change
    #image[127-y_coord][127-x_coord] = 0
  display.set_data( image )
  display.axes.figure.canvas.draw()
  find_circles(image)
  image = np.zeros( (128, 128), dtype='uint8' ) + 64
  time.sleep(0.00001)
