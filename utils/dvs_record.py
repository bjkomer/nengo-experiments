# records DVS images from VREP, saves them to a pickle file

import cPickle as pickle
import vrep
import numpy as np
import time
import ctypes
import signal
import matplotlib.pyplot as plt
import sys

if len(sys.argv) > 1:
  filename = str(sys.argv[1])
else:
  print("Need to specify file name")
  exit()

empty_str=""
raw_bytes = (ctypes.c_ubyte * len(empty_str)).from_buffer_copy(empty_str) 

fig = plt.figure(1)
ax = fig.add_subplot(131)
ax.set_title("DVS Image")
image = np.zeros( (128, 128), dtype='uint8' )
display = ax.imshow( image, cmap=plt.cm.gray, vmin=0, vmax = 127 )

# List of all images
video = []

fig.show()
display.axes.figure.canvas.draw()

cid = vrep.simxStart('127.0.0.1',20002,True,True,5000,5)

if cid != -1:
  print ('Connected to V-REP remote API server, client id: %s' % cid)
  vrep.simxStartSimulation( cid, vrep.simx_opmode_oneshot )
else:
  print ('Failed connecting to V-REP remote API server')
  exit(1)

def save_video( signal, frame ):
  pickle.dump( video, open(filename, "wb") )
  exit()

# Save video and exit when a keyboard interrupt is found
signal.signal(signal.SIGINT, save_video)

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
    #print( polarity, x_coord, y_coord, timestamp )
    
    # Distinguish between different polarities
    image[127-y_coord][127-x_coord] = polarity*127

    # Only caring about a change
    #image[127-y_coord][127-x_coord] = 0
  display.set_data( image )
  display.axes.figure.canvas.draw()
  #image = np.ceil((image + 95) / 2.5).astype('uint8')
  #image = np.ceil((image + 95 + 64*6) / (2.5+6)).astype('uint8')
  video.append( image.copy() )
  image = np.zeros( (128, 128), dtype='uint8' ) + 64
  time.sleep(0.00001)
