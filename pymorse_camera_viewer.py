import pymorse
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import deque
import numpy as np

class CameraViewer():

  def __init__( self, root='navbot' ):

    self.im_data = deque()

    self.im_fig = plt.figure( 1 )
    self.im_ax = self.im_fig.add_subplot(111)
    self.im_ax.set_title("DVS Image")
    self.im_im = self.im_ax.imshow( np.zeros( ( 256, 256 ),dtype='uint8' ), 
                                    cmap=plt.cm.gray, vmin=0, vmax=255 ) # Blank starting image
    self.im_fig.show()
    self.im_im.axes.figure.canvas.draw()

  # Just view greyscale dvs images for now
  def im_callback( self, data ):

    self.im_data.append( 
      np.asarray( data['image'] ).reshape((data['height'],data['width'])) )
  
    #cv_im = self.bridge.imgmsg_to_cv( data, "mono8" ) # Convert Image from ROS Message to greyscale CV Image
    #im = np.asarray( cv_im ) # Convert from CV image to numpy array
    #self.im_data.append( im )

  def run( self ):

    with pymorse.Morse() as sim:
      
      sim.robot.dvs.subscribe( self.im_callback )
      
      while True:
        if self.im_data:
          im = self.im_data.popleft()
          self.im_im.set_cmap( 'gray' ) # This doesn't seem to do anything
          self.im_im.set_data( im )
          self.im_im.axes.figure.canvas.draw()

def main():

  viewer = CameraViewer()
  viewer.run()

if __name__ == '__main__':
  main()
