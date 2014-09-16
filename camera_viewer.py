import matplotlib.pyplot as plt
from matplotlib import cm
from cv_bridge import CvBridge, CvBridgeError
import cv
from collections import deque
import rospy
from sensor_msgs.msg import Image
import numpy as np

class CameraViewer():

  def __init__( self, root='navbot' ):

    self.root = root
    self.im_data = deque()
    self.bridge = CvBridge() # For converting ROS images into a readable format

    self.im_fig = plt.figure( 1 )
    self.im_ax = self.im_fig.add_subplot(111)
    self.im_ax.set_title("DVS Image")
    self.im_im = self.im_ax.imshow( np.zeros( ( 256, 256 ),dtype='uint8' ), cmap=plt.cm.gray, vmin=0, vmax=255 ) # Blank starting image
    #self.im_im = self.im_ax.imshow( np.zeros( ( 256, 256 ),dtype='float32' ), cmap=plt.cm.gray ) # Blank starting image
    #self.im_im = self.im_ax.imshow( np.ones( ( 256, 256 ),dtype='uint8' ) *127 ) # Blank starting image
    #self.im_im = self.im_ax.imshow( np.zeros( ( 256, 256, 4 ) ) ) # TEMP
    self.im_fig.show()
    self.im_im.axes.figure.canvas.draw()

  # Just view greyscale dvs images for now
  def im_callback( self, data ):

    cv_im = self.bridge.imgmsg_to_cv2( data, "mono8" ) # Convert Image from ROS Message to greyscale CV Image
    #cv_im = self.bridge.imgmsg_to_cv( data, "bgra8" )
    #cv_im = self.bridge.imgmsg_to_cv( data, "rgba8" )
    im = np.asarray( cv_im )[:,:,0] # Convert from CV image to numpy array
    #im = np.asarray( cv_im, dtype='float32' ) / 256
    self.im_data.append( im )

  def run( self ):

    rospy.init_node('camera_viewer', anonymous=True)
    
    sub_im = rospy.Subscriber( self.root + '/camera/image', Image, self.im_callback)

    while not rospy.is_shutdown():
      if self.im_data:
        im = self.im_data.popleft()
        self.im_im.set_cmap( 'gray' ) # This doesn't seem to do anything
        self.im_im.set_data( im )
        #self.im_ax.imshow( im, cmap=plt.cm.gray )
        self.im_im.axes.figure.canvas.draw()

def main():

  viewer = CameraViewer( root='navbot' )
  viewer.run()

if __name__ == '__main__':
  main()
