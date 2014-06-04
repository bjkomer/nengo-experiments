# A model where the robot is attracted to specific colours while trying to avoid
# other colours

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Wrench
from std_msgs.msg import String
from sensor_msgs.msg import Image

import math
import time
import numpy as np
import nengo
from nengo.utils.ros import VideoCameraNode, OdometryNode, \
                            ForceTorqueNode, RotorcraftAttitudeNode, RosSubNode

CONTROL_PERIOD = 30

class ExternalInput( nengo.objects.Node ):
  """
  This node represents external input given to the robot.
  This is controlled externally using ROS messages.
  Messages can be published through command line with:
    rostopic pub /navbot/input geometry_msgs/Wrench "{force:[<x>,<y>,<z>], torque:[<x>,<y>,<z>]}"
  """
  
  def __init__( self, name ):

    self.force = 0
    self.torque = 0
    
    self.dest_sub = rospy.Subscriber( 'navbot/input', Wrench,
                                      self.callback )

    super( ExternalInput, self ).__init__( label=name, output=self.tick,
                                        size_in=0, size_out=2 )
  
  def callback( self, data ):
    self.force = data.force.x
    self.torque = data.torque.z
  
  def tick( self, t):
    
    return [ self.force,  self.torque ]

class ColourDetectionNode( RosSubNode ):
  """
  This node outputs how black or white sections of an image are
  """
  
  #TODO: add optional weights and transform function for output
  def __init__( self, name, topic ):
    """
    Parameters
    ----------
    name : str
        An arbitrary name for the object
    topic : str
        The name of the ROS topic that is being subscribed to
    """

    from cv_bridge import CvBridge, CvBridgeError
    import numpy

    self.bridge = CvBridge()

    #self.dimensions = 16 * 16
    self.dimensions = 4

    def fn( data ):
      rval = [0] * self.dimensions
      #cv_im = self.bridge.imgmsg_to_cv( data, "rgba8" )
      cv_im = self.bridge.imgmsg_to_cv( data, "mono8" )
      #TODO: make sure mono conversion is correct
      im = numpy.array( cv_im )

      # Generate a value between -1 and 1 corresponding to how black or white
      # each section of the image is
      
      l = numpy.average(im[0:3,:].ravel()) / 128 - 1
      ml = numpy.average(im[4:7,:].ravel()) / 128 - 1
      mr = numpy.average(im[8:11,:].ravel()) / 128 - 1
      r = numpy.average(im[12:15,:].ravel()) / 128 - 1
      """
      l = numpy.average(im[0:3,:].ravel())
      ml = numpy.average(im[4:7,:].ravel())
      mr = numpy.average(im[8:11,:].ravel())
      r = numpy.average(im[12:15,:].ravel())
      """

      #return [l, ml, mr, r]
      return [-l, -ml, -mr, -r]

    self.fn = fn

    super( ColourDetectionNode, self ).__init__( name=name, topic=topic,
                                             dimensions=self.dimensions,
                                             msg_type=Image, trans_fnc=self.fn )

rospy.init_node( 'robot', anonymous=True )

model = nengo.Network( 'Colour Preference', seed=13 )

with model:

  robot = ForceTorqueNode( name='Mouse', topic='navbot/control',
                           attributes=[ True, False, False, 
                                        False, False, True ] )

  vision = ColourDetectionNode( name='Vision', topic='navbot/vision/image' )

  odometry = OdometryNode( name='odom', topic='navbot/odometry',
                           attributes = [ False, False, False,
                                          True, True, True,
                                          False, False, False, False,
                                          False, False, False ] )
                          # attributes = [ True, True, True,
                          #                True, True, True,
                          #                True, True, True, True,
                          #                False, False, False ] )
  
  external_input = ExternalInput( 'Control' )

  # represents the way that the robot wants to move, one dimension is force, and
  # the other is torque
  motion = nengo.Ensemble( 500, 2, radius=100 )
  vis_temp = nengo.Ensemble( 500, 4 )

  temp = nengo.Ensemble( 500, 3 )

  nengo.Connection( external_input, robot )
  
  F = 100.0
  T = 50.0
  nengo.Connection( vision, motion, transform=[ [0.1*F,0.5*F,0.5*F,0.1*F], 
                                                [0.4*T,0.2*T,0.2*T,0.4*T] ] )
  ##nengo.Connection( vision, motion, transform=[ [0,0,0,0], 
  ##                                              [0,0,0,0] ] )
  #nengo.Connection( vision, motion ) 
  nengo.Connection( vision, vis_temp ) 
  
  nengo.Connection( motion, robot )
  
  nengo.Connection( odometry, temp )
  

  probe_odom = nengo.Probe(temp, synapse=0.1)
  probe_vis = nengo.Probe(motion, synapse=0.1)
  #probe_vis = nengo.Probe(vis_temp, synapse=0.1)

sim = nengo.Simulator( model )

before = time.time()
#sim.run(100)
sim.run(2)

after = time.time()
print( "time to run:" )
print( after - before )

import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
plt.plot(sim.trange(), sim.data[probe_odom], lw=2)
plt.title("Odometry")
plt.subplot(2, 1, 2)
plt.plot(sim.trange(), sim.data[probe_vis], lw=2)
plt.title("Vision")
plt.tight_layout()

plt.show()
