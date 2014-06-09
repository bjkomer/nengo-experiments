# A model where the robot moves to a specific point in local coordinates

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
      # 0 is black
      im = numpy.array( cv_im )

      # Generate a value between -1 and 1 corresponding to how black or white
      # each section of the image is
      #"""
      l = numpy.average(im[:,0:5].ravel()) / 128 - 1
      ml = numpy.average(im[:,3:8].ravel()) / 128 - 1
      mr = numpy.average(im[:,7:12].ravel()) / 128 - 1
      r = numpy.average(im[:,10:15].ravel()) / 128 - 1
      #"""
      """
      l = numpy.average(im[0:5,:].ravel()) / 128 - 1
      ml = numpy.average(im[3:8,:].ravel()) / 128 - 1
      mr = numpy.average(im[7:12,:].ravel()) / 128 - 1
      r = numpy.average(im[10:15,:].ravel()) / 128 - 1
      """
      """
      l = numpy.average(im[0:3,:].ravel()) / 128 - 1
      ml = numpy.average(im[4:7,:].ravel()) / 128 - 1
      mr = numpy.average(im[8:11,:].ravel()) / 128 - 1
      r = numpy.average(im[12:15,:].ravel()) / 128 - 1
      """
      """
      l = numpy.average(im[0:3,:].ravel())
      ml = numpy.average(im[4:7,:].ravel())
      mr = numpy.average(im[8:11,:].ravel())
      r = numpy.average(im[12:15,:].ravel())
      """

      #print( im )
      #print( im[:,0] )
      #print( [l, ml, mr, r] )
      return [l, ml, mr, r]
      #return [-l, -ml, -mr, -r]

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

  odometry = OdometryNode( name='odom', topic='navbot/odometry',
                           attributes = [ True, True, True,
                                          True, True, True,
                                          False, False, False, False,
                                          True, True, True ] )
                          # attributes = [ True, True, True,
                          #                True, True, True,
                          #                True, True, True, True,
                          #                False, False, False ] )
  
  external_input = ExternalInput( 'Control' )

  # represents the way that the robot wants to move, one dimension is force, and
  # the other is torque
  motion = nengo.Ensemble( 50, 2, radius=100 )

  
  # Where the robot thinks it is in local x-y-direction coordinates
  current_location = nengo.Ensemble( 50, 3, radius=10 )
  
  # Where the robot wants to go in local x-y-direction coordinates
  #destination_location = nengo.Ensemble( 500, 3 )
  
  
  # Where the robot thinks it is in local x-y coordinates
  #current_location_simple = nengo.Ensemble( 500, 2 )
  
  # Where the robot wants to go in local x-y coordinates
  destination_location_simple = nengo.Ensemble( 50, 2, radius=10 )

  # Angle that the robot needs to move to face its destination
  required_angle = nengo.Ensemble( 40, 1, radius=math.pi )

  # The absolute distance the robot is from its destination
  required_distance = nengo.Ensemble( 40, 1, radius=10 )

  # Combines the ensembles for current_location and destination_location_simple
  combined_location = nengo.Ensemble( 100, 5, radius=10 )

  def req_dis( x ):
    return math.sqrt( ( x[0] - x[3] ) ** 2 + ( x[1] - x[4] ) ** 2 )

  def req_ang( x ):
    return math.atan( ( x[1] - x[4] ) / ( x[0] - x[3] ) )

  odom_ensemble = nengo.Ensemble( 100, 9, radius=10 )
  nengo.Connection( odometry, odom_ensemble )

  nengo.Connection( odom_ensemble, current_location, transform=[[1,0,0,0,0,0,0,0,0],
                                                           [0,1,0,0,0,0,0,0,0],
                                                           [0,0,0,0,0,0,0,0,1]] )

  nengo.Connection( combined_location, required_distance, function=req_dis )
  nengo.Connection( combined_location, required_angle, function=req_ang )
  nengo.Connection( required_distance, motion, transform=[[20],[0]] )
  nengo.Connection( required_angle, motion, transform=[[0],[10]] )


  nengo.Connection( external_input, destination_location_simple )

  nengo.Connection( current_location, combined_location, transform=[[1,0,0],
                                                                    [0,1,0],
                                                                    [0,0,1],
                                                                    [0,0,0],
                                                                    [0,0,0]] )
  nengo.Connection( destination_location_simple, combined_location, 
                    transform=[[0,0],
                               [0,0],
                               [0,0],
                               [1,0],
                               [0,1]] )

  #nengo.Connection( external_input, robot )
  
  
  nengo.Connection( motion, robot )
  

  probe_odom = nengo.Probe(odom_ensemble, synapse=0.1)
  probe_cont = nengo.Probe(motion, synapse=0.1)

sim = nengo.Simulator( model )

"""
import nengo_gui
jv = nengo_gui.javaviz.View(model)
sim = nengo.Simulator(model)
jv.update_model(sim)
jv.view()
while True:
      sim.run(1)
"""
#"""
before = time.time()
#sim.run(150)
sim.run(50)

after = time.time()
print( "time to run:" )
print( after - before )

import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
plt.plot(sim.trange(), sim.data[probe_odom], lw=2)
plt.title("Odometry")
plt.subplot(2, 1, 2)
plt.plot(sim.trange(), sim.data[probe_cont], lw=2)
plt.title("Vision")
plt.tight_layout()

plt.show()
#"""
