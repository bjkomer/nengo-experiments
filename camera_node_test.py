# nengo controller for robot navigating through a maze

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Wrench
from std_msgs.msg import String

import math
import time
import numpy as np
import nengo
from nengo.utils.ros import VideoCameraNode, OdometryNode, \
                            ForceTorqueNode, RotorcraftAttitudeNode

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

rospy.init_node( 'robot', anonymous=True )

model = nengo.Network( 'Maze Navigation', seed=13 )

with model:

  robot = ForceTorqueNode( name='Mouse', topic='navbot/control',
                           attributes=[ True, False, False, 
                                        False, False, True ] )

  vision = VideoCameraNode( name='Vision', topic='navbot/vision/image' )

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

  temp = nengo.Ensemble( 500, 3 )
  
  #vis_ens = nengo.Ensemble( 16*16*8, 16*16 )
  vis_ens = nengo.Ensemble( 16*16, 16*16 )

  nengo.Connection( external_input, robot )
  
  nengo.Connection( odometry, temp )
  
  nengo.Connection( vision, vis_ens )
  

  probe_odom = nengo.Probe(temp, synapse=0.1)
  probe_vis = nengo.Probe(vis_ens, synapse=0.1)

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
