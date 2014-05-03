# nengo controller for robot navigating through a maze

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Wrench

import math
import time
import numpy as np
import nengo

CONTROL_PERIOD = 30

class ExternalInput( nengo.objects.Node ):
  """
  This node represents external input given to the robot.
  This is controlled externally using ROS messages.
  Messages can be published through command line with:
    rostopic pub /navbot/input geometry_msgs/Wrench "{force:[<x>,<y>,<z>], torque:[<x>,<y>,<z>]}"
  """
  
  def __init__( self, name ):

    #nengo.objects.Node.__init__( self, name, self.tick, dimensions=1 )

    self.dest_sub = None
    self.deepcopied = False

    self.force = 0
    self.torque = 0
    

    self.callback = self.dest_callback

    #nengo.objects.Node.__init__( self, label=name, output=self.tick, size_in=1 )
    super( ExternalInput, self ).__init__( label=name, output=self.tick,
                                        size_in=0, size_out=2 )
  
  def dest_callback( self, data ):
    self.force = data.force.x
    self.torque = data.torque.z
 
  # Workaround to builder issue
  def __getattribute__( self, name ):
    if name == '__class__':
      return nengo.objects.Node
    else:
      return super(ExternalInput, self).__getattribute__(name)
    
  def tick( self, t):
    
    #gross hack to get around deepcopy issue
    if self.dest_sub is None and self.deepcopied == True:
      # This cannot be defined in init due to errors caused by deepcopy
      self.dest_sub = rospy.Subscriber( 'navbot/input', Wrench,
                                        self.callback )

    if self.deepcopied == False:
      self.deepcopied = True
      
    return [ self.force,  self.torque ]

class Robot( nengo.objects.Node ):
  def __init__( self, name ):

    self.sensor_data = { 'x' :0, 'y' :0, 'z' :0,
                         'vx':0, 'vy':0, 'vz':0,
                         'wx':0, 'wy':0, 'wz':0,
                         'roll':0, 'pitch':0, 'yaw':0 }

    self.odom_sub = None
    self.cont_pub = None
    self.control_signal = Wrench()
    
    # Linear
    # Position

    self.x = 0
    self.y = 0
    self.z = 0

    # Velocity

    self.vx = 0
    self.vy = 0
    self.vz = 0
    
    # Angular
    # Position

    self.tx = 0
    self.ty = 0
    self.tz = 0
    
    # Velocity

    self.wx = 0
    self.wy = 0
    self.wz = 0
    
    self.counter = 0 # Used so that computations don't need to occur on every tick
    
    self.deepcopied = False


    self.callback = self.odom_callback

    super( Robot, self).__init__( label=name, output=self.tick,
                                       size_in=2, size_out=12 )

  # Workaround to issue with builder
  def __getattribute__( self, name ):
    if name == '__class__':
      return nengo.objects.Node
    else:
      return super(Robot, self).__getattribute__(name)
    
  # Callback for morse, when odometry data is available
  def odom_callback( self, data ):
    self.x = data.pose.pose.position.x
    self.y = data.pose.pose.position.y
    self.z = data.pose.pose.position.z

    self.vx = data.twist.twist.linear.x
    self.vy = data.twist.twist.linear.y
    self.vz = data.twist.twist.linear.z

    # FIXME: ROS outputs orientation in quaternions, need to account for this
    self.tx = data.pose.pose.orientation.x
    self.ty = data.pose.pose.orientation.y
    self.tz = data.pose.pose.orientation.z

    self.wx = data.twist.twist.angular.x
    self.wy = data.twist.twist.angular.y
    self.wz = data.twist.twist.angular.z
  
  def tick( self, t, values ):
    
    # gross hack to get around deepcopy issue
    if self.odom_sub is None and self.deepcopied == True:
      # This cannot be defined in init due to errors caused by deepcopy
      self.odom_sub = rospy.Subscriber( 'navbot/odometry', Odometry,
                                        self.callback )
      
      self.cont_pub = rospy.Publisher( 'navbot/control', Wrench )
    
    if self.deepcopied == False:
      self.deepcopied = True

    self.counter += 1
    if self.counter % CONTROL_PERIOD == 0:
      force = values[0]
      torque = values[1]
      if math.isnan( force ):
        force = 0
      if math.isnan( torque ):
        torque = 0
      
      # Send Control Signal
      self.control_signal.force.x = force
      self.control_signal.torque.z = torque
      self.cont_pub.publish( self.control_signal )

    return [ self.x,  self.y,  self.z,
             self.vx, self.vy, self.vz,
             self.tx, self.ty, self.tz,
             self.wx, self.wy, self.wz ]

rospy.init_node( 'robot', anonymous=True )

model = nengo.Network( 'Maze Navigation', seed=13 )

with model:

  robot = Robot( 'Mouse' )

  external_input = ExternalInput( 'Control' )

  nengo.Connection( external_input, robot )

sim = nengo.Simulator( model )

before = time.time()
sim.run(1000)

after = time.time()
print( "time to run:" )
print( after - before )
