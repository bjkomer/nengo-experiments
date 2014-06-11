# Control algorithm to drive a car to specific points, not using nengo
import rospy
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Wrench
from std_msgs.msg import String
from sensor_msgs.msg import Image

import math
import time
import numpy as np

destination_x = 0
destination_y = 0

current_x = 0
current_y = 0
current_angle = 0

max_force = 100
max_torque = 50

# minimum values so that the robot does not slow down too much before it reaches
# its targer. Once within a tolerance these will switch off to be 0
min_force = 35
min_torque = 20

distance_tolerance = 0.01
angle_tolerance = 0.01
torque_factor = 15
force_factor = 20

def input_callback( data ):
  global destination_x
  global destination_y
  destination_x = data.force.x
  destination_y = data.torque.z

def odom_callback( data ):
  global current_x
  global current_y
  global current_angle
  current_x = data.pose.pose.position.x
  current_y = data.pose.pose.position.y
  # ROS uses quaternions, so a conversion is needed here
  
  w = data.pose.pose.orientation.w
  x = data.pose.pose.orientation.x
  y = data.pose.pose.orientation.y
  z = data.pose.pose.orientation.z
  
  """
  x = data.pose.pose.orientation.w
  y = data.pose.pose.orientation.x
  z = data.pose.pose.orientation.y
  w = data.pose.pose.orientation.z
  """

  #print( w, x, y, z )

  #heading
  #current_angle = math.atan2( 2.0*y*w - 2.0*x*z, 1.0 - 2.0*y*y - 2.0*z*z )
  
  #attitude
  #current_angle = math.asin( 2*x*y + 2*z*w )

  #bank
  #current_angle = math.atan2( 2.0*x*w - 2.0*y*z, 1.0 - 2.0*x*x - 2.0*z*z )

  quaternion = ( x,y,z,w ) 
  euler = tf.transformations.euler_from_quaternion( quaternion )
  current_angle = euler[2]


rospy.init_node( 'robot', anonymous=True )
    
input_sub = rospy.Subscriber( 'navbot/input', Wrench,
                               input_callback )

odom_sub = rospy.Subscriber( 'navbot/odometry', Odometry,
                              odom_callback )
    
control_pub = rospy.Publisher( 'navbot/control', Wrench )

msg = Wrench()


# calculates the difference between two angles
def angle_difference( a1, a2 ):
  dif  = a1 - a2
  if abs( dif ) > math.pi:
    dif = math.copysign( abs( dif ) - math.pi, dif )
    """
    if a1 < 0:
      a1 += 2*math.pi
    if a2 < 0:
      a2 += 2*math.pi
    dif = a1 - a2
    """
  return dif
  """
  dif = ( a1 + 2 * math.pi ) - ( a2 + 2 * math.pi )
  while dif > math.pi:
    dif -= math.pi
  return dif
  """
while True:
  time.sleep( 0.05 )
  distance = math.sqrt( ( current_x - destination_x ) ** 2 + \
                        ( current_y - destination_y ) ** 2 ) 
  #angle = math.atan2( current_y - destination_y, current_x - destination_x )
  angle = math.atan2( destination_y - current_y, destination_x - current_x )

  #TODO: might need some cases for wrap-around
  #angle_dif = angle_difference( current_angle, angle )
  angle_dif = angle_difference( angle, current_angle )

  if ( abs(angle_dif) > math.pi / 2 ) or distance < distance_tolerance:
    force = 0
  else:
    force = min( max_force, max( distance * \
                                 force_factor * \
                                 ((math.pi / 2) - abs(angle_dif)),
                                 min_force ) )

  if distance < distance_tolerance or abs( angle_dif ) < angle_tolerance:
    torque = 0
  else:
    if abs( angle_dif * torque_factor ) > max_torque:
      torque = math.copysign( max_torque, angle_dif )
    else:
      torque = math.copysign( min( max_torque, 
                                   max( abs( angle_dif * torque_factor ),
                                        min_torque ) ), 
                              angle_dif )

  msg.force.x = force
  msg.torque.z = torque

  control_pub.publish( msg )

  print( "Cur: X: %s, Y: %s, A: %s" % (current_x, current_y, current_angle) )
  print( "Des: X: %s, Y: %s, A: %s" % (destination_x, destination_y, angle) )
  print( "Force: %s, Torque: %s" % (force, torque) )
  #print( distance, angle, current_angle )
