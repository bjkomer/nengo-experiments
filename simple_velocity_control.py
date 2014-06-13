# Control algorithm to drive a car to specific points with velocity, not using nengo
import rospy
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Wrench, Twist
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

max_linear = 4.0
max_angular = 1.0

# minimum values so that the robot does not slow down too much before it reaches
# its targer. Once within a tolerance these will switch off to be 0
min_linear = 0.1 #0
min_angular = 0.1 #0

distance_tolerance = 0.05
angle_tolerance = 0.05
linear_factor = 0.5
angular_factor = 0.5

TURN_THEN_DRIVE = True

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
    
velocity_control_pub = rospy.Publisher( 'navbot/velocity_control', Twist )

msg = Twist()


# calculates the difference between two angles
def angle_difference( a1, a2 ):

  if abs(a1) + abs(a2) > math.pi:
    if a1 < 0 and a2 > 0:
      dif  = a1 - a2 + 2 * math.pi
    elif a2 < 0 and a1 > 0:
      dif  = a1 - a2 - 2 * math.pi
    else:
      dif = a1 - a2
  else:
    dif = a1 - a2

  """
  dif  = a1 - a2
  if a1 < -1 * math.pi / 2:
    if a2 > math.pi / 2:
      a1 += 2 * math.pi
  if abs( dif ) > math.pi:
    dif = math.copysign( abs( dif ) - math.pi, dif )
  """
  return dif

while True:
  time.sleep( 0.05 )
  distance = math.sqrt( ( current_x - destination_x ) ** 2 + \
                        ( current_y - destination_y ) ** 2 ) 
  angle = math.atan2( destination_y - current_y, destination_x - current_x )

  #TODO: might need some cases for wrap-around
  angle_dif = angle_difference( angle, current_angle )

  # Controller turns until it faces the right direction, and then drives
  if TURN_THEN_DRIVE:

    if abs( angle_dif ) > angle_tolerance:
      angular = math.copysign( min( max_angular, 
                                    max( abs( angle_dif * angular_factor ),
                                         min_angular ) ), 
                              angle_dif )
      linear = 0
    elif distance > distance_tolerance:
      angular = 0
      linear = min( max_linear, max( distance * \
                                     linear_factor * \
                                     ((math.pi / 2) - abs(angle_dif)),
                                     min_linear ) )
    else:
      linear = 0
      angular = 0


  else:

    if ( abs(angle_dif) > math.pi / 2 ) or distance < distance_tolerance:
      linear = 0
    else:
      linear = min( max_linear, max( distance * \
                                     linear_factor * \
                                     ((math.pi / 2) - abs(angle_dif)),
                                     min_linear ) )

    if distance < distance_tolerance or abs( angle_dif ) < angle_tolerance:
      angular = 0
    else:
      if abs( angle_dif * angular_factor ) > max_angular:
        angular = math.copysign( max_angular, angle_dif )
      else:
        angular = math.copysign( min( max_angular, 
                                      max( abs( angle_dif * angular_factor ),
                                           min_angular ) ), 
                                angle_dif )

  msg.linear.x = linear
  msg.angular.z = angular

  velocity_control_pub.publish( msg )

  print( "Cur: X: %.5f, Y: %.5f, A: %.5f" % (current_x, current_y, current_angle) )
  print( "Des: X: %.5f, Y: %.5f, A: %.5f" % (destination_x, destination_y, angle) )
  print( "Linear: %.5f, Angular: %.5f" % (linear, angular) )
