from morse.builder import *
import math

robot = ATRV()

keyboard = Keyboard()
keyboard.properties( Speed=3.0 )
robot.append( keyboard )

pose = Pose()
pose.frequency( frequency=10 )
#pose.alter( 'Noise', pos_std = 0.3, rot_std=math.radians(10)) # Adding noise
robot.append(pose)

odom = Odometry()
odom.frequency( frequency=10 )
robot.append(odom)

pose.add_interface( 'ros', topic='navbot/pose' )
odom.add_interface( 'ros', topic='navbot/odometry' )

env = Environment( 'outdoors' )
env.place_camera( [ 10.0, -10.0, 10.0 ] )
env.aim_camera( [ 1.0470, 0, 0.7854 ] )
