from morse.builder import *
import math

robot = ATRV()

keyboard = Keyboard()
keyboard.properties( Speed=3.0 )
robot.append( keyboard )

pose = Pose()
pose.frequency( frequency=30 )
#pose.alter( 'Noise', pos_std = 0.3, rot_std=math.radians(10)) # Adding noise
robot.append(pose)

pose.add_interface( 'ros', topic='navbot/pose' )

env = Environment( 'outdoors' )
env.place_camera( [ 10.0, -10.0, 10.0 ] )
env.aim_camera( [ 1.0470, 0, 0.7854 ] )
