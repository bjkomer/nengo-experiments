# Morse scenario to set up space for driving
from morse.builder import *

mover = ATRV( 'target' )
#mover.translate(x=3.0)
mover.translate(x=2.0, z=3.0)
mover.properties( Object=True, Graspable=False, Label="Target" )

keyboard = Keyboard()
keyboard.properties( Speed=4.0 )
#mover.append( keyboard ) #TEMP

robot = ATRV()
robot.translate(z=3.0)

control = ForceTorque()
robot.append( control )

odom = Odometry()
robot.append( odom )

robot.append( keyboard ) #TEMP
#motionxyw = MotionXYW()
#robot.append( motionxyw )

#collision = Collision()
#robot.append( collision )

"""
video = SemanticCamera()
video.translate(x=0.4, z=1.0)
video.frequency(frequency=10)
robot.append(video)
video.properties(Vertical_Flip=False)
"""

video = VideoCamera()
video.properties(cam_far=900, cam_height=128, cam_width=128, Vertical_Flip=True)
video.translate(x=0.4, z=1.0)
video.frequency(frequency=10)
robot.append(video)
video.properties(Vertical_Flip=False)

control.add_interface( 'ros', topic="/navbot/control" )
#motionxyw.add_interface( 'ros', topic="/navbot/velocity_control" )
odom.add_interface( 'ros', topic="/navbot/odometry" )
video.add_interface( 'ros', topic="/navbot/vision" )
#collision.add_interface( 'ros', topic="/navbot/collision" )

#env = Environment('land-1/trees')
env = Environment( '/home/bjkomer/morse/data/environments/open_area.blend' )
env.place_camera([10.0, -10.0, 10.0])
env.aim_camera([1.0470, 0, 0.7854])

env.select_display_camera(video)
