# Morse scenario to set up maze
from morse.builder import *

robot = ATRV()

control = ForceTorque()
robot.append( control )

odom = Odometry()
robot.append( odom )

video = VideoCamera()
video.translate(x=0.4, z=1.0)
video.frequency(frequency=10)
robot.append(video)
video.properties(Vertical_Flip=False)

control.add_interface( 'ros', topic="/navbot/control" )
odom.add_interface( 'ros', topic="/navbot/odom" )
video.add_interface( 'ros', topic="/navbot/camera" )

env = Environment('land-1/trees')
env.place_camera([10.0, -10.0, 10.0])
env.aim_camera([1.0470, 0, 0.7854])

env.select_display_camera(video)
