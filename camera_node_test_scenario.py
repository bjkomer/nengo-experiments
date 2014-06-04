# Morse scenario to set up maze
from morse.builder import *

robot = ATRV()
robot.translate(z=3.0)

control = ForceTorque()
robot.append( control )

keyboard = Keyboard()
keyboard.properties( Speed=4.0 )
robot.append( keyboard )

odom = Odometry()
robot.append( odom )

#collision = Collision()
#robot.append( collision )

video = VideoCamera()
video.properties(cam_far=900, cam_height=16, cam_width=16, Vertical_Flip=True)
video.translate(x=0.4, z=1.0)
video.frequency(frequency=10)
robot.append(video)
video.properties(Vertical_Flip=False)

control.add_interface( 'ros', topic="/navbot/control" )
odom.add_interface( 'ros', topic="/navbot/odometry" )
video.add_interface( 'ros', topic="/navbot/vision" )
#collision.add_interface( 'ros', topic="/navbot/collision" )

#env = Environment('land-1/trees')
env = Environment( '/home/bjkomer/morse/data/robots/circuit_t_maze.blend' )
env.place_camera([10.0, -10.0, 10.0])
env.aim_camera([1.0470, 0, 0.7854])

env.select_display_camera(video)
