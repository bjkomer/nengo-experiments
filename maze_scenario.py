# Morse scenario to set up maze
from morse.builder import *

mover = ATRV( 'target' )
#mover.translate(x=3.0)
mover.translate(x=2.0, z=3.0)
mover.properties( Object=True, Graspable=False, Label="Target" )

keyboard = Keyboard()
keyboard.properties( Speed=4.0 )
mover.append( keyboard )

robot = ATRV()
robot.translate(z=3.0)

control = ForceTorque()
robot.append( control )

odom = Odometry()
robot.append( odom )

#collision = Collision()
#robot.append( collision )

video = SemanticCamera()
video.translate(x=0.4, z=1.0)
video.frequency(frequency=10)
robot.append(video)
video.properties(Vertical_Flip=False)

control.add_interface( 'ros', topic="/navbot/control" )
odom.add_interface( 'ros', topic="/navbot/odometry" )
video.add_interface( 'ros', topic="/navbot/semantic" )
#collision.add_interface( 'ros', topic="/navbot/collision" )

#env = Environment('land-1/trees')
env = Environment( '/home/bjkomer/morse/data/robots/circuit_t_maze.blend' )
env.place_camera([10.0, -10.0, 10.0])
env.aim_camera([1.0470, 0, 0.7854])

env.select_display_camera(video)
