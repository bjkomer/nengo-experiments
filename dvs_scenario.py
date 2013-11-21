# A simple scenario for driving around with a dvs camera
from morse.builder import *

robot = ATRV()

keyboard = Keyboard()
keyboard.properties(Speed=3.0)
robot.append(keyboard)

dvs = DVSCamera()
dvs.translate(x=1, z=1)
dvs.rotate(y=0.1)
dvs.frequency(frequency=15)
robot.append(dvs)
dvs.properties(cam_far=900, Vertical_Flip=True)

dvs.add_interface( 'ros', topic="/navbot/camera" )
#dvs.add_interface( 'socket' )

env = Environment('land-1/trees')
#env = Environment('/home/komer/Downloads/BUERAKI_v0.2.0/levels/terrains/vastland')
env.place_camera([10.0, -10.0, 10.0])
env.aim_camera([1.0470, 0, 0.7854])
#env.select_display_camera(dvs)
