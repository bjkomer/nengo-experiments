from morse.builder import *

gandalf = Gandalf('gandalf')
#gandalf.translate(x=-6.0, z=0.2)

#driver = ATRV()

#driver_pose = Pose()
#driver.append( driver_pose )

keyboard = Keyboard()
keyboard.properties(Speed=3.0)
gandalf.append(keyboard)

odom = Odometry()
gandalf.append( odom )

semantic = SemanticCamera()
semantic.translate(x=0.3, z=-0.05)
semantic.rotate(x=0.2)
semantic.frequency(frequency=30)
gandalf.append(semantic)
semantic.properties(Vertical_Flip=False)

force = ForceTorque()
force.level( 'local' )
gandalf.append( force )
force.add_interface( 'socket' )

odom.add_interface( 'socket' )

env = Environment('land-1/trees')
env.place_camera([10.0, -10.0, 10.0])
env.aim_camera([1.0470, 0, 0.7854])
env.select_display_camera(semantic)
