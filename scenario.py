from morse.builder import *

#gandalf = Gandalf('gandalf')
gandalf = Gandalf('body')
#gandalf.translate(x=-6.0, z=0.2)

#driver = ATRV()

#driver_pose = Pose()
#driver.append( driver_pose )

keyboard = Keyboard()
keyboard.properties(Speed=3.0)
gandalf.append(keyboard)

#odom = Odometry()
#odom.level( 'multiple' )
odom = LinkageOdometry()
gandalf.append( odom )

"""
semantic = SemanticCamera()
semantic.translate(x=0.3, z=-0.05)
semantic.rotate(x=0.2)
semantic.frequency(frequency=30)
gandalf.append(semantic)
semantic.properties(Vertical_Flip=False)
"""
"""
video = VideoCamera()
video.translate(x=0.3, z=-0.05)
video.rotate(x=0.2)
video.frequency(frequency=10)
gandalf.append(video)
video.properties(Vertical_Flip=False)
"""
"""
depth = DepthCamera()
depth.translate(x=0.3, z=-0.05)
depth.rotate(x=0.2)
depth.frequency(frequency=10)
gandalf.append(depth)
depth.properties(Vertical_Flip=False)
"""

dvs = DVSCamera()
dvs.translate(x=3, z=2.2)
dvs.rotate(y=0.4)
dvs.frequency(frequency=10)
gandalf.append(dvs)
dvs.properties(Vertical_Flip=False)


force = ForceTorque()
#force.level( 'local' )
force.level( 'multiple' )
gandalf.append( force )
force.add_interface( 'socket' )

odom.add_interface( 'socket' )

#dvs.add_interface( 'socket' )
dvs.add_interface( 'ros', topic="/navbot/camera" )
#depth.add_interface( 'ros', topic="/navbot/camera" )
#video.add_interface( 'ros', topic="/navbot/camera" )

env = Environment('land-1/trees')
env.place_camera([10.0, -10.0, 10.0])
env.aim_camera([1.0470, 0, 0.7854])
#env.select_display_camera(semantic)
env.select_display_camera(dvs)
