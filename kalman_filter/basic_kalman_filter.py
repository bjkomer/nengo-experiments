import numpy as np
import rospy
import time

from geometry_msgs.msg import Pose
from collections import deque

class KalmanFilter():

  def __init__( self, dt=0.01 ):

    self.dt = dt # this needs to align with the frequency of ROS messages

    # initial uncertainty
    self.P = np.matrix([[0.,0.,0.,0.],
                        [0.,0.,0.,0.],
                        [0.,0.,1000.,0.],
                        [0.,0.,0.,1000.]])
    
    # next state function
    self.F = np.matrix([[1.,0.,dt,0.],
                        [0.,1.,0.,dt],
                        [0.,0.,1.,0.],
                        [0.,0.,0.,1.]])
    
    # measurement function
    self.H = np.matrix([[1., 0, 0., 0],
                        [0., 1., 0, 0]])
    
    # measurement uncertainty
    self.R = np.matrix([[.1, 0.], 
                        [0., .1]])
    
    # identity matrix
    self.I = np.matrix([[1.,0.,0.,0.],
                        [0.,1.,0.,0.],
                        [0.,0.,1.,0.],
                        [0.,0.,0.,1.]])

    initial_xy = [0, 0]
    
    self.measurements = deque()
    rospy.init_node( 'kalman', anonymous=True )
    self.sub = rospy.Subscriber( 'navbot/pose', Pose, pose_callback )

  def pose_callback( self, data ):
    self.measurements.append( [data.position.x, data.position.y] )


  def run( self ):

    while not rospy.is_shutdown():
      if measurements:
        # prediction
        x = (F * x) + u
        P = F * P * F.transpose()

        #measurement update
        Z = np.matrix( self.measurements.popleft() )
        y = Z.transpose() - (H * x)
        S = H * P * H.transpose() + R
        K = P * H.transpose() * S.inverse()
        x = x + (K * Y)
        P = (I - (K * H)) * P
      else:
        time.sleep(0.001)

if __name__ == "__main__":
  k = KalmanFilter( dt = 1.0/30.0 ) # 1 / f
  k.run()
