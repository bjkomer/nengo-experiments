import numpy as np
import rospy
import time

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from collections import deque

class KalmanFilter():

  def __init__( self, dt=0.01 ):

    self.dt = dt # this needs to align with the frequency of ROS messages

    # initial uncertainty
    self.P = np.matrix([[1000.0,0.,0.,0.],
                        [0.,1000.0,0.,0.],
                        [0.,0.,1000.0,0.],
                        [0.,0.,0.,1000.0]])
    
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

    # Initial position and velocity
    self.X = np.matrix([[0],[0],[0],[0]])
    
    # External motion
    self.U = np.matrix([[0],[0],[0],[0]])
    
    self.measurements = deque()
    rospy.init_node( 'kalman', anonymous=True )
    
    # Used to read the robot pose from Morse
    self.sub = rospy.Subscriber( 'navbot/pose', PoseStamped, self.pose_callback )
    
    # Used to send the Estimated position and velocity to the gui to display
    self.pub = rospy.Publisher( 'navbot/estimate', Quaternion )
    #self.pub = rospy.Publisher( 'navbot/filter', Quaternion )
    self.estimate = Quaternion() #TEMP just using quaternion because it is 4D

  def pose_callback( self, data ):
    self.measurements.append( [data.pose.position.x, data.pose.position.y] )

  def set_measurements( self, measurements ):
    for m in measurements:
      self.measurements.append( [m[0], m[1]] )

  def run_ros( self ):

    #TODO: make this run faster, it goes too slow for real-time control
    while not rospy.is_shutdown():
      if self.measurements:
        # prediction
        self.X = (self.F * self.X) + self.U
        self.P = self.F * self.P * self.F.T

        #measurement update
        Z = np.matrix( self.measurements.popleft() )
        Y = Z.T - (self.H * self.X)
        S = self.H * self.P * self.H.T + self.R
        K = self.P * self.H.T * S.I
        self.X = self.X + (K * Y)
        self.P = (self.I - (K * self.H)) * self.P

        # Send to the gui
        self.estimate.x = self.X[0]
        self.estimate.y = self.X[1]
        self.estimate.z = self.X[2]
        self.estimate.w = self.X[3]
        self.pub.publish( self.estimate )

      else:
        time.sleep(0.001)

  def run( self ):
    for m in self.measurements:
      # prediction
      self.X = (self.F * self.X) + self.U
      self.P = self.F * self.P * self.F.T

      #measurement update
      Z = np.matrix( m )
      Y = Z.T - (self.H * self.X)
      S = self.H * self.P * self.H.T + self.R
      K = self.P * self.H.T * S.I
      self.X = self.X + (K * Y)
      self.P = (self.I - (K * self.H)) * self.P

      print( m )
      print( self.P )
      print( self.X )

ms = [[1.0, 0.0],
      [1.5, 1.0],
      [2.0, 2.0],
      [2.5, 3.0],
      [3.0, 4.0],
      [3.5, 5.0],
      [4.0, 6.0],
      [4.5, 7.0],
      [5.0, 8.0],
      [5.5, 9.0],
      [6.0, 10.0],
     ]

if __name__ == "__main__":
  k = KalmanFilter( dt = 1.0 ) # 1 / f
  k.set_measurements( ms )
  k.run()
  #k = KalmanFilter( dt = 1.0/10.0 ) # 1 / f
  #k.run_ros()
