from Tkinter import *
import rospy
import tf
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry

class ControlGUI():

  def __init__( self ):
    rospy.init_node( 'destination', anonymous=True )

    self.top = Tk()
    self.top.wm_title('Kalman Filter')
    
    self.op = "%.2f" # Odometry display precision

    #####################
    # Destination Input #
    #####################
    
    self.dest_pub = rospy.Publisher( 'navbot/destination', Quaternion )
    self.msg = Quaternion()

    self.x_var=StringVar()
    self.y_var=StringVar()
    self.z_var=StringVar()
    self.yaw_var=StringVar()

    # Labels
    x_label = Label( self.top, text="X Location" )
    y_label = Label( self.top, text="Y Location" )
    z_label = Label( self.top, text="Z Location" )
    yaw_label = Label( self.top, text="Yaw Direction" )

    x_label.grid( row=0, column=0 )
    y_label.grid( row=1, column=0 )
    z_label.grid( row=2, column=0 )
    yaw_label.grid( row=3, column=0 )

    #Text Fields
    x_entry = Entry( self.top, bd=5, textvariable=self.x_var )
    y_entry = Entry( self.top, bd=5, textvariable=self.y_var )
    z_entry = Entry( self.top, bd=5, textvariable=self.z_var )
    yaw_entry = Entry( self.top, bd=5, textvariable=self.yaw_var )

    x_entry.grid( row=0, column=1 )
    y_entry.grid( row=1, column=1 )
    z_entry.grid( row=2, column=1 )
    yaw_entry.grid( row=3, column=1 )

    #Send Button
    send = Button( self.top, text="send", command=self.send_callback )
    send.grid( row=4 )

    ###################
    # Odometry Output #
    ###################

    self.odom_sub = rospy.Subscriber( 'navbot/odometry', Odometry, self.odom_callback )
    
    self.odom_vars = []
    
    for i in xrange(12):
      self.odom_vars.append( StringVar() )
    
    # Labels

    odom_labels = []

    odom_labels.append( Label( self.top, text="Linear X Position" ) )
    odom_labels.append( Label( self.top, text="Linear Y Position" ) )
    odom_labels.append( Label( self.top, text="Linear Z Position" ) )
    
    odom_labels.append( Label( self.top, text="Linear X Velocity" ) )
    odom_labels.append( Label( self.top, text="Linear Y Velocity" ) )
    odom_labels.append( Label( self.top, text="Linear Z Velocity" ) )
    
    odom_labels.append( Label( self.top, text="Angular X Position" ) )
    odom_labels.append( Label( self.top, text="Angular Y Position" ) )
    odom_labels.append( Label( self.top, text="Angular Z Position" ) )
    
    odom_labels.append( Label( self.top, text="Angular X Velocity" ) )
    odom_labels.append( Label( self.top, text="Angular Y Velocity" ) )
    odom_labels.append( Label( self.top, text="Angular Z Velocity" ) )

    # Display Fields

    odom_values = []
    for i in xrange(12):
      odom_values.append( Label( self.top, textvariable=self.odom_vars[i] ) )
    
    label = 0
    for c in [0,2]:
      for r in xrange(5,11):
        odom_labels[ label ].grid( row=r, column=c )
        odom_values[ label ].grid( row=r, column=c+1 )
        label += 1
    
    # Read the output from the kalman filter
    # The four fields are x, y position and x, y velocity
    self.filter_sub = rospy.Subscriber( 'navbot/estimate', Quaternion, 
                                         self.filter_callback )
    
    self.filter_vars = []
  
    for i in xrange(4):
      self.filter_vars.append( StringVar() )
    
    # Labels

    filter_labels = []

    filter_labels.append( Label( self.top, text="Estimated X Position" ) )
    filter_labels.append( Label( self.top, text="Estimated Y Position" ) )
    filter_labels.append( Label( self.top, text="Estimated X Velocity" ) )
    filter_labels.append( Label( self.top, text="Estimated Y Velocity" ) )

    # Display Fields

    filter_values = []
    for i in xrange(4):
      filter_values.append( Label( self.top, textvariable=self.filter_vars[i] ) )
    
    label = 0
    for c in [0,2]:
      for r in xrange(12,14):
        filter_labels[ label ].grid( row=r, column=c )
        filter_values[ label ].grid( row=r, column=c+1 )
        label += 1

  def send_callback( self ):
    self.msg.x = float( self.x_var.get() )
    self.msg.y = float( self.y_var.get() )
    self.msg.z = float( self.z_var.get() )
    self.msg.w = float( self.yaw_var.get() )
    self.dest_pub.publish( self.msg )
  
  def filter_callback( self, data ):

    self.filter_vars[0].set( self.op % data.x )
    self.filter_vars[1].set( self.op % data.y )
    self.filter_vars[2].set( self.op % data.z )
    self.filter_vars[3].set( self.op % data.w )

  def odom_callback( self, data ):

    self.odom_vars[0].set( self.op % data.pose.pose.position.x )
    self.odom_vars[1].set( self.op % data.pose.pose.position.y )
    self.odom_vars[2].set( self.op % data.pose.pose.position.z )
    
    self.odom_vars[3].set( self.op % data.twist.twist.linear.x )
    self.odom_vars[4].set( self.op % data.twist.twist.linear.y )
    self.odom_vars[5].set( self.op % data.twist.twist.linear.z )
      
    x = data.pose.pose.orientation.x
    y = data.pose.pose.orientation.y
    z = data.pose.pose.orientation.z
    w = data.pose.pose.orientation.w

    quaternion = ( x, y, z, w ) 
    euler = tf.transformations.euler_from_quaternion( quaternion )
    
    self.odom_vars[6].set( self.op % euler[0] )
    self.odom_vars[7].set( self.op % euler[1] )
    self.odom_vars[8].set( self.op % euler[2] )
    
    self.odom_vars[9].set( self.op % data.twist.twist.angular.x )
    self.odom_vars[10].set( self.op % data.twist.twist.angular.y )
    self.odom_vars[11].set( self.op % data.twist.twist.angular.z )

  def start( self ):

    self.top.mainloop()


gui = ControlGUI()
gui.start()
