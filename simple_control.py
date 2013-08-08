import numeric as np
import socket
import sys
sys.path.append('/home/komer/Downloads/jyson-1.0.2/src')
sys.path.append('/home/komer/Downloads/jyson-1.0.2/lib/jyson-1.0.2.jar')
import com.xhaus.jyson.JysonCodec as json # Jython version of json
from com.xhaus.jyson import JSONDecodeError, JSONEncodeError

HOST = '127.0.0.1'
PORT_ODOM = 60000
PORT_CONT = 60001

import nef

PERIOD = 100

#components = [ 'head', 'neck', 'body', 'left_arm', 'right_arm', 'left_upper_leg', 'right_upper_leg',
#               'left_lower_leg', 'right_lower_leg', 'left_foot', 'right_foot' ]

#components = [ 'Neck', 'Body', 'Left_Upperarm', 'Right_Upperarm', 'Left_Leg', 'Right_Leg',
#               'Left_Hip', 'Right_Hip', 'Left_Forearm', 'Right_Forearm' ]

components = [ 'body', 'back_left_upper_leg', 'back_right_upper_leg', 'front_left_upper_leg', 
               'front_right_upper_leg', 'head' ]

def connect_port( port ):
  """ Establish the connection with the given MORSE port"""
  sock = None

  for res in socket.getaddrinfo(HOST, port, socket.AF_UNSPEC, socket.SOCK_STREAM):
    af, socktype, proto, canonname, sa = res
    try:
      sock = socket.socket(af, socktype, proto)
    except socket.error:
      sock = None
      continue
    try:
      sock.connect(sa)
    except socket.error:
      sock.close()
      sock = None
      continue
    break

  return sock

class Person( nef.Node ):
  def __init__( self, name ):

    nef.Node.__init__( self, name )
    # Connect with MORSE through a socket
    self.sock_out = connect_port( PORT_CONT ) # Outputs control signal (torque on each limb)
    if not self.sock_out:
      sys.exit( 1 )

    self.comp = {} # dictionary of populations representing commands for body components
    
    for c in components:
      self.comp[ c ] = self.make_input( c, dimensions=1 )

    self.counter = 0

  def tick( self ):

    self.counter += 1
    if self.counter % PERIOD == 0:

      data_list = []
      data_out = '{"data":['
      for key, value in self.comp.iteritems():

        if data_out != '{"data":[':
          data_out += ','

        val = np.array( value.get() )[0]
        #data_list.append( {"component":key,"force":[0, 0, 0],"torque":[0, val * 100, 0]} )
        data_out += '{"component":"%s","force":[0, 0, 0],"torque":[0, %f, 0]}' % ( key, val * 100 )

      #self.sock_out.send( str( data_list ) + '\n' )
      data_out += ']}\n'
      self.sock_out.send( data_out )

net = nef.Network( 'Humanoid Control' )

person = net.add( Person( 'person' ) )


for c in components:
  net.make_input( c, [0] )
  net.connect( c, person.getTermination( c ) )

net.view()
net.add_to_nengo()
