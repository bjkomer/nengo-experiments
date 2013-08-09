import numeric as np
import math
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

PERIOD = 10
STRENGTH = 20 # Constant factor applied to torque output

components = [ 'body', 'back_left_upper_leg', 'back_right_upper_leg', 'front_left_upper_leg', 
               'front_right_upper_leg', 'head', 'back_left_lower_leg', 'back_right_lower_leg',
               'front_left_lower_leg', 'front_right_lower_leg' ]

leg_components = [ 'back_left_upper_leg', 'back_right_upper_leg', 'front_left_upper_leg',
                   'front_right_upper_leg', 'back_left_lower_leg', 'back_right_lower_leg',
                   'front_left_lower_leg', 'front_right_lower_leg' ]

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
    self.sock_in = connect_port( PORT_ODOM ) # Reads odometry
    self.sock_out = connect_port( PORT_CONT ) # Outputs control signal (torque on each limb)
    if not self.sock_in or not self.sock_out:
      sys.exit( 1 )

    self.comp = {} # dictionary of populations representing commands for body components
    self.angle = {} # dictionary of populations representing current angles of body components

    for c in components:
      self.comp[ c ] = self.make_input( c, dimensions=1 )
      self.angle[ c ] = self.make_output( c + '_angle', dimensions=1 )

    self.counter = 0

  def tick( self ):

    self.counter += 1
    if self.counter % PERIOD == 0:
      
      # Send Data

      data_list = []
      data_out = '{"data":['
      for key, value in self.comp.iteritems():

        if data_out != '{"data":[':
          data_out += ','

        val = np.array( value.get() )[0]
        #data_list.append( {"component":key,"force":[0, 0, 0],"torque":[0, val * 100, 0]} )
        data_out += '{"component":"%s","force":[0, 0, 0],"torque":[0, %f, 0]}' % ( key, val * STRENGTH )

      #self.sock_out.send( str( data_list ) + '\n' )
      data_out += ']}\n'
      self.sock_out.send( data_out )
      
      # Read Data

      data_in = None
      morse = self.sock_in.makefile("r")
      try:
        data_in = json.loads(morse.readline())
      except JSONDecodeError:
        try:
          data_in = json.loads(morse.readline())
        except JSONDecodeError:
          print "ERROR: malformed message, dropping data, skipping cycle"
          return

      for comp in data_in[ 'data' ]:
        self.angle[ comp[ 'component' ] ].set( [ comp[ 't' ] ] )

class SineWave( nef.Node ):
  def __init__( self, name ):
    nef.Node.__init__( self, name )
    
    self.wave = self.make_output( 'wave', dimensions=1 )

  def tick( self ):

    self.wave.set( [ math.sin( 5 * self.t ) ] )

class UpperLegCycle( nef.Node ):
  def __init__( self, name, period, offset ):
    nef.Node.__init__( self, name )
    self.period = period
    self.offset = offset
    self.section = self.period / 8.0
    #self.angles = [ math.radians( 10 ), math.radians( 20 ), math.radians( 30 ), math.radians( -60 ),
    #                math.radians( -70 ), math.radians( -60 ), math.radians( 45 ), math.radians( 0 ) ]
    self.angles = [ math.radians( 20 ), math.radians( 20 ), math.radians( 20 ), math.radians( -30 ),
                    math.radians( -45 ), math.radians( -45 ), math.radians( -30 ), math.radians( 0 ) ]

    self.wave = self.make_output( 'wave', dimensions=1 )

  def func( self, t ):
    
    for s in xrange( 8 ):
      if t < self.section * ( s + 1 ):
        # ( corrected time ) * ( rise / run ) + offset
        return ( t - self.section * s ) * ( self.angles[s] - self.angles[s-1] ) / self.section + self.angles[s]

  def tick( self ):

    self.wave.set( [ self.func( ( self.t - self.offset ) - \
      self.period * math.floor( ( self.t - self.offset )/ self.period ) ) ] )

class LowerLegCycle( nef.Node ):
  def __init__( self, name, period, offset ):
    nef.Node.__init__( self, name )
    self.period = period
    self.offset = offset
    self.section = self.period / 8.0
    #self.angles = [ math.radians( 0 ), math.radians( 0 ), math.radians( 45 ), math.radians( 30 ),
    #                math.radians( 90 ), math.radians( 0 ), math.radians( 45 ), math.radians( 30 ) ]
    self.angles = [ math.radians( 0 ), math.radians( 0 ), math.radians( 45 ), math.radians( 30 ),
                    math.radians( 90 ), math.radians( 0 ), math.radians( 45 ), math.radians( 30 ) ]
    
    self.wave = self.make_output( 'wave', dimensions=1 )

  def func( self, t ):
 
    for s in xrange( 8 ):
      if t < self.section * ( s + 1 ):
        # ( corrected time ) * ( rise / run ) + offset
        return ( t - self.section * s ) * ( self.angles[s] - self.angles[s-1] ) / self.section + self.angles[s]

  def tick( self ):

    self.wave.set( [ self.func( ( self.t - self.offset ) - \
      self.period * math.floor( ( self.t - self.offset )/ self.period ) ) ] )


net = nef.Network( 'Four Legged Control', seed=13 )

person = net.add( Person( 'person' ) )

sine = net.add( SineWave( 'sine' ) )

upper_cycle_1 = net.add( UpperLegCycle( 'upper_cycle_1', 5, 0 ) )
upper_cycle_2 = net.add( UpperLegCycle( 'upper_cycle_2', 5, 2.5 ) )
lower_cycle_1 = net.add( LowerLegCycle( 'lower_cycle_1', 5, 0 ) )
lower_cycle_2 = net.add( LowerLegCycle( 'lower_cycle_2', 5, 2.5 ) )

#for c in components:
#  net.make_input( c, [0] )
#  net.connect( c, person.getTermination( c ) )

net.make( 'wave', 100, dimensions=1 )
net.make( 'back_left_upper_leg_pop', 100, dimensions=1 )
net.make( 'back_right_upper_leg_pop', 100, dimensions=1 )
net.make( 'front_left_upper_leg_pop', 100, dimensions=1 )
net.make( 'front_right_upper_leg_pop', 100, dimensions=1 )
net.make( 'back_left_lower_leg_pop', 100, dimensions=1 )
net.make( 'back_right_lower_leg_pop', 100, dimensions=1 )
net.make( 'front_left_lower_leg_pop', 100, dimensions=1 )
net.make( 'front_right_lower_leg_pop', 100, dimensions=1 )

net.connect( sine.getOrigin( 'wave' ), 'wave' )

#net.connect( 'wave', 'back_left_upper_leg_pop', weight=-1 )
#net.connect( 'wave', 'back_right_upper_leg_pop' )
#net.connect( 'wave', 'front_left_upper_leg_pop' )
#net.connect( 'wave', 'front_right_upper_leg_pop', weight=-1 )

#net.connect( 'wave', 'back_left_lower_leg_pop', weight=-1 )
#net.connect( 'wave', 'back_right_lower_leg_pop' )
#net.connect( 'wave', 'front_left_lower_leg_pop' )
#net.connect( 'wave', 'front_right_lower_leg_pop', weight=-1 )

net.connect( upper_cycle_2.getOrigin( 'wave' ), 'back_left_upper_leg_pop' )
net.connect( upper_cycle_1.getOrigin( 'wave' ), 'back_right_upper_leg_pop' )
net.connect( upper_cycle_1.getOrigin( 'wave' ), 'front_left_upper_leg_pop' )
net.connect( upper_cycle_2.getOrigin( 'wave' ), 'front_right_upper_leg_pop' )

net.connect( lower_cycle_2.getOrigin( 'wave' ), 'back_left_lower_leg_pop' )
net.connect( lower_cycle_1.getOrigin( 'wave' ), 'back_right_lower_leg_pop' )
net.connect( lower_cycle_1.getOrigin( 'wave' ), 'front_left_lower_leg_pop' )
net.connect( lower_cycle_2.getOrigin( 'wave' ), 'front_right_lower_leg_pop' )

for leg in leg_components:
  net.connect( person.getOrigin( leg + '_angle' ), leg + '_pop', weight=-1 )
  net.connect( leg + '_pop', person.getTermination( leg ) )

net.view()
net.add_to_nengo()
