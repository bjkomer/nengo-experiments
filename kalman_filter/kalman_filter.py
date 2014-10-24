# Kalman filter implemented in neurons
import nengo
import numpy as np
#import rospy

#from nengo.utils.ros import PoseNode

# 4D node that outputs a Quaternion, which will be used to represent the
# x,y position and x,y velocity estimate
#from nengo.utils.ros import RotorcraftAttitudeNode as EstimateNode

dt = 0.001
N = 20
radius = 5
# pstc = ???

model = nengo.Network(label='Kalman Filter', seed=13)
model.config[nengo.Ensemble].neuron_type=nengo.Direct()
#model.config[nengo.Ensemble].neuron_type=nengo.LIF()
model.config[nengo.Ensemble].radius=radius
#model.config[nengo.Ensemble].synpase=pstc

Xmat = np.matrix([[0.],[0.],[0.],[0.]])
Umat = np.matrix([[0.],[0.],[0.],[0.]])
Pmat = np.matrix([[0.,0.,0.,0.],
                  [0.,0.,0.,0.],
                  [0.,0.,1000.,0.],
                  [0.,0.,0.,1000.]])
Fmat = np.matrix([[1.,0.,dt,0.],
                  [0.,1.,0.,dt],
                  [0.,0.,1.,0.],
                  [0.,0.,0.,1.]])
Hmat = np.matrix([[1.,0.,0.,0.],
                  [0.,1.,0.,0.]])
Rmat = np.matrix([[.1,0.],
                  [0.,.1]])
Ymat = np.matrix([[0.],
                  [0.]])
Imat = np.matrix([[1.,0.,0.,0.],
                  [0.,1.,0.,0.],
                  [0.,0.,1.,0.],
                  [0.,0.,0.,1.]])

def trans(pre_dims, post_dims,
          weight=1.0, index_pre=None, index_post=None):
    """Helper function used to create a ``pre_dims`` by ``post_dims``
    linear transformation matrix.

    Parameters
    ----------
    pre_dims, post_dims : int
        The numbers of presynaptic and postsynaptic dimensions.
    weight : float, optional
        The weight value to use in the transform.

        All values in the transform are either 0 or ``weight``.

        **Default**: 1.0
    index_pre, index_post : iterable of int
        Determines which values are non-zero, and indicates which
        dimensions of the pre-synaptic ensemble should be routed to which
        dimensions of the post-synaptic ensemble.

    Returns
    -------
    transform : 2D matrix of floats
        A two-dimensional transform matrix performing the requested routing.

    Examples
    --------

      # Sends the first two dims of pre to the first two dims of post
      >>> gen_transform(pre_dims=2, post_dims=3,
                        index_pre=[0, 1], index_post=[0, 1])
      [[1, 0], [0, 1], [0, 0]]

    """
    t = [[0 for pre in xrange(pre_dims)] for post in xrange(post_dims)]
    if index_pre is None:
        index_pre = range(pre_dims)
    elif isinstance(index_pre, int):
        index_pre = [index_pre]

    if index_post is None:
        index_post = range(post_dims)
    elif isinstance(index_post, int):
        index_post = [index_post]

    for i in xrange(min(len(index_pre), len(index_post))):  # was max
        pre = index_pre[i]  # [i % len(index_pre)]
        post = index_post[i]  # [i % len(index_post)]
        t[post][pre] = weight
    return t

#TODO: make these functions more efficient
#      currently designed to be readable rather than optimal

# -(H * (F * np.asarray(x) + U))
def X_to_Y(x):
  #return (-(H * np.asarray(x))).tolist()
  val = -(Hmat * ((Fmat * np.matrix([x]).T) + Umat))
  return [val[0,0],val[1,0]]

# Sends the prediction update of P to the Q population
# F * P * F.T
def P_to_Q(x):

  Pmat = np.matrix([[x[0],0,0,0],
                    [0,x[1],0,0],
                    [0,0,x[2],0],
                    [0,0,0,x[3]]])
  Pnew = Fmat * Pmat * Fmat.T
  return [Pnew[0,0], Pnew[1,1], Pnew[2,2], Pnew[3,3]]

# uses full 4x4 P matrix
def P_to_Q_full(x):

  Pmat = np.matrix([[x[0],x[1],x[2],x[3]],
                    [x[4],x[5],x[6],x[7]],
                    [x[8],x[9],x[10],x[11]],
                    [x[12],x[13],x[14],x[15]]])
  Pnew = Fmat * Pmat * Fmat.T
  return Pnew.ravel().tolist()[0]

# Computes the delta P
# -(P * H.T * (H * P * H.T + R).I) * H * P
def Q_to_P(x):

  Pmat = np.matrix([[x[2],0,0,0],
                    [0,x[3],0,0],
                    [0,0,x[4],0],
                    [0,0,0,x[5]]])
  Pnew = -(Pmat * Hmat.T * (Hmat * Pmat * Hmat.T + Rmat).I) * Hmat * Pmat
  return [Pnew[0,0], Pnew[1,1], Pnew[2,2], Pnew[3,3]]

# uses full 4x4 P matrix
def Q_to_P_full(x):

  Pmat = np.matrix([[x[2],x[3],x[4],x[5]],
                    [x[6],x[7],x[8],x[9]],
                    [x[10],x[11],x[12],x[13]],
                    [x[14],x[15],x[16],x[17]]])
  Pnew = -(Pmat * Hmat.T * (Hmat * Pmat * Hmat.T + Rmat).I) * Hmat * Pmat
  #Pnew = Pnew * .1 # FIXME: temp
  return Pnew.ravel().tolist()[0]

# Computes the delta X
# P * H.T * (H * P * H.T + R).I * Y
def Q_to_X(x):

  Pmat = np.matrix([[x[2],0,0,0],
                    [0,x[3],0,0],
                    [0,0,x[4],0],
                    [0,0,0,x[5]]])
  Ymat = np.matrix([[x[0]],
                    [x[1]]])
  val = Pmat * Hmat.T * (Hmat * Pmat * Hmat.T + Rmat).I * Ymat
  return [val[0,0], val[1,0], val[2,0], val[3,0]]

# uses full 4x4 P matrix
def Q_to_X_full(x):

  Pmat = np.matrix([[x[2],x[3],x[4],x[5]],
                    [x[6],x[7],x[8],x[9]],
                    [x[10],x[11],x[12],x[13]],
                    [x[14],x[15],x[16],x[17]]])
  Ymat = np.matrix([[x[0]],
                    [x[1]]])
  val = Pmat * Hmat.T * (Hmat * Pmat * Hmat.T + Rmat).I * Ymat
  #val = val * .1 # FIXME: temp
  return [val[0,0], val[1,0], val[2,0], val[3,0]]

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

def test_pose_fun( t ):
  #return [t,0]
  #return ms[int(t)]
  return [t/2+1,t] 
  """
  if t < 3:
    return [t,-t]
  elif t < 6:
    return [(t-3)*1+3,-3]
  elif t < 9:
    return [6, (t-6)*1-3]
  else:
    return [6,0]
  """

#rospy.init_node( 'kalman_filter', anonymous=True )
"""
# TODO: initialize the uncertainty to something other than 0
# TODO: cut Y out of this model, it doesn't seem to be needed
with model:
  Z = PoseNode( 'Pose', mask=[1,1,0,0,0,0], topic='navbot/pose' )
  X = nengo.Ensemble(N, 4)
  Y = nengo.Ensemble(N, 2)
  Q = nengo.Ensemble(N, 6)
  Estimate = EstimateNode( 'Estimate', topic='navbot/estimate' )

  # Simplified to only include the diagonal
  P = nengo.Ensemble(N, 4)

  nengo.Connection(Z, Y)
  nengo.Connection(X, Y, function=X_to_Y)
  
  nengo.Connection(Y, Q, 
                   transform=trans(2, 6, index_pre=[0,1], index_post=[0,1]))
  nengo.Connection(P, Q, function=P_to_Q, 
                   transform=trans(4, 6, index_pre=[0,1,2,3], index_post=[2,3,4,5]))
  
  nengo.Connection(Q, X, function=Q_to_X)
  nengo.Connection(Q, P, function=Q_to_P)

  nengo.Connection(X, Estimate)
 
  inputP = nengo.Node(lambda t: [0,0,1000,1000] if t <.2 else [0,0,0,0])
  nengo.Connection(inputP, P)

  probe_pose = nengo.Probe(Z, "output", synapse=0.01)
  probe_estimate = nengo.Probe(X, "decoded_output", synapse=0.01)
  probe_Y = nengo.Probe(Y, "decoded_output", synapse=0.01)
  probe_Q = nengo.Probe(Q, "decoded_output", synapse=0.01)
  probe_p_diag = nengo.Probe(P, "decoded_output", synapse=0.01)
  
"""
"""
#FIXME: TEMP: trying out full 4x4 P matrix
with model:
  #Z = PoseNode( 'Pose', mask=[1,1,0,0,0,0], topic='navbot/pose' )
  Z = nengo.Node( test_pose_fun )
  X = nengo.Ensemble(N*4, 4)
  Y = nengo.Ensemble(N*2, 2)
  Q = nengo.Ensemble(N*18, 18)
  #Estimate = EstimateNode( 'Estimate', topic='navbot/estimate' )

  P = nengo.Ensemble(N*16, 16)
  
  velocity_estimate = nengo.Ensemble(N, 2)
  position_estimate = nengo.Ensemble(N, 2)

  nengo.Connection(X, position_estimate, 
                   transform=trans(4,2,index_pre=[0,1], index_post=[0,1]))
  nengo.Connection(X, velocity_estimate, 
                   transform=trans(4,2,index_pre=[2,3], index_post=[0,1]))

  nengo.Connection(Z, Y)
  nengo.Connection(X, Y, function=X_to_Y)
  
  nengo.Connection(Y, Q, 
                   transform=trans(2, 18, index_pre=[0,1], index_post=[0,1]))
  nengo.Connection(P, Q, function=P_to_Q_full, 
                   transform=trans(16, 18, index_pre=range(16), index_post=range(2,18)))
  
  nengo.Connection(Q, X, function=Q_to_X_full, synapse=.01)
  nengo.Connection(Q, P, function=Q_to_P_full, synapse=.01)
  
  # Integrators
  #nengo.Connection(P, P, synapse=.1)
  #nengo.Connection(X, X, synapse=.1)
  nengo.Connection(P, P, transform=trans(16,16, weight=0.1))
  nengo.Connection(X, X, transform=trans(4,4, weight=0.1))
  #nengo.Connection(X, X, transform=[[.1,0,0,0],[0,.1,0,0],[0,0,.1,0],[0,0,0,.1]])

  #nengo.Connection(X, Estimate)
 
  #inputP = nengo.Node([10,0,0,0,0,10,0,0,0,0,1000,0,0,0,0,1000])
  inputP = nengo.Node(lambda t: [1.0,0,0,0,0,1.0,0,0,0,0,2.0,0,0,0,0,2.0] if
                      t <.004 else [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  nengo.Connection(inputP, P)

  probe_pose = nengo.Probe(Z, "output", synapse=0.01)
  probe_estimate = nengo.Probe(X, "decoded_output", synapse=0.01)
  probe_Y = nengo.Probe(Y, "decoded_output", synapse=0.01)
  probe_Q = nengo.Probe(Q, "decoded_output", synapse=0.01)
  probe_p_diag = nengo.Probe(P, "decoded_output", synapse=0.01)
  probe_position = nengo.Probe(position_estimate, "decoded_output", synapse=0.01)
  probe_velocity = nengo.Probe(velocity_estimate, "decoded_output", synapse=0.01)
"""
#"""
#FIXME: TEMP: trying out full 4x4 P matrix with more math and single population

tau = 0.1#/dt

def prediction_X( x ):
  # A' = tau*F + (1-tau)*I
  Xmat = np.matrix([[x[0]],
                    [x[1]],
                    [x[2]],
                    [x[3]]])
  val = (tau*Fmat + (1-tau)*Imat)*Xmat
  return [val[0,0], val[1,0], val[2,0], val[3,0]]

def measurement_X( x ):
  # A' = (1+tau)*I - tau*K*H
  Xmat = np.matrix([[x[0]],
                    [x[1]],
                    [x[2]],
                    [x[3]]])
  Pmat = np.matrix([[x[6],x[7],x[8],x[9]],
                    [x[10],x[11],x[12],x[13]],
                    [x[14],x[15],x[16],x[17]],
                    [x[18],x[19],x[20],x[21]]])
  Kmat = Pmat * Hmat.T * (Hmat * Pmat * Hmat.T + Rmat).I
  Amat = (1+tau)*Imat - tau*Kmat*Hmat
  val = Amat*Xmat
  return [val[0,0], val[1,0], val[2,0], val[3,0]]

def measurement_X_input( x ):
  # B` = tau*K
  Zmat = np.matrix([[x[4]],
                    [x[5]]])
  Pmat = np.matrix([[x[6],x[7],x[8],x[9]],
                    [x[10],x[11],x[12],x[13]],
                    [x[14],x[15],x[16],x[17]],
                    [x[18],x[19],x[20],x[21]]])
  Kmat = Pmat * Hmat.T * (Hmat * Pmat * Hmat.T + Rmat).I
  val = tau*Kmat*Zmat
  return [val[0,0], val[1,0], val[2,0], val[3,0]]

#TODO: make sure the P update equations are correct
def prediction_P( x ):
  #TODO: find the correct update equation, this one may be incorrect
  # A` = ???
  # P = F*P*F.T
  # dP/dt = (F*P*F.T - P)/dt
  #f`(P) = tau*f(P) + P
  #
  # f`(dP) = tau*(F*P*F.T - P) + F*P*F.T - P
  Pmat = np.matrix([[x[6],x[7],x[8],x[9]],
                    [x[10],x[11],x[12],x[13]],
                    [x[14],x[15],x[16],x[17]],
                    [x[18],x[19],x[20],x[21]]])

  val = tau*(Fmat*Pmat*Fmat.T - Pmat) + Fmat*Pmat*Fmat.T - Pmat
  return val.ravel().tolist()[0]

def measurement_P( x ):
  # A` = -tau*K*H + I
  Pmat = np.matrix([[x[6],x[7],x[8],x[9]],
                    [x[10],x[11],x[12],x[13]],
                    [x[14],x[15],x[16],x[17]],
                    [x[18],x[19],x[20],x[21]]])
  Kmat = Pmat * Hmat.T * (Hmat * Pmat * Hmat.T + Rmat).I
  Amat = -tau*Kmat*Hmat + Imat
  val = Amat*Pmat
  return val.ravel().tolist()[0]

with model:
  #Z = PoseNode( 'Pose', mask=[1,1,0,0,0,0], topic='navbot/pose' )
  Z = nengo.Node( test_pose_fun )
  X = nengo.Ensemble(N*4, 4)
  #Y = nengo.Ensemble(N*2, 2)

  # 0:4 -> X
  # 4:6 -> Z
  # 6:22 -> P
  Q = nengo.Ensemble(N*22, 22)
  #Estimate = EstimateNode( 'Estimate', topic='navbot/estimate' )

  #P = nengo.Ensemble(N*16, 16)
  
  velocity_estimate = nengo.Ensemble(N, 2)
  position_estimate = nengo.Ensemble(N, 2)

  nengo.Connection(X[0:2], position_estimate[0:2] ) 
  nengo.Connection(X[2:4], velocity_estimate[0:2] ) 

  # Add the measurement as input
  nengo.Connection(Z, Q[4:6] )
  
  # Read out the value of X
  nengo.Connection(Q[0:4], X, synapse=0.1)

  nengo.Connection(Q, Q[0:4], function=prediction_X, synapse=0.1 )
  nengo.Connection(Q, Q[0:4], function=measurement_X, synapse=0.1 )
  nengo.Connection(Q, Q[0:4], function=measurement_X_input, synapse=0.1 )
  nengo.Connection(Q, Q[6:22], function=prediction_P, synapse=0.1 )
  nengo.Connection(Q, Q[6:22], function=measurement_P, synapse=0.1 )

  #nengo.Connection(X, Estimate)
 
  #inputP = nengo.Node([10,0,0,0,0,10,0,0,0,0,1000,0,0,0,0,1000])
  inputP = nengo.Node(lambda t:
                      [1000.0,0,0,0,0,1000.0,0,0,0,0,1000.0,0,0,0,0,1000.0] if
                      t <1.004 else [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  nengo.Connection(inputP, Q[6:22])

  probe_pose = nengo.Probe(Z, "output", synapse=0.01)
  probe_estimate = nengo.Probe(X, "decoded_output", synapse=0.01)
  #probe_Y = nengo.Probe(Y, "decoded_output", synapse=0.01)
  probe_Q = nengo.Probe(Q, "decoded_output", synapse=0.01)
  probe_p_diag = nengo.Probe(Q[6:22], "decoded_output", synapse=0.01)
  probe_position = nengo.Probe(position_estimate, "decoded_output", synapse=0.01)
  probe_velocity = nengo.Probe(velocity_estimate, "decoded_output", synapse=0.01)
#"""

def update_X( x ):
  Xmat = np.matrix([[x[0]],
                    [x[1]],
                    [x[2]],
                    [x[3]]])
  Zmat = np.matrix([[x[4]],
                    [x[5]]])
  Pmat = np.matrix([[x[6],x[7],x[8],x[9]],
                    [x[10],x[11],x[12],x[13]],
                    [x[14],x[15],x[16],x[17]],
                    [x[18],x[19],x[20],x[21]]])
  Kmat = Pmat * Hmat.T * (Hmat * Pmat * Hmat.T + Rmat).I
  Xnew = Fmat * (Xmat + Kmat * (Zmat - Hmat*Xmat))
  return Xnew.ravel().tolist()[0]

def update_P( x ):
  Pmat = np.matrix([[x[6],x[7],x[8],x[9]],
                    [x[10],x[11],x[12],x[13]],
                    [x[14],x[15],x[16],x[17]],
                    [x[18],x[19],x[20],x[21]]])
  Kmat = Pmat * Hmat.T * (Hmat * Pmat * Hmat.T + Rmat).I
  Pnew = Fmat * (Imat - Kmat * Hmat) * Pmat * Fmat.T
  return Pnew.ravel().tolist()[0]

# Combining prediction and measurement update together, and trying that out
with model:
  #Z = PoseNode( 'Pose', mask=[1,1,0,0,0,0], topic='navbot/pose' )
  Z = nengo.Node( test_pose_fun )
  X = nengo.Ensemble(N*4, 4)
  #Y = nengo.Ensemble(N*2, 2)

  # 0:4 -> X
  # 4:6 -> Z
  # 6:22 -> P
  Q = nengo.Ensemble(N*22, 22)
  #Estimate = EstimateNode( 'Estimate', topic='navbot/estimate' )

  #P = nengo.Ensemble(N*16, 16)
  
  velocity_estimate = nengo.Ensemble(N, 2)
  position_estimate = nengo.Ensemble(N, 2)

  nengo.Connection(X[0:2], position_estimate[0:2] ) 
  nengo.Connection(X[2:4], velocity_estimate[0:2] ) 

  # Add the measurement as input
  nengo.Connection(Z, Q[4:6] )
  
  # Read out the value of X
  nengo.Connection(Q[0:4], X, synapse=0.1)

  nengo.Connection(Q, Q[0:4], function=update_X, synapse=0.1 )
  nengo.Connection(Q, Q[6:22], function=update_P, synapse=0.1 )

  #nengo.Connection(X, Estimate)
 
  #inputP = nengo.Node([10,0,0,0,0,10,0,0,0,0,1000,0,0,0,0,1000])
  inputP = nengo.Node(lambda t:
                      [1000.0,0,0,0,0,1000.0,0,0,0,0,1000.0,0,0,0,0,1000.0] if
                      t <.004 else [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  nengo.Connection(inputP, Q[6:22])

  probe_pose = nengo.Probe(Z, "output", synapse=0.01)
  probe_estimate = nengo.Probe(X, "decoded_output", synapse=0.01)
  #probe_Y = nengo.Probe(Y, "decoded_output", synapse=0.01)
  probe_Q = nengo.Probe(Q, "decoded_output", synapse=0.01)
  probe_p_diag = nengo.Probe(Q[6:22], "decoded_output", synapse=0.01)
  probe_position = nengo.Probe(position_estimate, "decoded_output", synapse=0.01)
  probe_velocity = nengo.Probe(velocity_estimate, "decoded_output", synapse=0.01)
#"""


import time
print( "starting simulator..." )
before = time.time()

#sim = nengo.Simulator( model, fixed_time=True )
sim = nengo.Simulator( model, dt=dt )

after = time.time()
print( "time to build:" )
print( after - before )

print( "running simulator..." )
before = time.time()

sim.run(10)

after = time.time()
print( "time to run:" )
print( after - before )

import matplotlib.pyplot as plt

plt.subplot(5, 1, 1)
plt.plot(sim.trange(), sim.data[probe_pose], lw=2)
plt.title("Pose")

#plt.subplot(5, 1, 2)
#plt.plot(sim.trange(), sim.data[probe_estimate], lw=2)
#plt.title("Estimate")

plt.subplot(5, 1, 2)
plt.plot(sim.trange(), sim.data[probe_position], lw=2)
plt.title("Position Estimate")

plt.subplot(5, 1, 3)
plt.plot(sim.trange(), sim.data[probe_velocity], lw=2)
plt.title("Velocity Estimate")

#plt.subplot(5, 1, 4)
#plt.plot(sim.trange(), sim.data[probe_Y], lw=2)
#plt.title("Y")

plt.subplot(5, 1, 5)
plt.plot(sim.trange(), sim.data[probe_p_diag], lw=2)
plt.title("P diagonal")

#plt.subplot(5, 1, 5)
#plt.plot(sim.trange(), sim.data[probe_Q], lw=2)
#plt.title("Q")

plt.tight_layout()

plt.show()
