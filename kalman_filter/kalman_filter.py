# Kalman filter implemented in neurons
import nengo
import numpy as np
import rospy

from nengo.utils.ros import PoseNode

# 4D node that outputs a Quaternion, which will be used to represent the
# x,y position and x,y velocity estimate
from nengo.utils.ros import RotorcraftAttitudeNode as EstimateNode

dt = 0.001
N = 100
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
                  [0.,1.,0,dt],
                  [0.,0.,1.,0.],
                  [0.,0.,0.,1.]])
Hmat = np.matrix([[1.,0.,0.,0.],
                  [0.,1.,0.,0.]])
Rmat = np.matrix([[.1,0.],
                  [0.,.1]])
Ymat = np.matrix([[0.],
                  [0.]])

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

# Computes the delta P
# -(P * H.T * (H * P * H.T + R).I) * H * P
def Q_to_P(x):

  #return (-K * H * P).tolist()
  Pmat = np.matrix([[x[2],0,0,0],
                    [0,x[3],0,0],
                    [0,0,x[4],0],
                    [0,0,0,x[5]]])
  Pnew = -(Pmat * Hmat.T * (Hmat * Pmat * Hmat.T + Rmat).I) * Hmat * Pmat
  return [Pnew[0,0], Pnew[1,1], Pnew[2,2], Pnew[3,3]]

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

rospy.init_node( 'kalman_filter', anonymous=True )

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
 
  inputP = nengo.Node([10,10,1000,1000])
  nengo.Connection(inputP, P)

  probe_pose = nengo.Probe(Z, "output", synapse=0.01)
  probe_estimate = nengo.Probe(X, "decoded_output", synapse=0.01)
  probe_Y = nengo.Probe(Y, "decoded_output", synapse=0.01)
  probe_Q = nengo.Probe(Q, "decoded_output", synapse=0.01)
  probe_p_diag = nengo.Probe(P, "decoded_output", synapse=0.01)
  

import time
print( "starting simulator..." )
before = time.time()

sim = nengo.Simulator( model, fixed_time=True )

after = time.time()
print( "time to build:" )
print( after - before )

print( "running simulator..." )
before = time.time()

sim.run(20)

after = time.time()
print( "time to run:" )
print( after - before )

import matplotlib.pyplot as plt

plt.subplot(5, 1, 1)
plt.plot(sim.trange(), sim.data[probe_pose], lw=2)
plt.title("Pose")

plt.subplot(5, 1, 2)
plt.plot(sim.trange(), sim.data[probe_estimate], lw=2)
plt.title("Estimate")

plt.subplot(5, 1, 3)
plt.plot(sim.trange(), sim.data[probe_Y], lw=2)
plt.axvline(0.2, c='k')
plt.title("Y")

plt.subplot(5, 1, 4)
plt.plot(sim.trange(), sim.data[probe_p_diag], lw=2)
plt.title("P diagonal")

plt.subplot(5, 1, 5)
plt.plot(sim.trange(), sim.data[probe_Q], lw=2)
plt.title("Q")

plt.tight_layout()

plt.show()
