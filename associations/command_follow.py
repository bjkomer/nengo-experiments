# agent moves to the location it is commanded to go
# Associate place and flavour using the Voja learning rule
import nengo
import numpy as np
from world import FlavourLand
from utils import RandomRun

random_inputs = False

flavours = {"Banana":(1,2),
            "Peach":(4,4),
            "Lemon":(6,1),
            "Apple":(2,7)
           }

shape = (10,10)

model = nengo.Network(seed=14)

# Scales the location output to be between -1 and 1
def scale_location(x):
    x_out = x[0]/(shape[0]/2.) - 1
    y_out = x[1]/(shape[1]/2.) - 1
    th_out = x[2]/np.pi #TODO: should this get scaled at all?

    return [x_out, y_out, th_out]

def scale_xy_node(t, x):
    x_out = x[0]/(shape[0]/2.) - 1
    y_out = x[1]/(shape[1]/2.) - 1

    return x_out, y_out

def loc_to_surface(x):
    # scale to between -1 and 1
    #x_in = x[0]/(shape[0]/2.) - 1
    #y_in = x[1]/(shape[1]/2.) - 1
    x_in = x[0]
    y_in = x[1]

    # take sin and cos between -pi and pi
    xx = np.cos(x_in*np.pi)
    xy = np.sin(x_in*np.pi)
    
    yx = np.cos(y_in*np.pi)
    yy = np.sin(y_in*np.pi)

    #NOTE: normalizing here changes the intercept from 1 to 0, huge improvement!
    denom = np.sqrt(xx**2+xy**2+yx**2+yy**2)
    xx/=denom
    xy/=denom
    yx/=denom
    yy/=denom

    return xx, xy, yx, yy

# temporary artificial basal ganglia
# x[0]: control signal
# x[[1,2]] : environment position
# x[[3,4]] : query position
def control(t, x):
    recall_mode = x[0] > .5
    if recall_mode:
        return x[3], x[4]
    else:
        return x[1], x[2]

# compute a control signal to get to the location
def compute_velocity(x):
    # which way the agent should face to go directly to the target
    desired_ang = np.arctan2(-x[1], -x[0])
    
    ang_diff = -1*(x[2] - desired_ang)
    
    if ang_diff > np.pi:
        ang_diff -= 2*np.pi
    elif ang_diff < -np.pi:
        ang_diff += 2*np.pi

    ang_vel = ang_diff*.9
    if np.sqrt(x[0]**2+x[1]**2) < .001:
        lin_vel = 0
        ang_vel = 0
    elif abs(ang_diff) < np.pi/4.:
        lin_vel = .6*np.sqrt(x[0]**2+x[1]**2)
    elif abs(ang_diff) < np.pi/2.:
        lin_vel = .4*np.sqrt(x[0]**2+x[1]**2)
    else:
        lin_vel = 0
    
    return lin_vel, ang_vel

model.config[nengo.Ensemble].neuron_type=nengo.Direct()

with model:
    fl = FlavourLand(shape=shape, flavours=flavours, motion_type='velocity')
    env = nengo.Node(fl, size_in=2, size_out=3+len(flavours))

    command = nengo.Node([0,0])

    vel_input = nengo.Ensemble(n_neurons=200, dimensions=2, radius=3)

    nengo.Connection(vel_input, env)

    pos_error = nengo.Ensemble(n_neurons=300, dimensions=3, radius=10)

    nengo.Connection(env[:3], pos_error, transform=1)
    nengo.Connection(command, pos_error[:2], transform=-1)

    nengo.Connection(pos_error, vel_input, function=compute_velocity)
