import nengo
import numpy as np
from world import FlavourLand

#TODO: only import the functions that are used
from utils import to_xy, to_xyz, cyclic_to_xy, ang_to_cyclic, ang_to_xy, integrate_velocity

#NOTE: this is currently linked to this directory
from html_plots import CompleteSpatialSpikePlot

model = nengo.Network(seed=13)

flavours = {"Banana":(1,2),
            "Peach":(3,3),
            "Lemon":(4,1),
            "Apple":(2,5)
           }

shape = (10,10)

# Scales the location output to be between -1 and 1
def scale_location(x):
    x_out = x[0]/(shape[0]/2) - 1
    y_out = x[1]/(shape[1]/2) - 1
    th_out = x[2]/np.pi #TODO: should this get scaled at all?

    return [x_out, y_out, th_out]

with model:
    fl = FlavourLand(shape=shape, flavours=flavours)
    env = nengo.Node(fl, size_in=2, size_out=3+len(flavours))

    # linear and angular velocity
    velocity = nengo.Node([0,0])

    # x, y, th
    location = nengo.Ensemble(n_neurons=300, dimensions=3)

    taste = nengo.Ensemble(n_neurons=10*len(flavours), dimensions=len(flavours))

    nengo.Connection(velocity, env)

    nengo.Connection(env[:3], location, function=scale_location)

    nengo.Connection(env[3:], taste)


