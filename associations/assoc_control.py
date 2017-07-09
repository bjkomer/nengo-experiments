# Associate place and flavour using the Voja learning rule
# Learn the associations both ways simultaneously. Can query either location or flavour
import nengo
import numpy as np
from world import FlavourLand
from utils import RandomRun
from collections import OrderedDict

random_inputs = False

flavours = OrderedDict({"Banana":(1,2),
            "Peach":(4,4),
            "Lemon":(6,1),
            "Apple":(2,7)
           })

shape = (10,10)

model = nengo.Network(seed=14)

# compute a control signal to get to the location
def compute_velocity(x):
    # which way the agent should face to go directly to the target
    desired_ang = np.arctan2(x[1], x[0])
    ang_diff = x[2] - desired_ang
    #ang_diff = desired_ang - x[2]
    if ang_diff > np.pi:
        ang_diff -= 2*np.pi
    elif ang_diff < -np.pi:
        ang_diff += 2*np.pi

    ang_vel = ang_diff*.9
    if abs(ang_diff) < np.pi/2.:
        #lin_vel = .8*np.sqrt(x[0]**2+x[1]**2)
        lin_vel = 0
    else:
        #lin_vel = 0
        lin_vel = 3.0*np.sqrt(x[0]**2+x[1]**2)
        ang_vel *= lin_vel # this will stop oscillations once it reaches the destination
    
    return lin_vel, ang_vel

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

def env_scale_node(t, x):
    x_out = (x[0]+1)*(shape[0]/2.)
    y_out = (x[1]+1)*(shape[1]/2.)

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

def loc_scale_to_surface(x):
    # scale to between -1 and 1
    x_in = x[0]/(shape[0]/2.) - 1
    y_in = x[1]/(shape[1]/2.) - 1
    #x_in = x[0]
    #y_in = x[1]

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

def surface_to_env(x):

    xp = np.arctan2(x[1], x[0]) /np.pi
    yp = np.arctan2(x[3], x[2]) /np.pi

    return xp, yp

# temporary artificial basal ganglia
# x[0]: control signal
# x[[1,2]] : environment position
# x[[3,4]] : query position
def control_loc(t, x):
    recall_mode = x[0] > .5
    if recall_mode:
        return x[3], x[4]
    else:
        return x[1], x[2]

def control_flav(t, x):
    recall_mode = x[0] > .5
    if recall_mode:
        return x[5], x[6], x[7], x[8]
    else:
        return x[1], x[2], x[3], x[4]

#intercept = (np.dot(keys, keys.T) - np.eye(num_items)).flatten().max()
intercept = 0 # calculated from another script

with model:
    fl = FlavourLand(shape=shape, flavours=flavours, 
                     flavour_rad=.1, motion_type='velocity')
    env = nengo.Node(fl, size_in=2, size_out=3+len(flavours))

    #TODO: put in basal ganglia that switches between learning and recall modes
    control_node_loc = nengo.Node(control_loc, size_in=5, size_out=2)
    control_node_flav = nengo.Node(control_flav, size_in=9, size_out=4)

    # x, y, th
    location = nengo.Ensemble(n_neurons=300, dimensions=3)

    taste = nengo.Ensemble(n_neurons=10*len(flavours), dimensions=len(flavours), radius=1.2)

    # linear and angular velocity
    if random_inputs:
        velocity = nengo.Node(RandomRun())
    else:
        velocity = nengo.Node([0,0])
    nengo.Connection(velocity[:2], env[:2])

    #nengo.Connection(env[:3], location, function=scale_location)
    scaled_loc = nengo.Node(size_in=3,size_out=3)
    nengo.Connection(env[:3], scaled_loc, function=scale_location)

    nengo.Connection(env[3:], taste)

    voja_loc = nengo.Voja(post_tau=None, learning_rate=5e-2)
    voja_flav = nengo.Voja(post_tau=None, learning_rate=5e-2)

    #memory = nengo.Ensemble(n_neurons=300, dimensions=3)
    #memory = nengo.Ensemble(n_neurons=200, dimensions=2)
    #memory = nengo.Ensemble(n_neurons=400, dimensions=4, intercepts=[intercept]*400)
    #memory = nengo.Ensemble(n_neurons=200, dimensions=2, intercepts=[intercept]*200)
    
    #memory_loc = nengo.Ensemble(n_neurons=400, dimensions=4)
    memory_loc = nengo.Ensemble(n_neurons=400, dimensions=4)
    memory_flav = nengo.Ensemble(n_neurons=200, dimensions=4, intercepts=[intercept]*200)
    
    # Query a location to get a response for what flavour was there
    query_location = nengo.Node([0,0])
    query_loc_scaled = nengo.Node(scale_xy_node, size_in=2, size_out=2)
    nengo.Connection(query_location, query_loc_scaled, synapse=None)

    query_flavour = nengo.Node([0,0,0,0])
    
    # Choose between learning and recall #TODO: use basal ganglia instead
    task = nengo.Node([0])

    nengo.Connection(task, control_node_loc[0])
    nengo.Connection(task, control_node_flav[0])
    
    nengo.Connection(scaled_loc[:2], control_node_loc[[1,2]])
    nengo.Connection(query_loc_scaled, control_node_loc[[3,4]])
    
    nengo.Connection(env[3:], control_node_flav[[1,2,3,4]])
    nengo.Connection(query_flavour, control_node_flav[[5,6,7,8]])

    conn_in_loc = nengo.Connection(control_node_loc, memory_loc, function=loc_to_surface, 
                                   learning_rule_type=voja_loc, synapse=None)
    conn_in_flav = nengo.Connection(control_node_flav, memory_flav, 
                                    learning_rule_type=voja_flav, synapse=None)

    nengo.Connection(task, conn_in_loc.learning_rule, synapse=None, transform=-1)
    nengo.Connection(task, conn_in_flav.learning_rule, synapse=None, transform=-1)
    
    #conn_in = nengo.Connection(env[:3], memory, learning_rule_type=voja)
    ##conn_in = nengo.Connection(env[:2], memory, learning_rule_type=voja)

    # Try only learning when flavours present
    #learning = nengo.Ensemble(n_neurons=100, dimensions=1)
    learning = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.Direct())
    inhibition = nengo.Node([-1])
    nengo.Connection(inhibition, learning)
    
    nengo.Connection(learning, conn_in_loc.learning_rule, synapse=None, transform=1)
    nengo.Connection(learning, conn_in_flav.learning_rule, synapse=None, transform=1)

    nengo.Connection(env[3:], learning, transform=1*np.ones((1,len(flavours))))

    recall_flav = nengo.Ensemble(n_neurons=200, dimensions=len(flavours))
    recall_loc = nengo.Ensemble(n_neurons=400, dimensions=4)

    conn_out_flav = nengo.Connection(memory_loc, recall_flav,
                                learning_rule_type=nengo.PES(1e-3),
                                function=lambda x: np.random.random(len(flavours))
                               )
    conn_out_loc = nengo.Connection(memory_flav, recall_loc,
                                learning_rule_type=nengo.PES(1e-3),
                                function=lambda x: np.random.random(4)
                               )

    error_flav = nengo.Ensemble(n_neurons=200, dimensions=len(flavours))
    error_loc = nengo.Ensemble(n_neurons=400, dimensions=4)

    # Connect up error populations
    nengo.Connection(env[3:], error_flav, transform=-1, synapse=None)
    nengo.Connection(recall_flav, error_flav, transform=1, synapse=None)
    nengo.Connection(error_flav, conn_out_flav.learning_rule)
    
    surface_loc = nengo.Node(size_in=4,size_out=4)
    nengo.Connection(env[:2], surface_loc, function=loc_scale_to_surface, synapse=None)
    nengo.Connection(surface_loc, error_loc, transform=-1, synapse=None)
    nengo.Connection(recall_loc, error_loc, transform=1, synapse=None)
    nengo.Connection(error_loc, conn_out_loc.learning_rule)

    # inhibit learning based on learning signal
    nengo.Connection(learning, error_flav.neurons, transform=[[3]] * 200, synapse=None)
    nengo.Connection(task, error_flav.neurons, transform=[[-3]] * 200, synapse=None)
    
    nengo.Connection(learning, error_loc.neurons, transform=[[3]] * 400, synapse=None)
    nengo.Connection(task, error_loc.neurons, transform=[[-3]] * 400, synapse=None)

    scaled_recall_loc = nengo.Node(env_scale_node, size_in=2, size_out=2)
    nengo.Connection(recall_loc, scaled_recall_loc, function=surface_to_env)

    # if this flag is set to 0, the agent will drive towards the target in scaled_recall_loc
    drive = nengo.Node([1])


    vel_input = nengo.Ensemble(n_neurons=200, dimensions=2)

    nengo.Connection(drive, vel_input.neurons, transform=[[-30]]*200, synapse=None)

    nengo.Connection(vel_input, env)

    #FIXME: switch this to LIF once its working
    pos_error = nengo.Ensemble(n_neurons=300, dimensions=3, neuron_type=nengo.Direct())

    nengo.Connection(env[:3], pos_error, transform=1)
    nengo.Connection(scaled_recall_loc, pos_error[:2], transform=-1)

    nengo.Connection(pos_error, vel_input, function=compute_velocity)
    
    
    # This part follows the users commands
    c_drive = nengo.Node([1])

    command = nengo.Node([0,0])
    c_vel_input = nengo.Ensemble(n_neurons=200, dimensions=2)

    nengo.Connection(c_drive, c_vel_input.neurons, transform=[[-30]]*200, synapse=None)

    nengo.Connection(c_vel_input, env)

    c_pos_error = nengo.Ensemble(n_neurons=300, dimensions=3, neuron_type=nengo.Direct())

    nengo.Connection(env[:3], c_pos_error, transform=1)
    nengo.Connection(command, c_pos_error[:2], transform=-1)

    nengo.Connection(c_pos_error, c_vel_input, function=compute_velocity)
