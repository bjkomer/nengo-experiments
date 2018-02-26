# Associate place and item using the Voja learning rule
# Learn the associations both ways simultaneously. Can query either location or item
# this version has a bunch of stuff cut out/direct mode to run faster

# TODO: at 'test time' query an item, which will give a location,
#       slowly move the location query from the current location to that items location
#       this can be done by taking the vector difference as the direction, and using a
#       unit velocity vector for movement in that direction. Could also use the distance
#       directly and have it saturate
#       this simulates the 'scanning' across the map
#       stop once the recalled item is close enough to the queried item
#       log the time that it takes, and repeat the experiment for a bunch of pairs

# TODO: make more populations use neurons

import nengo
import nengo.spa as spa
import numpy as np
from world import FlavourLand
from utils import RandomRun
from collections import OrderedDict
from html_plots import EncoderPlot, WeightPlot

random_inputs = False

# if values are normalized before going into the voja rule
normalize = True

items = OrderedDict({"Tree":(1,2),
            "Pond":(4,4),
            "Well":(6,1),
            "Rock":(2,7),
            "Reed":(9,3),
            "Lake":(8,8),
            "Bush":(1,3),
           })

shape = (10,10)

model = spa.SPA(seed=13)


action_vocab = spa.Vocabulary(3, randomize=False)

item_vocab = spa.Vocabulary(len(items), randomize=False)
# item_vocab = spa.Vocabulary(len(items), randomize=True)

"""
keys = np.array([[1,0,0,0],
          [0,1,0,0],
          [0,0,1,0],
          [0,0,0,1],
         ])
"""

keys = np.eye(len(items))

for i, f in enumerate(items.keys()):
    item_vocab.add(f,keys[i]) # fixed vector
    # item_vocab.parse(f) # random vector
#for f in items.keys():
#    item_vocab.add(f,len(items))

# Explore the environment and learn from it. 'Command' controls the desired position
action_vocab.add('EXPLORE', [1,0,0])

# Recall locations or items based on queries
action_vocab.add('RECALL', [0,1,0])

# Move to the location of the queried item
action_vocab.add('FIND', [0,0,1])


# compute a control signal to get to the location
def compute_velocity(x):
    # which way the agent should face to go directly to the target
    desired_ang = np.arctan2(-x[1], -x[0])
    
    ang_diff = -1*(x[2] - desired_ang)
    
    if ang_diff > np.pi:
        ang_diff -= 2*np.pi
    elif ang_diff < -np.pi:
        ang_diff += 2*np.pi

    ang_vel = ang_diff*2.5
    if np.sqrt(x[0]**2+x[1]**2) < .001:
        lin_vel = 0
        ang_vel = 0
    elif abs(ang_diff) < np.pi/4.:
        lin_vel = 1.6*np.sqrt(x[0]**2+x[1]**2)
    elif abs(ang_diff) < np.pi/2.:
        lin_vel = .8*np.sqrt(x[0]**2+x[1]**2)
    else:
        lin_vel = 0
    
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

def normalize(x):

    if np.linalg.norm(x) > 0:
        return x / np.linalg.norm(x)
    else:
        return x

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

def control_item(t, x):
    recall_mode = x[0] > .5
    if recall_mode:
        return x[5], x[6], x[7], x[8]
    else:
        return x[1], x[2], x[3], x[4]

def voja_inhib_func(t, x):
    if x < -1:
        return -1
    else:
        return x

#intercept = (np.dot(keys, keys.T) - np.eye(num_items)).flatten().max()
intercept = 0 # calculated from another script
intercept = .70 # calculated from another script
intercept_item_to_loc = .7 # calculated from another script
intercept_loc_to_item = .5 #5.55111512313e-17 # calculated from another script

with model:
    # NOTE: FlavourLand is currently hardcoded for orthogonal non-random vectors
    # TODO: make a version that can be given a vocabulary
    fl = FlavourLand(shape=shape, flavours=items, 
                     flavour_rad=.1, motion_type='teleport')
    env = nengo.Node(fl, size_in=3, size_out=3+len(items))

    taste = nengo.Ensemble(n_neurons=10*len(items), dimensions=len(items),
                           radius=1.2, neuron_type=nengo.LIF())

    """
    # linear and angular velocity
    if random_inputs:
        velocity = nengo.Node(RandomRun())
    else:
        velocity = nengo.Node([0,0])
    nengo.Connection(velocity[:2], env[:2])
    """
    nengo.Connection(env[3:], taste)

    voja_loc = nengo.Voja(post_tau=None, learning_rate=5e-2)
    voja_item = nengo.Voja(post_tau=None, learning_rate=5e-2)

    memory_loc = nengo.Ensemble(n_neurons=400, dimensions=4, intercepts=[intercept_loc_to_item]*400)
    memory_item = nengo.Ensemble(
        n_neurons=len(items)*50,
        dimensions=len(items),
        intercepts=[intercept_item_to_loc]*(len(items)*50)
    )
    
    # Query a location to get a response for what item was there
    query_location = nengo.Node([0,0])

    cfg = nengo.Config(nengo.Ensemble)
    cfg[nengo.Ensemble].neuron_type=nengo.Direct()
    with cfg:
        # Pick the item to query using a semantic pointer input
        model.query_item = spa.State(len(items), vocab=item_vocab)
        
        # Choose between learning, recall, and item finding
        model.action = spa.State(3, vocab=action_vocab)
    
    task = nengo.Node(size_in=1, size_out=1)

    # The position that the agent will try to move to using its controller
    desired_pos = nengo.Ensemble(n_neurons=200, dimensions=2, neuron_type=nengo.Direct())

    # TODO: switch from direct mode once things work
    working_loc = nengo.Ensemble(n_neurons=200, dimensions=2, neuron_type=nengo.Direct())
    working_item = nengo.Ensemble(n_neurons=200, dimensions=len(items), neuron_type=nengo.Direct())

    if normalize:
        # NOTE: loc_to_surface already normalizes
        conn_in_loc = nengo.Connection(working_loc, memory_loc, function=loc_scale_to_surface, 
                                       learning_rule_type=voja_loc, synapse=None)
        conn_in_item = nengo.Connection(working_item, memory_item, function=normalize,
                                        learning_rule_type=voja_item, synapse=None)
    else:
        conn_in_loc = nengo.Connection(working_loc, memory_loc, function=loc_scale_to_surface, 
                                       learning_rule_type=voja_loc, synapse=None)
        conn_in_item = nengo.Connection(working_item, memory_item, 
                                        learning_rule_type=voja_item, synapse=None)

    # makes sure the voja learning connection only receives 0 (learning) or -1 (not learning) exactly
    voja_inhib = nengo.Node(voja_inhib_func, size_in=1,size_out=1)

    nengo.Connection(task, voja_inhib, synapse=None, transform=-1)

    nengo.Connection(voja_inhib, conn_in_loc.learning_rule, synapse=None)
    nengo.Connection(voja_inhib, conn_in_item.learning_rule, synapse=None)

    # Try only learning when items present
    learning = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.Direct())
    inhibition = nengo.Node([-1])
    nengo.Connection(inhibition, learning, synapse=None)
    
    nengo.Connection(learning, voja_inhib, synapse=None, transform=1)

    nengo.Connection(env[3:], learning, transform=1*np.ones((1,len(items))))

    recall_item = nengo.Ensemble(n_neurons=200, dimensions=len(items), neuron_type=nengo.Direct())
    recall_loc = nengo.Ensemble(n_neurons=400, dimensions=4, neuron_type=nengo.Direct())

    conn_out_item = nengo.Connection(memory_loc, recall_item,
                                learning_rule_type=nengo.PES(1e-3),
                                function=lambda x: np.random.random(len(items))
                               )
    conn_out_loc = nengo.Connection(memory_item, recall_loc,
                                learning_rule_type=nengo.PES(1e-3),
                                function=lambda x: np.random.random(4)
                               )

    error_item = nengo.Ensemble(n_neurons=200, dimensions=len(items))
    error_loc = nengo.Ensemble(n_neurons=400, dimensions=4)

    # Connect up error populations
    nengo.Connection(env[3:], error_item, transform=-1, synapse=None)
    nengo.Connection(recall_item, error_item, transform=1, synapse=None)
    nengo.Connection(error_item, conn_out_item.learning_rule)
    
    surface_loc = nengo.Node(size_in=4,size_out=4)
    nengo.Connection(env[:2], surface_loc, function=loc_scale_to_surface, synapse=None)
    nengo.Connection(surface_loc, error_loc, transform=-1, synapse=None)
    nengo.Connection(recall_loc, error_loc, transform=1, synapse=None)
    nengo.Connection(error_loc, conn_out_loc.learning_rule)

    # inhibit learning based on learning signal
    nengo.Connection(learning, error_item.neurons, transform=[[3]] * 200, synapse=None)
    nengo.Connection(task, error_item.neurons, transform=[[-3]] * 200, synapse=None)
    
    nengo.Connection(learning, error_loc.neurons, transform=[[3]] * 400, synapse=None)
    nengo.Connection(task, error_loc.neurons, transform=[[-3]] * 400, synapse=None)

    scaled_recall_loc = nengo.Node(env_scale_node, size_in=2, size_out=2)
    nengo.Connection(recall_loc, scaled_recall_loc, function=surface_to_env)

    vel_input = nengo.Ensemble(n_neurons=200, dimensions=2, neuron_type=nengo.Direct())

    # This is only relevant when using velocity commands from a controller
    # nengo.Connection(vel_input, env[:2])

    pos_error = nengo.Ensemble(n_neurons=300, dimensions=3, neuron_type=nengo.Direct())

    nengo.Connection(env[:3], pos_error, transform=1)
    nengo.Connection(desired_pos, pos_error[:2], transform=-1)

    nengo.Connection(pos_error, vel_input, function=compute_velocity)
    
    # A user specified command for the location of the agent
    command = nengo.Node([0,0])

    # Used directly in teleport mode
    nengo.Connection(command, env[:2])
    
    def control_structure(t, x):
        # action == EXPLORE
        if x[0] > .5:
            # desired_pos=command
            # working_loc=current_loc
            # working_item=current_item
            # task=0 (learning on)
            return x[3], x[4], x[7], x[8], x[11], x[12], x[13], x[14], x[15], x[16], x[17], 0
        # action == RECALL
        elif x[1] > .5:
            # desired_pos=command
            # working_loc=query_loc_scaled
            # working_item=query_item
            # task=1 (learning off)
            return x[3], x[4], x[9], x[10], x[18], x[19], x[20], x[21], x[22], x[23], x[24], 1
        # action == FIND
        elif x[2] > .5:
            # desired_pos=scaled_recall_loc
            # working_loc=query_loc_scaled
            # working_item=query_item
            # task=1 (learning off)
            return x[5], x[6], x[9], x[10], x[18], x[19], x[20], x[21], x[22], x[23], x[24], 1
        # action == LEARN (real room value is given in this case)
        else:
            # desired_pos=command
            # working_loc=current_loc
            # working_item=current_item
            # task=0 (learning on)
            return x[3], x[4], x[7], x[8], x[11], x[12], x[13], x[14], x[15], x[16], x[17], 0
    
    bg_node = nengo.Node(control_structure, size_in=3+2+2+2+2+7+7, size_out=2+2+7+1)

    nengo.Connection(bg_node[[0,1]], desired_pos, synapse=None)
    nengo.Connection(bg_node[[2,3]], working_loc, synapse=None)
    nengo.Connection(bg_node[[4,5,6,7,8,9,10]], working_item, synapse=None)
    nengo.Connection(bg_node[11], task, synapse=None)

    nengo.Connection(model.action.output, bg_node[[0,1,2]], synapse=None)
    nengo.Connection(command, bg_node[[3,4]], synapse=None)
    nengo.Connection(scaled_recall_loc, bg_node[[5,6]], synapse=None)
    nengo.Connection(env[:2], bg_node[[7,8]], synapse=None)
    nengo.Connection(query_location, bg_node[[9,10]], synapse=None) #NOTE: this is not scaled yet
    nengo.Connection(env[3:], bg_node[[11,12,13,14,15,16,17]], synapse=None)
    nengo.Connection(model.query_item.output, bg_node[[18,19,20,21,22,23,24]], synapse=None)
    
    plot_item = EncoderPlot(conn_in_item)
    plot_loc = EncoderPlot(conn_in_loc)

def on_step(sim):
    plot_item.update(sim)
    plot_loc.update(sim)
