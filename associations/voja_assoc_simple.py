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

#intercept = (np.dot(keys, keys.T) - np.eye(num_items)).flatten().max()
intercept = .001 # arbitrary for now

with model:
    fl = FlavourLand(shape=shape, flavours=flavours, motion_type='teleport')
    env = nengo.Node(fl, size_in=3, size_out=3+len(flavours))

    #TODO: put in basal ganglia that switches between learning and recall modes
    control_node = nengo.Node(control, size_in=5, size_out=2)

    # x, y, th
    location = nengo.Ensemble(n_neurons=300, dimensions=3)

    taste = nengo.Ensemble(n_neurons=10*len(flavours), dimensions=len(flavours))

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

    voja = nengo.Voja(post_tau=None, learning_rate=5e-2)

    #memory = nengo.Ensemble(n_neurons=300, dimensions=3)
    #memory = nengo.Ensemble(n_neurons=200, dimensions=2)
    memory = nengo.Ensemble(n_neurons=400, dimensions=4)
    #memory = nengo.Ensemble(n_neurons=200, dimensions=2, intercepts=[intercept]*200)
    
    # Query a location to get a response for what flavour was there
    query_location = nengo.Node([0,0])
    query_loc_scaled = nengo.Node(scale_xy_node, size_in=2, size_out=2)
    nengo.Connection(query_location, query_loc_scaled, synapse=None)
    
    # Choose between learning and recall #TODO: use basal ganglia instead
    task = nengo.Node([0])

    nengo.Connection(task, control_node[0])
    #nengo.Connection(env[:2], control_node[[1,2]])
    nengo.Connection(scaled_loc[:2], control_node[[1,2]])
    nengo.Connection(query_loc_scaled, control_node[[3,4]])

    conn_in = nengo.Connection(control_node, memory, function=loc_to_surface, learning_rule_type=voja,
                               synapse=None)
    nengo.Connection(task, conn_in.learning_rule, synapse=None, transform=-1)
    
    #conn_in = nengo.Connection(env[:3], memory, learning_rule_type=voja)
    ##conn_in = nengo.Connection(env[:2], memory, learning_rule_type=voja)

    # Try only learning when flavours present
    #learning = nengo.Ensemble(n_neurons=100, dimensions=1)
    learning = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.Direct())
    inhibition = nengo.Node([-1])
    nengo.Connection(inhibition, learning)
    nengo.Connection(learning, conn_in.learning_rule, synapse=None, transform=1)

    nengo.Connection(env[3:], learning, transform=1*np.ones((1,len(flavours))))

    nengo.Connection(learning, conn_in.learning_rule)

    recall = nengo.Ensemble(n_neurons=200, dimensions=len(flavours))

    conn_out = nengo.Connection(memory, recall,
                                learning_rule_type=nengo.PES(1e-3),
                                function=lambda x: np.random.random(len(flavours))
                                #function=lambda x: np.zeros(len(flavours))
                               )

    error = nengo.Ensemble(n_neurons=200, dimensions=len(flavours))

    nengo.Connection(env[3:], error, transform=-1, synapse=None)
    nengo.Connection(recall, error, transform=1, synapse=None)
    nengo.Connection(error, conn_out.learning_rule)

    # inhibit learning based on learning signal
    nengo.Connection(learning, error.neurons, transform=[[10]] * 200, synapse=None)
    nengo.Connection(task, error.neurons, transform=[[-10]] * 200, synapse=None)


    ##nengo.Connection(query, memory)

