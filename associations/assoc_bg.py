# Associate place and flavour using the Voja learning rule
# Learn the associations both ways simultaneously. Can query either location or flavour
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

flavours = OrderedDict({"Banana":(1,2),
            "Peach":(4,4),
            "Lemon":(6,1),
            "Apple":(2,7)
           })

shape = (10,10)

model = spa.SPA(seed=13)


action_vocab = spa.Vocabulary(3, randomize=False)

flavour_vocab = spa.Vocabulary(len(flavours), randomize=False)

keys = np.array([[1,0,0,0],
          [0,1,0,0],
          [0,0,1,0],
          [0,0,0,1],
         ])

for i, f in enumerate(flavours.keys()):
    flavour_vocab.add(f,keys[i])
#for f in flavours.keys():
#    flavour_vocab.add(f,len(flavours))

# Explore the environment and learn from it. 'Command' controls the desired position
action_vocab.add('EXPLORE', [1,0,0])

# Recall locations or flavours based on queries
action_vocab.add('RECALL', [0,1,0])

# Move to the location of the queried flavour
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

    #ang_vel = ang_diff*.9
    ang_vel = ang_diff*2.5
    if np.sqrt(x[0]**2+x[1]**2) < .001:
        lin_vel = 0
        ang_vel = 0
    elif abs(ang_diff) < np.pi/4.:
        #lin_vel = .6*np.sqrt(x[0]**2+x[1]**2)
        lin_vel = 1.6*np.sqrt(x[0]**2+x[1]**2)
    elif abs(ang_diff) < np.pi/2.:
        #lin_vel = .4*np.sqrt(x[0]**2+x[1]**2)
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

def control_flav(t, x):
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
intercept_flav_to_loc = .7 # calculated from another script
intercept_loc_to_flav = .5 #5.55111512313e-17 # calculated from another script

with model:
    fl = FlavourLand(shape=shape, flavours=flavours, 
                     flavour_rad=.1, motion_type='velocity')
    env = nengo.Node(fl, size_in=2, size_out=3+len(flavours))

    #TODO: put in basal ganglia that switches between learning and recall modes
    ##control_node_loc = nengo.Node(control_loc, size_in=5, size_out=2)
    control_node_flav = nengo.Node(control_flav, size_in=9, size_out=4)

    # x, y, th
    ##location = nengo.Ensemble(n_neurons=300, dimensions=3)

    taste = nengo.Ensemble(n_neurons=10*len(flavours), dimensions=len(flavours), radius=1.2)
    """
    # linear and angular velocity
    if random_inputs:
        velocity = nengo.Node(RandomRun())
    else:
        velocity = nengo.Node([0,0])
    nengo.Connection(velocity[:2], env[:2])
    """
    #nengo.Connection(env[:3], location, function=scale_location)
    ##scaled_loc = nengo.Node(size_in=3,size_out=3)
    ##nengo.Connection(env[:3], scaled_loc, function=scale_location)

    nengo.Connection(env[3:], taste)

    voja_loc = nengo.Voja(post_tau=None, learning_rate=5e-2)
    voja_flav = nengo.Voja(post_tau=None, learning_rate=5e-2)

    #memory = nengo.Ensemble(n_neurons=300, dimensions=3)
    #memory = nengo.Ensemble(n_neurons=200, dimensions=2)
    #memory = nengo.Ensemble(n_neurons=400, dimensions=4, intercepts=[intercept]*400)
    #memory = nengo.Ensemble(n_neurons=200, dimensions=2, intercepts=[intercept]*200)
    
    #memory_loc = nengo.Ensemble(n_neurons=400, dimensions=4)
    memory_loc = nengo.Ensemble(n_neurons=400, dimensions=4, intercepts=[intercept_loc_to_flav]*400)
    memory_flav = nengo.Ensemble(n_neurons=200, dimensions=4, intercepts=[intercept_flav_to_loc]*200)
    
    # Query a location to get a response for what flavour was there
    query_location = nengo.Node([0,0])
    #query_loc_scaled = nengo.Node(scale_xy_node, size_in=2, size_out=2)
    #nengo.Connection(query_location, query_loc_scaled, synapse=None)

    cfg = nengo.Config(nengo.Ensemble)
    cfg[nengo.Ensemble].neuron_type=nengo.Direct()
    with cfg:
        # Pick the flavour to query using a semantic pointer input
        model.query_flavour = spa.State(len(flavours), vocab=flavour_vocab)
        
        # Choose between learning, recall, and flavour finding
        model.action = spa.State(3, vocab=action_vocab)
    
    task = nengo.Node(size_in=1, size_out=1)

    # The position that the agent will try to move to using its controller
    desired_pos = nengo.Ensemble(n_neurons=200, dimensions=2, neuron_type=nengo.Direct())

    # TODO: switch from direct mode once things work
    working_loc = nengo.Ensemble(n_neurons=200, dimensions=2, neuron_type=nengo.Direct())


    ##nengo.Connection(task, control_node_loc[0])
    nengo.Connection(task, control_node_flav[0], synapse=None)
    
    ##nengo.Connection(scaled_loc[:2], control_node_loc[[1,2]])
    ##nengo.Connection(query_loc_scaled, control_node_loc[[3,4]])
    
    nengo.Connection(env[3:], control_node_flav[[1,2,3,4]], synapse=None)
    nengo.Connection(model.query_flavour.output, control_node_flav[[5,6,7,8]], synapse=None)

    #conn_in_loc = nengo.Connection(control_node_loc, memory_loc, function=loc_to_surface, 
    if normalize:
        # NOTE: loc_to_surface already normalizes
        conn_in_loc = nengo.Connection(working_loc, memory_loc, function=loc_scale_to_surface, 
                                       learning_rule_type=voja_loc, synapse=None)
        conn_in_flav = nengo.Connection(control_node_flav, memory_flav, function=normalize,
                                        learning_rule_type=voja_flav, synapse=None)
    else:
        conn_in_loc = nengo.Connection(working_loc, memory_loc, function=loc_scale_to_surface, 
                                       learning_rule_type=voja_loc, synapse=None)
        conn_in_flav = nengo.Connection(control_node_flav, memory_flav, 
                                        learning_rule_type=voja_flav, synapse=None)

    # makes sure the voja learning connection only receives 0 (learning) or -1 (not learning) exactly
    voja_inhib = nengo.Node(voja_inhib_func, size_in=1,size_out=1)

    nengo.Connection(task, voja_inhib, synapse=None, transform=-1)

    nengo.Connection(voja_inhib, conn_in_loc.learning_rule, synapse=None)
    nengo.Connection(voja_inhib, conn_in_flav.learning_rule, synapse=None)

    # Try only learning when flavours present
    learning = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.Direct())
    inhibition = nengo.Node([-1])
    nengo.Connection(inhibition, learning, synapse=None)
    
    nengo.Connection(learning, voja_inhib, synapse=None, transform=1)
    #nengo.Connection(learning, conn_in_loc.learning_rule, synapse=None, transform=1)
    #nengo.Connection(learning, conn_in_flav.learning_rule, synapse=None, transform=1)

    nengo.Connection(env[3:], learning, transform=1*np.ones((1,len(flavours))))

    recall_flav = nengo.Ensemble(n_neurons=200, dimensions=len(flavours), neuron_type=nengo.Direct())
    #recall_loc = nengo.Ensemble(n_neurons=400, dimensions=4, neuron_type=nengo.Direct())
    recall_loc = nengo.Ensemble(n_neurons=400, dimensions=4, neuron_type=nengo.LIF())

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

    #with nengo.Network() as control_system:
    vel_input = nengo.Ensemble(n_neurons=200, dimensions=2, neuron_type=nengo.Direct())

    nengo.Connection(vel_input, env)

    #FIXME: switch this to LIF once its working
    pos_error = nengo.Ensemble(n_neurons=300, dimensions=3, neuron_type=nengo.Direct())

    nengo.Connection(env[:3], pos_error, transform=1)
    nengo.Connection(desired_pos, pos_error[:2], transform=-1)

    nengo.Connection(pos_error, vel_input, function=compute_velocity)
    
    # A user specified command for the location of the agent
    command = nengo.Node([0,0])
    
    def control_structure(t, x):
        # action == EXPLORE
        #if max(x[:3]) == x[0]:#x[0] > .8:
        if x[0] > .5:
            #desired_pos=command, working_loc=current_loc, task=0 (learning on)
            return x[3], x[4], x[7], x[8], 0
        # action == RECALL
        #elif max(x[:3]) == x[1]:#x[1] > .8:
        elif x[1] > .5:
            # desired_pos=command, working_loc=query_loc_scaled, task=1 (learning off)
            return x[3], x[4], x[9], x[10], 1
        # action == FIND
        #elif max(x[:3]) == x[2]:#x[2] > .8:
        elif x[2] > .5:
            # desired_pos=scaled_recall_loc, working_loc=query_loc_scaled, task=1 (learning off)
            return x[5], x[6], x[9], x[10], 1
        else:
            # desired_pos=current_loc, working_loc=current_loc, task=0 (learning on)
            return x[7], x[8], x[7], x[8], 0
    
    bg_node = nengo.Node(control_structure, size_in=3+2+2+2+2, size_out=2+2+1)

    nengo.Connection(bg_node[[0,1]], desired_pos, synapse=None)
    nengo.Connection(bg_node[[2,3]], working_loc, synapse=None)
    nengo.Connection(bg_node[4], task, synapse=None)

    nengo.Connection(model.action.output, bg_node[[0,1,2]], synapse=None)
    nengo.Connection(command, bg_node[[3,4]], synapse=None)
    nengo.Connection(scaled_recall_loc, bg_node[[5,6]], synapse=None)
    nengo.Connection(env[:2], bg_node[[7,8]], synapse=None)
    #nengo.Connection(query_loc_scaled, bg_node[[9,10]], synapse=None)
    nengo.Connection(query_location, bg_node[[9,10]], synapse=None) #NOTE: this is not scaled yet
    plot_flav = EncoderPlot(conn_in_flav)
    plot_loc = EncoderPlot(conn_in_loc)

def on_step(sim):
    plot_flav.update(sim)
    plot_loc.update(sim)

# Do some data recording and save encoders/decoders here
if __name__ == '__main__':
    with model:
        pass
    print("test")
