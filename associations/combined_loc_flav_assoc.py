import nengo
import nengo.spa as spa
import numpy as np
from world import FlavourLand, FlavourLandMultiRoom
from utils import RandomRun
from collections import OrderedDict

from html_plots import EncoderPlot

#TODO: convert space into toroid

automatic_inputs = True

flavours_A = OrderedDict({"Banana":(1,2),
            "Peach":(4,4),
            "Lemon":(6,1),
            "Apple":(2,7)
           })

flavours_B = OrderedDict({"Banana":(8,9),
            "Peach":(6,6),
            "Lemon":(1,3),
            "Apple":(3,4)
           })

rooms = [flavours_A, flavours_B]

shape = (10,10)

D = 2#16

mem_dim = 4#2 #4

normalize = True

action_vocab = spa.Vocabulary(3, randomize=False)

flavour_vocab = spa.Vocabulary(len(flavours_A), randomize=False)

#place_vocab = spa.Vocabulary(D)
place_vocab = spa.Vocabulary(2, randomize=False)

place_vocab.add('A', [1,0])
place_vocab.add('B', [0,1])

for f in flavours_A.keys():
    flavour_vocab.add(f,len(flavours_A))

model = spa.SPA(seed=13)

# Scales the location output to be between -1 and 1
def scale_location_xy(x):
    x_out = x[0]/(shape[0]/2.) - 1
    y_out = x[1]/(shape[1]/2.) - 1

    return [x_out, y_out]

def room_vec_to_scalar(t, x):
    for i, val in enumerate(x):
        if val > .5:
            return i+.1
    return 0

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

# Move through all the the inputs to train the network automatically
class CycleInputs(object):

    def __init__(self, rooms, period=.5, dt=.001):

        self.rooms = rooms
        self.num_rooms = len(rooms)
        self.num_flavours = len(rooms[0])
        self.count = 0
        self.switch = int(round(period/dt))
        self.cur_room = 0
        self.cur_flav = 0

        self.x = 0
        self.y = 0
        self.th = 0
        self.f = [0]*self.num_flavours
        self.room_vec = [0]*self.num_rooms

    def __call__(self, t):

        self.count += 1
        if self.count == self.switch:
            
            flav = self.rooms[self.cur_room].values()[self.cur_flav]
            
            self.x = flav[0]
            self.y = flav[1]

            self.f = [0]*self.num_flavours
            self.f[self.cur_flav] = 1
            self.room_vec = [0]*self.num_rooms
            self.room_vec[self.cur_room] = 1

            self.count = 0
            self.cur_flav += 1
            if self.cur_flav == self.num_flavours:
                self.cur_flav = 0
                self.cur_room += 1
                if self.cur_room == self.num_rooms:
                    self.cur_room = 0

        return [self.x, self.y] + self.room_vec

def normalize(x):

    if np.linalg.norm(x) > 0:
        return x / np.linalg.norm(x)
    else:
        return x

intercept = .70
with model:
    #fl = FlavourLand(shape=shape, flavours=flavours, 
    #                 flavour_rad=.1, motion_type='teleport')
    fl = FlavourLandMultiRoom(shape=shape, room=rooms, 
                              flavour_rad=.1, motion_type='teleport')
    env = nengo.Node(fl, size_in=4, size_out=3+len(flavours_A))



    """
    cfg = nengo.Config(nengo.Ensemble)
    cfg[nengo.Ensemble].neuron_type=nengo.Direct()
    with cfg:
        # Pick the flavour to query using a semantic pointer input
        model.flavour_input = spa.State(len(flavours), vocab=flavour_vocab)
    """

    model.room_recall = spa.State(D, vocab=place_vocab)
    
    cfg = nengo.Config(nengo.Ensemble)
    cfg[nengo.Ensemble].neuron_type=nengo.Direct()
    with cfg:
        # Pick the flavour to query using a semantic pointer input
        model.current_room = spa.State(D, vocab=place_vocab)
    
    if automatic_inputs:
        pos_input = nengo.Node(CycleInputs(rooms=rooms, period=.75))
        nengo.Connection(pos_input[[0,1]], env[[0,1]])
        nengo.Connection(pos_input[2:], model.current_room.input)
    else:
        pos_input = nengo.Node([0,0])
        nengo.Connection(pos_input, env[[0,1]])

    vec_to_scalar = nengo.Node(room_vec_to_scalar, size_in=D, size_out=1)

    #nengo.Connection(model.current_room.output, env[-1], function=room_vec_to_scalar)
    nengo.Connection(model.current_room.output, vec_to_scalar, synapse=None)
    nengo.Connection(vec_to_scalar, env[-1], synapse=None)

    if mem_dim == 2:
        multimodal = nengo.Ensemble(n_neurons=500, dimensions=2+len(flavours_A), neuron_type=nengo.Direct())
        memory = nengo.Ensemble(n_neurons=500, dimensions=2+len(flavours_A), intercepts=[intercept]*500)
        nengo.Connection(env[[0,1]], multimodal[[0,1]], function=scale_location_xy)
        nengo.Connection(env[3:], multimodal[2:])
    elif mem_dim == 4:
        multimodal = nengo.Ensemble(n_neurons=500, dimensions=4+len(flavours_A), neuron_type=nengo.Direct())
        memory = nengo.Ensemble(n_neurons=500, dimensions=4+len(flavours_A), intercepts=[intercept]*500)
        #memory = nengo.Ensemble(n_neurons=500, dimensions=4+len(flavours_A), intercepts=[intercept]*250+[-intercept]*250)
        #memory = nengo.Ensemble(n_neurons=500, dimensions=4+len(flavours_A))
        nengo.Connection(env[[0,1]], multimodal[[0,1,2,3]], function=loc_scale_to_surface)
        nengo.Connection(env[3:], multimodal[4:])


    #nengo.Connection(model.flavour_input.output, multimodal[2:])

    voja = nengo.Voja(post_tau=None, learning_rate=5e-2)

    if normalize:
        conn_in = nengo.Connection(multimodal, memory, learning_rule_type=voja, function=normalize)
    else:
        conn_in = nengo.Connection(multimodal, memory, learning_rule_type=voja)

    conn_out = nengo.Connection(memory, model.room_recall.input, learning_rule_type=nengo.PES(1e-3),
                     function=lambda x: np.zeros(D))

    error = nengo.Ensemble(n_neurons=500, dimensions=D)

    nengo.Connection(model.room_recall.output, error)
    nengo.Connection(model.current_room.output, error, transform=-1)

    nengo.Connection(error, conn_out.learning_rule)

    # learning is off when negative
    #learning = nengo.Node([0])
    learning = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.Direct())
    inhibition = nengo.Node([-1])
    nengo.Connection(inhibition, learning, synapse=None)
    nengo.Connection(env[3:], learning, transform=1*np.ones((1,len(flavours_A))), synapse=None)

    nengo.Connection(learning, error.neurons, transform=[[10]]*500, synapse=None)
    nengo.Connection(learning, conn_in.learning_rule, synapse=None)
    
    # 0 is learning, 1 is recall only
    task = nengo.Node([0])
    nengo.Connection(task, conn_in.learning_rule, synapse=None, transform=-1)
    nengo.Connection(task, error.neurons, transform=[[-10]] * 500, synapse=None)

    
    #plot = EncoderPlot(conn_in, scaling='normalize')
    plot = EncoderPlot(conn_in, scaling='max')

def on_step(sim):
    plot.update(sim)

if __name__ == '__main__':
    pass
