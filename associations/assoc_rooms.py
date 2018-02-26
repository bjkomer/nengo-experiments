# Associate place and flavour using the Voja learning rule
# Learn the associations both ways simultaneously. Can query either location or flavour
# also learn the associations dependent on the current room, this will provide context for the recall
import nengo
import nengo.spa as spa
import numpy as np
from world import FlavourLand, FlavourLandMultiRoom
from utils import *
from collections import OrderedDict
from html_plots import EncoderPlot, WeightPlot
import cPickle as pickle

#TODO: get this to work properly in the future

# If all inputs will cycle
automatic_inputs = False#True

#random_inputs = False

# Load in saved encoders and decoders
load_enc_dec = True

if load_enc_dec:
    enc_dec = pickle.load(open('room_enc_dec.pkl','r'))

    enc_flav = enc_dec['p_enc_flav']
    enc_loc = enc_dec['p_enc_loc']
    enc_room = enc_dec['p_enc_room']
    
    dec_flav = enc_dec['p_dec_flav']
    dec_loc = enc_dec['p_dec_loc']
    dec_room = enc_dec['p_dec_room']

    dec_flav_solver = SpecifyDecoders(dec_flav)
    dec_loc_solver = SpecifyDecoders(dec_loc)
    dec_room_solver = SpecifyDecoders(dec_room)

# If control is teleport style
teleport = True #False

# if keys are normalized before going into the voja rule
normalize_keys = True

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

flavours_C = OrderedDict({"Banana":(3,1),
            "Peach":(8,2),
            "Lemon":(7,6),
            "Apple":(6,1)
           })

rooms = [flavours_A, flavours_B, flavours_C]

shape = (10,10)

model = spa.SPA(seed=13)


action_vocab = spa.Vocabulary(3, randomize=False)

flavour_vocab = spa.Vocabulary(len(flavours_A), randomize=False)

room_vocab = spa.Vocabulary(len(rooms), randomize=False)

room_vocab.add('A', [1,0,0])
room_vocab.add('B', [0,1,0])
room_vocab.add('C', [0,0,1])

keys = np.array([[1,0,0,0],
          [0,1,0,0],
          [0,0,1,0],
          [0,0,0,1],
         ])

for i, f in enumerate(flavours_A.keys()):
    flavour_vocab.add(f,keys[i])

# Explore the environment and learn from it. 'Command' controls the desired position
action_vocab.add('EXPLORE', [1,0,0])

# Recall locations or flavours based on queries
action_vocab.add('RECALL', [0,1,0])

# Move to the location of the queried flavour
action_vocab.add('FIND', [0,0,1])


#intercept = (np.dot(keys, keys.T) - np.eye(num_items)).flatten().max()
intercept = 0 # calculated from another script
intercept = .70 # calculated from another script
intercept_flav_to_loc = .7 # calculated from another script
intercept_loc_to_flav = .5 #5.55111512313e-17 # calculated from another script
intercept_to_room = .7

with model:
    if teleport:
        fl = FlavourLandMultiRoom(shape=shape, room=rooms, 
                         flavour_rad=.1, motion_type='teleport')
        env = nengo.Node(fl, size_in=4, size_out=3+len(flavours_A))
    else:
        fl = FlavourLandMultiRoom(shape=shape, room=rooms, 
                         flavour_rad=.1, motion_type='velocity')
        env = nengo.Node(fl, size_in=3, size_out=3+len(flavours_A))

    taste = nengo.Ensemble(n_neurons=10*len(flavours_A), dimensions=len(flavours_A), radius=1.2)

    #nengo.Connection(env[:3], location, function=scale_location)
    scaled_loc = nengo.Node(size_in=3,size_out=3)
    nengo.Connection(env[:3], scaled_loc, function=scale_location, synapse=None)

    nengo.Connection(env[3:], taste)


    # contains location, flavour, and room information
    multimodal = nengo.Ensemble(n_neurons=500, 
                                dimensions=4+len(flavours_A)+len(rooms), 
                                neuron_type=nengo.Direct())
    
    if load_enc_dec:
        # room + location to flavour
        memory_rl_to_f = nengo.Ensemble(n_neurons=700, dimensions=4+len(rooms),
                                        encoders=enc_flav,
                                        intercepts=[intercept_loc_to_flav]*700)
        # room + flavour to location
        memory_rf_to_l = nengo.Ensemble(n_neurons=700, dimensions=len(flavours_A)+len(rooms), 
                                        encoders=enc_loc,
                                        intercepts=[intercept_flav_to_loc]*700)
        # location + flavour to room
        memory_lf_to_r = nengo.Ensemble(n_neurons=700, dimensions=4+len(flavours_A), 
                                        encoders=enc_room,
                                        intercepts=[intercept_to_room]*700)
    else:
        # room + location to flavour
        memory_rl_to_f = nengo.Ensemble(n_neurons=700, dimensions=4+len(rooms), 
                                        intercepts=[intercept_loc_to_flav]*700)
        # room + flavour to location
        memory_rf_to_l = nengo.Ensemble(n_neurons=700, dimensions=len(flavours_A)+len(rooms), 
                                        intercepts=[intercept_flav_to_loc]*700)
        # location + flavour to room
        memory_lf_to_r = nengo.Ensemble(n_neurons=700, dimensions=4+len(flavours_A), 
                                        intercepts=[intercept_to_room]*700)
    
    # Query a location to get a response for what flavour was there
    query_location = nengo.Node([0,0])
    query_loc_scaled = nengo.Node(scale_xy_node, size_in=2, size_out=2)
    nengo.Connection(query_location, query_loc_scaled, synapse=None)

    cfg = nengo.Config(nengo.Ensemble)
    cfg[nengo.Ensemble].neuron_type=nengo.Direct()
    with cfg:
        # Pick the flavour to query using a semantic pointer input
        model.query_flavour = spa.State(len(flavours_A), vocab=flavour_vocab)
        
        # Choose between learning, recall, and flavour finding
        model.action = spa.State(3, vocab=action_vocab)
        
        # The ground-truth current room (fed into the environment)
        model.current_room = spa.State(len(rooms), vocab=room_vocab)
    
    vec_to_scalar = nengo.Node(room_vec_to_scalar, size_in=len(rooms), size_out=1)
    nengo.Connection(model.current_room.output, vec_to_scalar, synapse=None)
    nengo.Connection(vec_to_scalar, env[-1], synapse=None)
    
    # Used for inhibiting learning connections based on the task (EXPLORE, RECALL, FIND)
    task = nengo.Node(size_in=1, size_out=1)

    # The position that the agent will try to move to using its controller
    desired_pos = nengo.Ensemble(n_neurons=200, dimensions=2, neuron_type=nengo.Direct())

    # TODO: switch from direct mode once things work
    working_loc = nengo.Ensemble(n_neurons=200, dimensions=2, neuron_type=nengo.Direct())
    working_flav = nengo.Ensemble(n_neurons=200, dimensions=4, neuron_type=nengo.Direct())
    working_room = nengo.Ensemble(n_neurons=200, dimensions=3, neuron_type=nengo.Direct())
    
    nengo.Connection(working_loc, multimodal[[0,1,2,3]], function=loc_scale_to_surface, synapse=None)
    nengo.Connection(working_flav, multimodal[[4,5,6,7]], synapse=None)
    nengo.Connection(working_room, multimodal[[8,9,10]], synapse=None)

    voja_loc = nengo.Voja(post_tau=None, learning_rate=5e-2)
    voja_flav = nengo.Voja(post_tau=None, learning_rate=5e-2)
    voja_room = nengo.Voja(post_tau=None, learning_rate=5e-2)
    
    if normalize_keys:
        conn_in_loc = nengo.Connection(multimodal[[4,5,6,7,8,9,10]], memory_rf_to_l, function=normalize,
                                       learning_rule_type=voja_loc, synapse=None)
        conn_in_flav = nengo.Connection(multimodal[[0,1,2,3,8,9,10]], memory_rl_to_f, function=normalize,
                                       learning_rule_type=voja_flav, synapse=None)
        conn_in_room = nengo.Connection(multimodal[[0,1,2,3,4,5,6,7]], memory_lf_to_r, function=normalize,
                                       learning_rule_type=voja_room, synapse=None)
    else:
        conn_in_loc = nengo.Connection(multimodal[[4,5,6,7,8,9,10]], memory_rf_to_l,
                                       learning_rule_type=voja_loc, synapse=None)
        conn_in_flav = nengo.Connection(multimodal[[0,1,2,3,8,9,10]], memory_rl_to_f,
                                       learning_rule_type=voja_flav, synapse=None)
        conn_in_room = nengo.Connection(multimodal[[0,1,2,3,4,5,6,7]], memory_lf_to_r,
                                       learning_rule_type=voja_room, synapse=None)

    # makes sure the voja learning connection only receives 0 (learning) or -1 (not learning) exactly
    voja_inhib = nengo.Node(voja_inhib_func, size_in=1,size_out=1)

    nengo.Connection(task, voja_inhib, synapse=None, transform=-1)

    nengo.Connection(voja_inhib, conn_in_loc.learning_rule, synapse=None)
    nengo.Connection(voja_inhib, conn_in_flav.learning_rule, synapse=None)
    nengo.Connection(voja_inhib, conn_in_room.learning_rule, synapse=None)

    # Try only learning when flavours present
    learning = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.Direct())
    inhibition = nengo.Node([-1])
    nengo.Connection(inhibition, learning, synapse=None)
    
    nengo.Connection(learning, voja_inhib, synapse=None, transform=1)

    nengo.Connection(env[3:], learning, transform=1*np.ones((1,len(flavours_A))))

    recall_flav = nengo.Ensemble(n_neurons=200, dimensions=len(flavours_A), neuron_type=nengo.Direct())
    recall_loc = nengo.Ensemble(n_neurons=400, dimensions=4, neuron_type=nengo.Direct()) #neuron_type=nengo.LIF())
    recall_room = nengo.Ensemble(n_neurons=150, dimensions=len(rooms), neuron_type=nengo.Direct())

    # The room that the agent believes it is in based on exploration
    model.est_room = spa.State(len(rooms), vocab=room_vocab, feedback=1)
    nengo.Connection(recall_room, model.est_room.input) #TODO: make sure this is correct
 
    if load_enc_dec:
        conn_out_flav = nengo.Connection(memory_rl_to_f.neurons, recall_flav,
                                    learning_rule_type=nengo.PES(1e-3),
                                    #solver=dec_flav_solver,
                                    transform=dec_flav,
                                    #function=lambda x: np.random.random(len(flavours_A))
                                   )
        conn_out_loc = nengo.Connection(memory_rf_to_l.neurons, recall_loc,
                                    learning_rule_type=nengo.PES(1e-3),
                                    #solver=dec_loc_solver,
                                    transform=dec_loc,
                                    #function=lambda x: np.random.random(4)
                                   )
        conn_out_room = nengo.Connection(memory_lf_to_r.neurons, recall_room,
                                    learning_rule_type=nengo.PES(1e-3),
                                    #solver=dec_room_solver,
                                    transform=dec_room,
                                    #function=lambda x: np.random.random(len(rooms))
                                   )
    else:
        conn_out_flav = nengo.Connection(memory_rl_to_f, recall_flav,
                                    learning_rule_type=nengo.PES(1e-3),
                                    function=lambda x: np.random.random(len(flavours_A))
                                   )
        conn_out_loc = nengo.Connection(memory_rf_to_l, recall_loc,
                                    learning_rule_type=nengo.PES(1e-3),
                                    function=lambda x: np.random.random(4)
                                   )
        conn_out_room = nengo.Connection(memory_lf_to_r, recall_room,
                                    learning_rule_type=nengo.PES(1e-3),
                                    function=lambda x: np.random.random(len(rooms))
                                   )

    error_flav = nengo.Ensemble(n_neurons=200, dimensions=len(flavours_A))
    error_loc = nengo.Ensemble(n_neurons=400, dimensions=4)
    error_room = nengo.Ensemble(n_neurons=150, dimensions=len(rooms))

    # Connect up error populations
    nengo.Connection(env[3:], error_flav, transform=-1, synapse=None)
    nengo.Connection(recall_flav, error_flav, transform=1, synapse=None)
    nengo.Connection(error_flav, conn_out_flav.learning_rule)
    
    surface_loc = nengo.Node(size_in=4,size_out=4)
    nengo.Connection(env[:2], surface_loc, function=loc_scale_to_surface, synapse=None)
    nengo.Connection(surface_loc, error_loc, transform=-1, synapse=None)
    nengo.Connection(recall_loc, error_loc, transform=1, synapse=None)
    nengo.Connection(error_loc, conn_out_loc.learning_rule)

    nengo.Connection(recall_room, error_room, synapse=None)
    nengo.Connection(model.current_room.output, error_room, transform=-1, synapse=None)
    nengo.Connection(error_room, conn_out_room.learning_rule)

    # inhibit learning based on learning signal
    nengo.Connection(learning, error_flav.neurons, transform=[[3]] * 200, synapse=None)
    nengo.Connection(task, error_flav.neurons, transform=[[-3]] * 200, synapse=None)
    
    nengo.Connection(learning, error_loc.neurons, transform=[[3]] * 400, synapse=None)
    nengo.Connection(task, error_loc.neurons, transform=[[-3]] * 400, synapse=None)
    
    nengo.Connection(learning, error_room.neurons, transform=[[3]] * 150, synapse=None)
    nengo.Connection(task, error_room.neurons, transform=[[-3]] * 150, synapse=None)
    
    # If any action besides the default 'LEARN' is chosen, inhibit the error population
    nengo.Connection(model.action.output, error_room.neurons, transform=[[-3,-3,-3]] * 150, synapse=None)
    nengo.Connection(model.action.output, voja_inhib, transform=[[-1,-1,-1]], synapse=None)

    scaled_recall_loc = nengo.Node(env_scale_node, size_in=2, size_out=2)
    nengo.Connection(recall_loc, scaled_recall_loc, function=surface_to_env)

    vel_input = nengo.Ensemble(n_neurons=200, dimensions=2, neuron_type=nengo.Direct())


    #FIXME: switch this to LIF once its working
    pos_error = nengo.Ensemble(n_neurons=300, dimensions=3, neuron_type=nengo.Direct())

    nengo.Connection(env[:3], pos_error, transform=1)
    nengo.Connection(desired_pos, pos_error[:2], transform=-1)

    nengo.Connection(pos_error, vel_input, function=compute_velocity)
    
    # A user specified command for the location of the agent
    command = nengo.Node([0,0])
    
    def control_structure(t, x):
        # action == EXPLORE
        if x[0] > .5:
            # desired_pos=command
            # working_loc=current_loc
            # working_flav=current_flav
            # working_room=est_room
            # task=0 (learning on)
            return x[3], x[4], x[7], x[8], x[11], x[12], x[13], x[14], x[22], x[23], x[24], 0
        # action == RECALL
        elif x[1] > .5:
            # desired_pos=command
            # working_loc=query_loc_scaled
            # working_flav=query_flavour
            # working_room=est_room??
            # task=1 (learning off)
            return x[3], x[4], x[9], x[10], x[15], x[16], x[17], x[18], x[22], x[23], x[24], 1
            #return x[3], x[4], x[9], x[10], x[15], x[16], x[17], x[18], x[19], x[20], x[21], 1
        # action == FIND
        elif x[2] > .5:
            # desired_pos=scaled_recall_loc
            # working_loc=query_loc_scaled
            # working_flav=query_flavour
            # working_room=est_room
            # task=1 (learning off)
            return x[5], x[6], x[9], x[10], x[15], x[16], x[17], x[18], x[22], x[23], x[24], 1
        # action == LEARN (real room value is given in this case)
        else:
            # desired_pos=command
            # working_loc=current_loc
            # working_flav=current_flav
            # working_room=current_room
            # task=0 (learning on)
            return x[3], x[4], x[7], x[8], x[11], x[12], x[13], x[14], x[19], x[20], x[21], 0
    
    bg_node = nengo.Node(control_structure, size_in=3+2+2+2+2+4+4+3+3, size_out=2+2+4+3+1)

    nengo.Connection(bg_node[[0,1]], desired_pos, synapse=None)
    nengo.Connection(bg_node[[2,3]], working_loc, synapse=None)
    nengo.Connection(bg_node[[4,5,6,7]], working_flav, synapse=None)
    nengo.Connection(bg_node[[8,9,10]], working_room, synapse=None)
    nengo.Connection(bg_node[11], task, synapse=None)

    nengo.Connection(model.action.output, bg_node[[0,1,2]], synapse=None)
    nengo.Connection(command, bg_node[[3,4]], synapse=None)
    nengo.Connection(scaled_recall_loc, bg_node[[5,6]], synapse=None)
    nengo.Connection(env[:2], bg_node[[7,8]], synapse=None)
    #nengo.Connection(query_loc_scaled, bg_node[[9,10]], synapse=None)
    nengo.Connection(query_location, bg_node[[9,10]], synapse=None) #NOTE: this is not scaled yet
    nengo.Connection(env[3:], bg_node[[11,12,13,14]], synapse=None)
    nengo.Connection(model.query_flavour.output, bg_node[[15,16,17,18]], synapse=None)
    nengo.Connection(model.current_room.output, bg_node[[19,20,21]], synapse=None)
    nengo.Connection(model.est_room.output, bg_node[[22,23,24]], synapse=None)

    if automatic_inputs:
        pos_input = nengo.Node(CycleInputs(rooms=rooms, period=.75))
        nengo.Connection(pos_input[[0,1]], env[[0,1]])
        nengo.Connection(pos_input[2:], model.current_room.input)
    else:
        if teleport:
            nengo.Connection(command, env[[0,1]])
        else:
            nengo.Connection(vel_input, env[[0,1]])


    plot_flav = EncoderPlot(conn_in_flav)
    plot_loc = EncoderPlot(conn_in_loc)
    plot_room = EncoderPlot(conn_in_room)

def on_step(sim):
    plot_flav.update(sim)
    plot_loc.update(sim)
    plot_room.update(sim)

# Do some data recording and save encoders/decoders here
if __name__ == '__main__':
    with model:
        p_enc_flav = nengo.Probe(memory_rl_to_f, 'scaled_encoders')
        p_enc_loc = nengo.Probe(memory_rf_to_l, 'scaled_encoders')
        p_enc_room = nengo.Probe(memory_lf_to_r, 'scaled_encoders')
        
        p_dec_flav = nengo.Probe(conn_out_flav, 'weights')
        p_dec_loc = nengo.Probe(conn_out_loc, 'weights')
        p_dec_room = nengo.Probe(conn_out_room, 'weights')
        #p_dec_flav = nengo.Probe(memory_rl_to_f, 'weights')
        #p_dec_loc = nengo.Probe(memory_rf_to_l, 'weights')
        #p_dec_room = nengo.Probe(memory_lf_to_r, 'weights')

    print("Starting Simulator")
    with nengo.Simulator(model) as sim:
        sim.run(.75*4*3*4)

    print("Saving Encoders and Decoders")
    pickle.dump({'p_enc_flav':sim.data[p_enc_flav][-1].copy(),
                 'p_enc_loc':sim.data[p_enc_loc][-1].copy(),
                 'p_enc_room':sim.data[p_enc_room][-1].copy(),
                 'p_dec_flav':sim.data[p_dec_flav][-1].copy(),
                 'p_dec_loc':sim.data[p_dec_loc][-1].copy(),
                 'p_dec_room':sim.data[p_dec_room][-1].copy(),
                },open('room_enc_dec.pkl','w'))

