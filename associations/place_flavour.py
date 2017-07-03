import nengo
from world import FlavourLand

model = nengo.Network(seed=13)

flavours = {"Banana":(1,2),
            "Peach":(3,3),
            "Lemon":(4,1),
            "Apple":(2,5)
           }

with model:
    fl = FlavourLand(shape=(10,10), flavours=flavours)
    env = nengo.Node(fl, size_in=2, size_out=3+len(flavours))

    # linear and angular velocity
    velocity = nengo.Node([0,0])

    # x, y, th
    location = nengo.Ensemble(n_neurons=300, dimensions=3)

    taste = nengo.Ensemble(n_neurons=10*len(flavours), dimensions=len(flavours))

    nengo.Connection(velocity, env)

    nengo.Connection(env[:3], location)

    nengo.Connection(env[3:], taste)


