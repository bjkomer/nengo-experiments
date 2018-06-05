import nengo
import numpy as np
import argparse
import redis
from functools import partial
#from multiprocessing import Process

parser = argparse.ArgumentParser("Generate a Nengo agent to interact in a common environment")

parser.add_argument('--agent-id', type=int, default=0)
parser.add_argument('--fov', type=float, default=90, help='field of view of distance sensors, in degrees')
parser.add_argument('--n-sensors', type=int, default=5, help='number of distance sensors')
parser.add_argument('--top-speed', type=int, default=2, help='maximum speed of the agent')
parser.add_argument('--max-sensor-dist', type=int, default=5, help='maximum sensor reading')

args = parser.parse_args()

model = nengo.Network(seed=args.agent_id)

def sense_to_ang_vel(x, n_sensors):
        
    rotation_weights = np.linspace(-1, 1, n_sensors)

    return np.dot(rotation_weights, np.array(x))

class RandomVelocity(object):

    def __init__(self, top_speed=2):

        self.top_speed = top_speed
        self.current_speed = top_speed / 2.

    def __call__(self, t):

        # New speed based on a gaussian distribution around the current speed
        self.current_speed = np.random.normal(self.current_speed)
        
        # Keep the speed within the limits
        self.current_speed = np.clip(self.current_speed, 0, self.top_speed)

        return self.current_speed

class EnvironmentInterface(object):

    def __init__(self, agent_id, fov=90, n_sensors=20, max_sensor_dist=5):

        self.r = redis.StrictRedis('localhost', db=0)

        self.agent_id = agent_id
        self.fov = fov
        self.fov_rad = self.fov * np.pi / 180
        self.n_sensors = n_sensors
        self.max_sensor_dist = max_sensor_dist

        # Send the parameters of this agent to the environment
        params = np.array([self.n_sensors, self.fov_rad, self.max_sensor_dist])
        #self.r.set('agent_params:{0}'.format(self.agent_id), 
        #           np.array([self.n_sensors, self.fov_rad, self.max_sensor_dist]))

        self.r.set('agent_params:{0}'.format(self.agent_id), 
                   params.ravel().tostring())

    def __call__(self, t, x):

        #self.r.set('agent_act:{0}'.format(self.agent_id), x)
        self.r.set('agent_act:{0}'.format(self.agent_id), x.ravel().tostring())
        #state = self.r.get('agent_obs:{0}'.format(self.agent_id).decode("utf-8"))
        state = self.r.get('agent_obs:{0}'.format(self.agent_id))
        if state is None:
            state = np.zeros(3 + self.n_sensors)
        else:
            #state = np.loads(state)
            state = np.fromstring(state)

        return state

control_func = partial(sense_to_ang_vel, n_sensors=args.n_sensors)

with model:
    angular_velocity = nengo.Ensemble(n_neurons=50, dimensions=1)
    linear_velocity = nengo.Ensemble(n_neurons=50, dimensions=1)
    # TODO: make an environment interface that gathers the data for this agent
    # TODO: use redis for communication
    environment = nengo.Node(
        EnvironmentInterface(
            agent_id=args.agent_id,
            fov=args.fov,
            n_sensors=args.n_sensors,
            max_sensor_dist=args.max_sensor_dist,
        ),
        size_in=2,
        size_out=3+args.n_sensors,
    )
    
    speed_input = nengo.Node(
        RandomVelocity(top_speed=args.top_speed),
        size_in=0,
        size_out=1,
    )
    nengo.Connection(speed_input, linear_velocity)
    nengo.Connection(environment[3:], angular_velocity,
                     function=control_func)

    nengo.Connection(linear_velocity, environment[0], synapse=None)
    nengo.Connection(angular_velocity, environment[1], synapse=None)

sim = nengo.Simulator(model)

# Run until killed
while True:
    sim.run(10)
