import nengo
import redis
import scipy.ndimage
import numpy as np
from gym_maze.envs.generators import RandomBlockMazeGenerator

# TODO: make env class that can have arbitrary number of agents
#       no size in or size out, just for visualization
#       make a different colour for each agent
# TODO: use redis for easy communication, start the server in the environment?

model = nengo.Network(seed=13)

def generate_sensor_readings(map_arr,
                             zoom_level=4,
                             n_sensors=30,
                             fov_rad=np.pi,
                             x=0,
                             y=0,
                             th=0,
                             max_sensor_dist=10,
                            ):
    """
    Given a map, agent location in the map, number of sensors, field of view
    calculate the distance readings of each sensor to the nearest obstacle
    uses supersampling to find the approximate collision points
    """
    arr_zoom = scipy.ndimage.zoom(map_arr, zoom_level, order=0)
    dists = np.zeros((n_sensors,))

    angs = np.linspace(-fov_rad / 2. + th, fov_rad / 2. + th, n_sensors)

    for i, ang in enumerate(angs):
        dists[i] = get_collision_coord(arr_zoom, x*zoom_level, y*zoom_level, ang, max_sensor_dist*zoom_level) / zoom_level

    return dists

def get_collision_coord(map_array, x, y, th,
                        max_sensor_dist=10*4,
                       ):
    """
    Find the first occupied space given a start point and direction
    Meant for a zoomed in map_array
    """
    # Define the step sizes
    dx = np.cos(th)
    dy = np.sin(th)

    # Initialize to starting point
    cx = x
    cy = y

    for i in range(max_sensor_dist):
        # Move one unit in the direction of the sensor
        cx += dx
        cy += dy
        if map_array[int(cx), int(cy)] == 1:
            return i

    return max_sensor_dist

# TODO: add some way of removing agents
class MultiAgentEnvironment(object):

    def __init__(self, size=20, show_distances=False, dt=0.001):

        self.size = size

        self.show_distances = show_distances

        self.dt = dt

        self._nengo_html_ = ''

        self.current_seed = 0

        self.n_agents = 0

        self.agent_ids = []

        # Parameters of each agent, with ids as keys
        self.agent_params = {}
        
        # Position and orientation of each agent, along with distance measurements
        self.agent_obs = {}
        
        # Inputs/Actions for the agent (linear and angular velocity)
        self.agent_act = {}

        self.r = redis.StrictRedis('localhost', db=0)
        
        # Set up svg element templates to be filled in later
        self.tile_template = '<rect x={0} y={1} width=1 height=1 style="fill:black"/>'
        self.agent_template = '<polygon points="0.25,0.25 -0.25,0.25 0,-0.5" style="fill:rgb({3},{4},{5})" transform="translate({0},{1}) rotate({2})"/>'
        self.sensor_template = '<line x1="{0}" y1="{1}" x2="{2}" y2="{3}" style="stroke:rgb(128,128,128);stroke-width:.1"/>'
        # TODO: check which order height and width should be
        self.svg_header = '<svg width="100%%" height="100%%" viewbox="0 0 {0} {1}">'.format(self.size, self.size)
        
        self._generate_map()
        self._generate_svg()
    
    def colour_from_id(self, agent_id):

        # TODO: implement this function
        return 128, 128, 128

    def _generate_map(self):
        """
        Generate a new map based on the current seed
        """
        np.random.seed(self.current_seed)

        maze = RandomBlockMazeGenerator(maze_size=self.size - 2, # the -2 is because the outer wall gets added
                                        obstacle_ratio=.2,
                                       )
        self.map = maze.maze
    
    def _generate_svg(self):
        
        # TODO: make sure coordinates are correct (e.g. inverted y axis)
        # NOTE: x and y currently flipped from: https://github.com/tcstewar/ccmsuite/blob/master/ccm/ui/nengo.py
        # draw tiles
        tiles = []
        for i in range(self.size):
            for j in range(self.size):
                # For simplicity and efficiency, only draw the walls and not the empty space
                # This will have to change when tiles can have different colours
                if self.map[i, j] == 1:
                    tiles.append(self.tile_template.format(i, j))

        svg = self.svg_header
        
        svg += ''.join(tiles)

        for agent_id in self.agent_ids:
            #act = np.loads(self.r.get('agent_act:{0}'.format(agent_id)).decode("utf-8"))
            act = np.fromstring(self.r.get('agent_act:{0}'.format(agent_id)))
            lin_vel = act[0]
            ang_vel = act[1]
            
            n_sensors = int(self.agent_params[agent_id][0])
            fov_rad = self.agent_params[agent_id][1]
            max_sensor_dist = int(self.agent_params[agent_id][2]) # TODO: allow this to be non-integer

            #self.th += v[2] * self.dt
            #self.x += np.cos(self.th) * v[0] * self.dt 
            #self.y += np.sin(self.th) * v[0] * self.dt 
            self.agent_obs[agent_id][2] += ang_vel * self.dt
            th = self.agent_obs[agent_id][2]
            self.agent_obs[agent_id][0] += np.cos(th) * lin_vel * self.dt 
            self.agent_obs[agent_id][1] += np.sin(th) * lin_vel * self.dt
            x = self.agent_obs[agent_id][0]
            y = self.agent_obs[agent_id][1]

            # draw agent
            direction = th * 180. / np.pi + 90. #TODO: make sure angle conversion is correct
            r, g, b = self.colour_from_id(agent_id)
            agent_svg = self.agent_template.format(x, y, direction, r, g, b)

            # draw distance sensors
            lines = []
            sensor_dists = generate_sensor_readings(
                map_arr=self.map,
                zoom_level=8,
                n_sensors=n_sensors,
                fov_rad=fov_rad,
                x=x,
                y=y,
                th=th,
                max_sensor_dist=max_sensor_dist,
            )
            self.agent_obs[agent_id][3:] = sensor_dists
            ang_interval = fov_rad / n_sensors
            start_ang = -fov_rad/2. + th
            
            if self.show_distances:
                for i, dist in enumerate(sensor_dists):
                    sx = dist*np.cos(start_ang + i*ang_interval) + x
                    sy = dist*np.sin(start_ang + i*ang_interval) + y
                    lines.append(self.sensor_template.format(x, y, sx, sy))
                svg += ''.join(lines)

            svg += agent_svg
            svg += '</svg>'
            
            output_key = 'agent_observation:{0}'.format(agent_id)
            #self.r.set(output_key, self.agent_obs[agent_id])
            self.r.set(output_key, self.agent_obs[agent_id].ravel().tostring())

        self._nengo_html_ = svg

    def random_start_state(self):

        x = np.random.uniform(self.size)
        y = np.random.uniform(self.size)
        th = np.random.uniform(-np.pi, np.pi)

        return (x, y, th)

    def __call__(self, t, x):
        """
        Only input is the seed for random map generation
        """
        seed = int(x[0])

        if seed != self.current_seed:
            self.current_seed = seed
            self._generate_map()

        for agent_key in self.r.scan_iter('agent_act:*'):
            agent_id = int(agent_key.decode("utf-8").split(':')[1])
            if agent_id not in self.agent_params.keys():
                # Add the new agent to the system
                #agent_params = np.loads(self.r.get('agent_params:{0}'.format(agent_id)).decode("utf-8"))
                #agent_params = np.loads(self.r.get('agent_params:{0}'.format(agent_id)))
                agent_params = np.fromstring(self.r.get('agent_params:{0}'.format(agent_id)))
                # Format of params is [n_sensors, fov, max_sensor_dist]
                self.agent_params[agent_id] = agent_params
                self.agent_obs[agent_id] = np.zeros(3 + int(self.agent_params[agent_id][0]))
                self.agent_obs[agent_id][:3] = self.random_start_state()
                self.agent_ids.append(agent_id)
                self.n_agents += 1
            #output_key = 'agent_observation:{0}'.format(agent_id)
            #self.r.set(output_key, obs) #NOTE: moving this into _generate_svg
        
        # NOTE: this will also update the simulation and send data over redis
        self._generate_svg()


with model:

    map_selector = nengo.Node([0])

    env = nengo.Node(
        MultiAgentEnvironment(size=20, dt=0.01),
        size_in=1,
        size_out=0,
    )

    nengo.Connection(map_selector, env)
