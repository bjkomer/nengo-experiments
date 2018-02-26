import numpy as np
import nengo

# Defining axes of 2D representation with respect to the 3D hexagonal one
x_axis = np.array([1,-1,0])
y_axis = np.array([-1,-1,2])
x_axis = x_axis / np.linalg.norm(x_axis)
y_axis = y_axis / np.linalg.norm(y_axis)

# Converts a 3D hexagonal point to the 2D equivalent
# Projecting to the plane defined by the (1,1,1) normal
def to_xy(coord):

    x = np.dot(x_axis, coord)
    y = np.dot(y_axis, coord)

    return np.array([x,y])

# Converts a 2D coordinate into the corresponding
# 3D coordinate in the hexagonal representation
def to_xyz(coord):
    return x_axis*coord[1]+y_axis*coord[0]

# Convert 6D cyclic hex representation to 2D coordinates
def cyclic_to_xy(v):
    xh = np.arctan2(v[0],v[1]) / np.pi
    yh = np.arctan2(v[2],v[3]) / np.pi
    zh = np.arctan2(v[4],v[5]) / np.pi

    return to_xy(np.array([xh,yh,zh]))

# Converts input angles to cyclic coordinates
def ang_to_cyclic(v):
    return np.array([np.cos(v[0]),
                     np.sin(v[0]),
                     np.cos(v[1]),
                     np.sin(v[1]),
                     np.cos(v[2]),
                     np.sin(v[2]),
                    ])

def xy_to_cyclic_6d(x):
    return ang_to_cyclic(to_xyz(x))

def ang_to_xy(v):
    return cyclic_to_xy(ang_to_cyclic(v))

tau = .1
def integrate_velocity(x):

    vel_2d = x[6:]
    vel_3d = to_xyz(vel_2d)

    ret = np.zeros((3,2))
    for i in range(3):
        ret[i,0] = (x[i*2] + x[i*2+1]*vel_3d[i]*tau)*1.1
        ret[i,1] = (x[i*2+1] - x[i*2]*vel_3d[i]*tau)*1.1

    return ret.flatten(order='C')

def integrate_velocity_3d(x):

    vel_3d = x[6:]

    ret = np.zeros((3,2))
    for i in range(3):
        ret[i,0] = (x[i*2] + x[i*2+1]*vel_3d[i]*tau)*1.1
        ret[i,1] = (x[i*2+1] - x[i*2]*vel_3d[i]*tau)*1.1

    return ret.flatten(order='C')

def integrate_velocity_1d(x):

    return (x[0] + x[1]*x[2]*tau)*1.1, (x[1] - x[0]*x[2]*tau)*1.1

# Integrate a 2D velocity signal to update 2D position
def integrate_vel_2d(x):
    xp = x[0] + x[2]*tau
    yp = x[1] + x[3]*tau
    
    return (xp, yp)

def cyclic_to_ang_1d(v):
    return np.arctan2(v[0],v[1]) / np.pi

def twist_to_vel_3d(x):
    lin_vel = x[0]
    ang_vel = x[1] #NOTE: currently not using angular velocity
    ang = x[2]

    vel_x = np.cos(ang)*lin_vel
    vel_y = np.sin(ang)*lin_vel

    return to_xyz((vel_x, vel_y))

def wrap(x):
    xp = x[0]
    yp = x[1]

    # Do the wrapping
    while xp > 1:
        xp -= 2
    while xp < -1:
        xp += 2
    while yp > 1:
        yp -=2
    while yp < -1:
        yp += 2

    return (xp, yp)

class RandomWalk:
    def __init__(self, dt=0.001, lin_vel_max=5, ang_vel_max=5, shape=(10,10),
                 fixed_environment=True, init_x=0, init_y=0, init_ang=0,
                ):
        self.x = init_x # current x position
        self.y = init_y # current y position
        self.lin_vel_max = lin_vel_max # maximum linear velocity
        self.ang_vel_max = ang_vel_max # maximum angular velocity
        self.ang = init_ang # current facing angle
        self.ang_vel = 0 # initial angular velocity
        self.dt = dt

        self.shape = shape
        self.fixed_environment = fixed_environment
    
    def bounded(self, new_x, new_y):
        # Returns true if the agent will remain in the environment with its current trajectory
        if new_x > self.xlim[1]:
            return False
        if new_x < self.xlim[0]:
            return False
        if new_y > self.ylim[1]:
            return False
        if new_y < self.ylim[0]:
            return False
        return True
    
    def __call__(self, t):

        # randomly vary direction angle between -135 and +135 degrees
        #ang_diff = (np.random.random() * np.pi - np.pi/2) * 1.5
        
        lin_vel = np.random.random() * self.lin_vel_max
        ang_vel = (np.random.random()-.5) * self.ang_vel_max
        self.ang_vel += ang_vel*self.dt*10

        #TODO: have some way of putting realistic limits on this

        """
        new_ang = self.ang + ang_diff
        new_x = self.x + (vel*np.cos(new_ang))*self.dt
        new_y = self.y + (vel*np.sin(new_ang))*self.dt

        if self.fixed_environment:
            #TODO: make sure this can't get infinitely stuck
            while not self.bounded(new_x, new_y):
                ang_diff = (np.random.random() * np.pi - np.pi/2) * 1.5
                vel = np.random.random() * self.vel_max
                new_ang = self.ang + ang_diff
                new_x = self.x + (vel*np.cos(new_ang))*self.dt
                new_y = self.y + (vel*np.sin(new_ang))*self.dt
            self.ang = new_ang
            self.x = new_x
            self.y = new_y
        else:
            self.ang = new_ang
            self.x = new_x
            self.y = new_y

            # Loop around if the environment is not fixed
            if self.x > 1:
                self.x -= 2
            if self.x < -1:
                self.x += 2
            if self.y > 1:
                self.y -= 2
            if self.y < -1:
                self.y += 2
        
        return (self.x, self.y, vel*np.cos(self.ang), vel*np.sin(self.ang))
        """
        return (lin_vel, self.ang_vel)

# Like a random walk, but outputs velocity
class RandomRun:
    def __init__(self, dt=0.001, lin_vel_max=5, ang_vel_max=5, shape=(10,10),
                 fixed_environment=True, init_x=0, init_y=0, init_ang=0,
                ):
        self.x = init_x # current x position
        self.y = init_y # current y position
        self.lin_vel_max = lin_vel_max # maximum linear velocity
        self.ang_vel_max = ang_vel_max # maximum angular velocity
        self.ang = init_ang # current facing angle
        self.ang_vel = 0 # initial angular velocity
        self.dt = dt

        self.shape = shape
        self.fixed_environment = fixed_environment
    
    def bounded(self, new_x, new_y):
        # Returns true if the agent will remain in the environment with its current trajectory
        if new_x > self.xlim[1]:
            return False
        if new_x < self.xlim[0]:
            return False
        if new_y > self.ylim[1]:
            return False
        if new_y < self.ylim[0]:
            return False
        return True
    
    def __call__(self, t):

        # randomly vary direction angle between -135 and +135 degrees
        #ang_diff = (np.random.random() * np.pi - np.pi/2) * 1.5
        
        lin_vel = np.random.random() * self.lin_vel_max
        ang_vel = (np.random.random()-.5) * self.ang_vel_max
        self.ang_vel += ang_vel*self.dt*10

        #TODO: have some way of putting realistic limits on this

        """
        new_ang = self.ang + ang_diff
        new_x = self.x + (vel*np.cos(new_ang))*self.dt
        new_y = self.y + (vel*np.sin(new_ang))*self.dt

        if self.fixed_environment:
            #TODO: make sure this can't get infinitely stuck
            while not self.bounded(new_x, new_y):
                ang_diff = (np.random.random() * np.pi - np.pi/2) * 1.5
                vel = np.random.random() * self.vel_max
                new_ang = self.ang + ang_diff
                new_x = self.x + (vel*np.cos(new_ang))*self.dt
                new_y = self.y + (vel*np.sin(new_ang))*self.dt
            self.ang = new_ang
            self.x = new_x
            self.y = new_y
        else:
            self.ang = new_ang
            self.x = new_x
            self.y = new_y

            # Loop around if the environment is not fixed
            if self.x > 1:
                self.x -= 2
            if self.x < -1:
                self.x += 2
            if self.y > 1:
                self.y -= 2
            if self.y < -1:
                self.y += 2
        
        return (self.x, self.y, vel*np.cos(self.ang), vel*np.sin(self.ang))
        """
        return (lin_vel, self.ang_vel)

def coord_to_pos_vec(x, 
                     num_offsets=4, 
                     multiple_frequencies=True,
                     freq_scale=1,
                     D=16,
                     input_scale=1
                    ):

    ret = []

    if num_offsets == 1:
        offsets = [0]
    elif num_offsets == 2:
        offsets = [0, 1]
    elif num_offsets == 3:
        offsets = [0, .66, 1.33]
    elif num_offsets == 4:
        offsets = [0, 0.5, 1, 1.5]

    denom=np.sqrt(D/8)

    p = 0

    for i in range(int(np.ceil(D/(4.*num_offsets)))):
        for m in offsets:
            if multiple_frequencies:
                ret += [np.cos( freq_scale*( m*np.pi + x[0]*input_scale ) / ( i + 1.)) / denom,
                        np.sin( freq_scale*( m*np.pi + x[0]*input_scale ) / ( i + 1.)) / denom,
                        np.cos( freq_scale*( m*np.pi + x[1]*input_scale ) / ( i + 1.)) / denom,
                        np.sin( freq_scale*( m*np.pi + x[1]*input_scale ) / ( i + 1.)) / denom]
            else:
                ret += [np.cos( freq_scale*( m*np.pi + x[0]*input_scale )) / denom,
                        np.sin( freq_scale*( m*np.pi + x[0]*input_scale )) / denom,
                        np.cos( freq_scale*( m*np.pi + x[1]*input_scale )) / denom,
                        np.sin( freq_scale*( m*np.pi + x[1]*input_scale )) / denom]
            p+=4
            # This is needed for when num_offsets doesn't divide D evenly
            if p >= D:
                break

    return ret

def pos_vec_to_coord(x,
                     num_offsets=4,
                     multiple_frequencies=True,
                     freq_scale=1,
                     D=16,
                     input_scale=1,
                    ):
    
    if num_offsets == 1:
        offsets = [0]
    elif num_offsets == 2:
        offsets = [0, 1]
    elif num_offsets == 3:
        offsets = [0, .66, 1.33]
    elif num_offsets == 4:
        offsets = [0, 0.5, 1, 1.5]

    denom = np.sqrt(D/8)
    
    # estimates of the coordinates
    # FIXME: this might break around the -pi/pi transition point
    x_total = 0
    y_total = 0

    p = 0 # current position in the vector
    for i in range(int(np.ceil(D/(4*num_offsets)))):
        for m in offsets:
            if multiple_frequencies:
                x_est = (np.arctan2(x[p+1], x[p]) / (freq_scale * (i+1.)) - m*np.pi)/input_scale
                y_est = (np.arctan2(x[p+3], x[p+2]) / (freq_scale * (i+1.)) - m*np.pi)/input_scale
            else:
                x_est = (np.arctan2(x[p+1], x[p]) / freq_scale - m*np.pi)/input_scale
                y_est = (np.arctan2(x[p+3], x[p+2]) / freq_scale - m*np.pi)/input_scale
            
            while x_est < -np.pi:
                x_est += 2*np.pi
            while x_est > np.pi:
                x_est -= 2*np.pi
            while y_est < -np.pi:
                y_est += 2*np.pi
            while y_est > np.pi:
                y_est -= 2*np.pi

            x_total += x_est
            y_total += y_est

            p+=4
            # This is needed for when num_offsets doesn't divide D evenly
            if p >= D:
                break

    # Make the estimates be an average
    x_total /= p/4
    y_total /= p/4
    
    return x_total, y_total


# functions for the voja association code

# compute a control signal to get to the location
def compute_velocity(x):
    # which way the agent should face to go directly to the target
    desired_ang = np.arctan2(-x[1], -x[0])
    
    ang_diff = -1*(x[2] - desired_ang)
    
    if ang_diff > np.pi:
        ang_diff -= 2*np.pi
    elif ang_diff < -np.pi:
        ang_diff += 2*np.pi

    ang_vel = ang_diff*.9
    if np.sqrt(x[0]**2+x[1]**2) < .001:
        lin_vel = 0
        ang_vel = 0
    elif abs(ang_diff) < np.pi/4.:
        lin_vel = .6*np.sqrt(x[0]**2+x[1]**2)
    elif abs(ang_diff) < np.pi/2.:
        lin_vel = .4*np.sqrt(x[0]**2+x[1]**2)
    else:
        lin_vel = 0
    
    return lin_vel, ang_vel

# Scales the location output to be between -1 and 1
def scale_location(x, shape=(10,10)):
    x_out = x[0]/(shape[0]/2.) - 1
    y_out = x[1]/(shape[1]/2.) - 1
    th_out = x[2]/np.pi #TODO: should this get scaled at all?

    return [x_out, y_out, th_out]

def scale_xy_node(t, x, shape=(10,10)):
    x_out = x[0]/(shape[0]/2.) - 1
    y_out = x[1]/(shape[1]/2.) - 1

    return x_out, y_out

def env_scale_node(t, x, shape=(10,10)):
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

def loc_scale_to_surface(x, shape=(10,10)):
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

def room_vec_to_scalar(t, x):
    for i, val in enumerate(x):
        if val > .5:
            return i+.1
    return 0

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

class SpecifyDecoders(nengo.solvers.Solver):
  def __init__(self, decoders, weights=False):
    
    super(SpecifyDecoders, self).__init__()
    
    #self.weights = weights
    self.decoders = decoders

  def __call__(self, A, Y, E=None, rng=None):
    return self.decoders, []
