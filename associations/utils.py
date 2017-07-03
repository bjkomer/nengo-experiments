import numpy as np

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
