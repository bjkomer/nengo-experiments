import numpy as np

class FlavourLand(object):

    def __init__(self, shape, flavours, flavour_rad = .5,
                 start_x=1, start_y=1, start_th=0, dt=0.001
                ):

        self.shape = shape
        self.flavours = flavours # dictionary of name:location
        self.num_flavours = len(flavours)
        self.x = start_x
        self.y = start_y
        self.th = start_th
        self.flavour_rad = flavour_rad # radius of flavour effect
        self.dt = dt

        # Scaling for positions on the HTML plot
        self.scale_x = 100. / self.shape[0]
        self.scale_y = 100. / self.shape[1]

        # Colours to display each flavour as
        self.colour_list = ['blue', 'green', 'red', 'magenta', 'cyan', 'yellow', 'purple', 'fuschia', 'grey', 'lime']
        self.num_colours = len(self.colour_list)

        self.build_html_string()

        self._nengo_html_ = self.base_html.format(self.x, self.y, self.th)

    def build_html_string(self):

        # Used to display HTML plot
        self.base_html = '''<svg width="100%" height="100%" viewbox="0 0 100 100">'''
        
        # Draw the outer rectangle
        self.base_html += '<rect width="100" height="100" stroke-width="2.0" stroke="black" fill="white" />'

        # Draw circles for each flavour
        for i, loc in enumerate(self.flavours.itervalues()):
            self.base_html += '<circle cx="{0}" cy="{1}" r="{2}" stroke-width="1.0" stroke="{3}" fill="{3}" />'.format(loc[0]*self.scale_x, 100-loc[1]*self.scale_y, self.flavour_rad*self.scale_x, self.colour_list[i%self.num_colours])

        # Set up the agent to be filled in later with 'format()'
        self.base_html += '<polygon points="{0}" stroke="black" fill="black" />'

        # Close the svg
        self.base_html += '</svg>'

    def move(self, vel):

        self.th += vel[1]*self.dt
        if self.th > np.pi:
            self.th -= 2*np.pi
        if self.th < -np.pi:
            self.th += 2*np.pi

        self.x += np.cos(self.th)*vel[0]*self.dt
        self.y += np.sin(self.th)*vel[0]*self.dt

        if self.x > self.shape[0]:
            self.x = self.shape[0]
        if self.x < 0:
            self.x = 0
        if self.y > self.shape[1]:
            self.y = self.shape[1]
        if self.y < 0:
            self.y = 0

        self.update_html()
    
    def move_random(self, vel=5):

        # randomly vary direction angle between -90 and +90 degrees
        #ang_diff = np.random.random() * np.pi - np.pi/2
        
        # randomly vary direction angle between -5 and +5 degrees
        ang_diff = (np.random.random() * np.pi - np.pi/2)/18

        self.th += ang_diff

        dx = vel * np.cos(self.th)
        dy = vel * np.sin(self.th)

        self.x += dx * self.dt
        self.y += dy * self.dt

        if self.x > self.shape[0]:
            self.x = self.shape[0]
        if self.x < 0:
            self.x = 0
        if self.y > self.shape[1]:
            self.y = self.shape[1]
        if self.y < 0:
            self.y = 0

        self.update_html()
        
    def update_html(self):
        # Define points of the triangular agent based on x, y, and th
        x1 = (self.x + 0.5*np.cos(self.th - 2*np.pi/3))*self.scale_x
        y1 = 100-(self.y + 0.5*np.sin(self.th - 2*np.pi/3))*self.scale_y
        
        x2 = (self.x + np.cos(self.th))*self.scale_x
        y2 = 100-(self.y + np.sin(self.th))*self.scale_y
        
        x3 = (self.x + 0.5*np.cos(self.th + 2*np.pi/3))*self.scale_x
        y3 = 100-(self.y + 0.5*np.sin(self.th + 2*np.pi/3))*self.scale_y

        points = "{0},{1} {2},{3} {4},{5}".format(x1,y1,x2,y2,x3,y3)

        # Update the html plot
        self._nengo_html_ = self.base_html.format(points)

    def get_flavours(self):

        f = [0]*self.num_flavours
        for i, loc in enumerate(self.flavours.itervalues()):
            f[i] = self.detect(loc)
        return f

    def detect(self, loc):

        dist = np.sqrt((self.x - loc[0])**2 + (self.y - loc[1])**2)
        if dist < self.flavour_rad:
            return 1
        else:
            return 0

    def generate_data(self, steps=1000, vel=5):
        """
        Generate artificial data corresponding to randomly moving through the environment
        """

        data = np.zeros((steps, 3 + self.num_flavours))

        for i in range(steps):

            self.move_random(vel)

            f = self.get_flavours()
            data[i,:] = [self.x, self.y, self.th] + f

        return data

    def __call__(self, t, x):

        #self.move(x)
        self.move_random()

        f = self.get_flavours() #TODO: figure out the format of the flavours

        return [self.x, self.y, self.th] + f
