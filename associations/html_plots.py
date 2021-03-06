import sys
import numpy as np
from PIL import Image
import base64
import nengo

# TODO: add options to be compatible with Python3?
import cStringIO

class SpatialSpikePlot(object):
    """
    Plots spiking activity of a neuron in relation to the spatial location of an agent
    """
    def __init__(self, grid_size=512, xlim=(-1,1), ylim=(-1,1)):
        self.grid_size = grid_size
        self.xlim = xlim
        self.ylim = ylim
        self.space = np.zeros((grid_size, grid_size))
        self.x_len = xlim[1] - xlim[0]
        self.y_len = ylim[1] - ylim[0]

    def __call__(self, t, x):
        
        x_loc = x[0]
        y_loc = x[1]
        spike = x[2]
        
        # Get coordinates in the image
        x_im = int((x_loc + self.x_len/2.)*1.*self.grid_size/self.x_len)
        y_im = int((y_loc + self.y_len/2.)*1.*self.grid_size/self.y_len)
        # Place a spike or travelled path in the image
        if spike > 0:
            self.space[x_im,y_im] = 255
        else:
            if self.space[x_im,y_im] == 0:
                self.space[x_im,y_im] = 128
        
        values = self.space.astype('uint8')
        png = Image.fromarray(values)
        buffer = cStringIO.StringIO()
        png.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue())
        self._nengo_html_ = '''
            <svg width="100%%" height="100%%" viewbox="0 0 %s %s">''' % (self.grid_size, self.grid_size)
        self._nengo_html_ += '''
            <image width="100%%" height="100%%"
                   xlink:href="data:image/png;base64,%s"
                   style="image-rendering: pixelated;">
            </svg>''' % (''.join(img_str))

class CompleteSpatialSpikePlot(object):
    """
    Plots spiking activity of a neuron in relation to the spatial location of an agent
    Records activity of all neurons, and selects which one to show with an index input
    """
    def __init__(self, grid_size=512, n_neurons=500, xlim=(-1,1), ylim=(-1,1)):
        self.grid_size = grid_size
        self.xlim = xlim
        self.ylim = ylim
        self.space = np.zeros((n_neurons, grid_size, grid_size))
        self.x_len = xlim[1] - xlim[0]
        self.y_len = ylim[1] - ylim[0]
        self.n_neurons = n_neurons

    def __call__(self, t, x):
        
        x_loc = x[0]
        y_loc = x[1]
        index = int(x[2])
        spikes = x[3:]
 
        index = max(0, index)
        index = min(self.n_neurons-1, index)

        # Get coordinates in the image
        #x_im = int((x_loc + self.x_len/2.)*1.*self.grid_size/self.x_len)
        #y_im = int((y_loc + self.y_len/2.)*1.*self.grid_size/self.y_len)
        x_im = int((x_loc + self.x_len - self.xlim[1])*1.*self.grid_size/self.x_len)
        y_im = int((y_loc + self.y_len - self.ylim[1])*1.*self.grid_size/self.y_len)
        
        if x_im >= 0 and x_im < self.grid_size and y_im >= 0 and y_im < self.grid_size:

            # Place a spike or travelled path in the image
            for i, spike in enumerate(spikes):
                if spike > 0:
                    self.space[i, x_im,y_im] = 255
                else:
                    if self.space[i, x_im,y_im] == 0:
                        self.space[i, x_im,y_im] = 128
        
        values = self.space[index,:,:].astype('uint8')
        png = Image.fromarray(values)
        buffer = cStringIO.StringIO()
        png.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue())
        self._nengo_html_ = '''
            <svg width="100%%" height="100%%" viewbox="0 0 %s %s">''' % (self.grid_size, self.grid_size)
        self._nengo_html_ += '''
            <image width="100%%" height="100%%"
                   xlink:href="data:image/png;base64,%s"
                   style="image-rendering: pixelated;">
            </svg>''' % (''.join(img_str))

class TuningHeatMap(object):
    """
    Plots spiking activity of a neuron in relation to some other variable
    Records activity of all neurons, and selects which one to show with an index input
    Selecting an index of -1 will show all neurons at once
    """
    def __init__(self, data_size=512, n_neurons=500, data_range=(-1,1)):
        self.data_size = data_size # number of bins
        self.data_range = data_range # range of possible values
        self.data_len = data_range[1] - data_range[0]
        self.data = np.zeros((n_neurons, data_size))
        #self.bins = np.linspace(data_range[0], data_range[1], data_size)
        self.n_neurons = n_neurons

    def __call__(self, t, x):
        
        value = x[0]
        index = int(x[1])
        spikes = x[2:]
 
        index = max(-1, index)
        index = min(self.n_neurons-1, index)

        # Find the corresponding 'bin' number for the data point
        bin_index = int((value - self.data_range[0]) * 1.*self.data_size/self.data_len)

        bin_index = max(0, bin_index)
        bin_index = min(self.data_size-1, bin_index)

        # Place a spike or travelled path in the image
        for i, spike in enumerate(spikes):
            if spike > 0:
                self.data[i, bin_index] += 1

                # Cap the value based on what the image will show
                # TODO: do something more flexible in the future
                if self.data[i, bin_index] > 255:
                    self.data[i, bin_index] = 255
        
        if index == -1:
            values = self.data.astype('uint8')
        else:
            # Stack many copies of itself in the image to make it visible
            values = np.zeros((self.n_neurons, self.data_size), dtype=np.uint8)
            values[:,:] = self.data[index,:]

        png = Image.fromarray(values)
        buffer = cStringIO.StringIO()
        png.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue())
        self._nengo_html_ = '''
            <svg width="100%%" height="100%%" viewbox="0 0 %s %s">''' % (self.n_neurons, self.data_size)
        self._nengo_html_ += '''
            <image width="100%%" height="100%%"
                   xlink:href="data:image/png;base64,%s"
                   style="image-rendering: pixelated;">
            </svg>''' % (''.join(img_str))

class EncoderPlot(nengo.Node):

    def __init__(self, connection, scaling='max'):

        self.connection = connection

        self.ensemble = connection.post_obj

        self.encoder_probe = nengo.Probe(connection.learning_rule, 'scaled_encoders')

        self.encoders = None

        self.scaling = scaling

        def plot(t):
            if self.encoders is None:
                return
            plot._nengo_html_ = '<svg width="100%" height="100%" viewbox="0 0 100 100">'
            #mx = (np.max(self.encoders) + np.mean(self.encoders))/2.
            if self.scaling == 'max':
                mx = np.max(self.encoders)
                if mx > 0:
                    self.encoders = self.encoders * (50. /mx)
            for e in self.encoders:
                if self.scaling == 'normalize':
                    e /= np.linalg.norm(e)
                    e *= 50
                plot._nengo_html_ += '<circle cx="{0}" cy="{1}" r="{2}"  stroke-width="1.0" stroke="blue" fill="blue" />'.format(e[0]+50, e[1]+50, 1)
            plot._nengo_html_ += '</svg>'

        super(EncoderPlot, self).__init__(plot, size_in=0, size_out=0)
        self.output._nengo_html_ = '<svg width="100%" height="100%" viewbox="0 0 100 100"></svg>'

    def update(self, sim):
        if sim is None:
            return

        self.encoders = sim._probe_outputs[self.encoder_probe][-1]
        del sim._probe_outputs[self.encoder_probe][:]

class WeightPlot(nengo.Node):
    def __init__(self, connection, scaling='max'):

        self.connection = connection

        self.ensemble = connection.post_obj

        self.weight_probe = nengo.Probe(connection, 'weights', sample_every=0.01)

        self.weights = None

        self.scaling = scaling

        def plot(t):
            if self.weights is None:
                return
            mn = np.min(self.weights)
            mx = np.max(self.weights)
            rn = mx-mn
            self.weights = ((self.weights + mn)/rn)*255
            values = self.weights.astype('uint8')
            png = Image.fromarray(values)
            buffer = cStringIO.StringIO()
            png.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue())
            plot._nengo_html_ = '''
                <svg width="100%%" height="100%%" viewbox="0 0 %s %s">''' % (self.weights.shape[1], self.weights.shape[0])
            plot._nengo_html_ += '''
                <image width="100%%" height="100%%"
                       xlink:href="data:image/png;base64,%s"
                       style="image-rendering: pixelated;">
                </svg>''' % (''.join(img_str))

        super(WeightPlot, self).__init__(plot, size_in=0, size_out=0)
        self.output._nengo_html_ = '<svg width="100%" height="100%" viewbox="0 0 100 100"></svg>'

    def update(self, sim):
        if sim is None:
            return
        try:
            self.weights = sim._probe_outputs[self.weight_probe][-1]
            del sim._probe_outputs[self.weight_probe][:]
        except:
            self.weights = None

