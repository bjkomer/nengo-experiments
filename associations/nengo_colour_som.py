import nengo
import numpy as np
import seaborn as sns

class SOM(object):
    def __init__(self):
        pass

    def __call__(self, t, x):
        pass

palette = sns.color_palette()
palette = sns.hls_palette(8, l=.3, s=.8)
class ColourInput(object):

    def __init__(self, palette):

        self.palette = palette
        self.num_colours = len(palette)
        self.base_html = '''<svg width="100%" height="100%" viewbox="0 0 100 100">'''
        self.base_html += '<rect width="100" height="100" stroke-width="2.0" stroke="black" fill="rgb({0}, {1}, {2})" />'
        self.base_html += '</svg>'
        self.random_colour()

    def random_colour(self):

        self.colour = self.palette[np.random.randint(0,self.num_colours)]
        self._nengo_html_ = self.base_html.format(int(self.colour[0]*256), int(self.colour[1]*256), int(self.colour[2]*256))

    def __call__(self, t):

        if t % 1 == 0:
            self.random_colour()

        return self.colour


model = nengo.Network()

with model:
    colour_input = nengo.Node(ColourInput(palette), size_in=0, size_out=3)
