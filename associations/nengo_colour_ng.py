import nengo
import numpy as np
import seaborn as sns
from PIL import Image
import base64
import cStringIO
from nengo_colour_som import ColourInput, SOM

class NeuralGas(SOM):

    def update_neurons(self, vec):

        dists =  np.sqrt(((vec - self.neurons)**2)).sum(axis=-1)
        k = np.argsort(np.argsort(dists.flatten())).reshape(dists.shape)
        h = np.exp(-k / self.sigma)

        self.lr = self.lr * .99
        self.sigma = self.sigma * .99

        self.lr = max(self.lr, self.lr_f)
        self.sigma = max(self.sigma, self.sigma_f)
        
        delta = vec - self.neurons
        for j in range(self.neurons.shape[-1]):
            self.neurons[..., j] += self.lr * h * delta[..., j]
        

palette = sns.color_palette()
#palette = sns.hls_palette(8, l=.3, s=.8)

model = nengo.Network()

with model:
    colour_input = nengo.Node(ColourInput(palette), size_in=0, size_out=3)

    #ng = NeuralGas(n=8, lr_i=.1)
    ng = NeuralGas(n=8, lr_f=0.5, sigma_f=2)
    ng_node = nengo.Node(ng, size_in=3, size_out=0)

    nengo.Connection(colour_input, ng_node)
