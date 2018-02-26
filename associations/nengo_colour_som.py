import nengo
import numpy as np
import seaborn as sns
from PIL import Image
import base64
import cStringIO

def dist_func(a,b):
    #return np.linalg.norm(a-b)
    diff = a-b
    #d = np.sqrt(diff[0]**2+diff[1]**2+diff[2]**2)
    d = diff[0]**2+diff[1]**2+diff[2]**2
    return d

class SOM(object):
    def __init__(self, n, weight_dim=3, lr_i=0.5, lr_f=0.005, sigma_i=10, sigma_f=0.01):
        self.n = n
        self.weight_dim = weight_dim
        self.neurons = np.random.rand(n,n,weight_dim)

        self.lr_i = lr_i
        self.lr_f = lr_f
        self.sigma_i = sigma_i
        self.sigma_f = sigma_f

        self.lr = lr_i
        self.sigma = sigma_i

        self.update_html()

    def get_best_match(self, vec):
        d = np.zeros((self.n,self.n))
        #TODO: do this without a loop
        for i in range(self.n):
            for j in range(self.n):
                d[i,j] = dist_func(vec, self.neurons[i,j])
        # this is the flattened index
        index = np.argmax(d)
        return (index/self.n, index%self.n)

    def calc_dist(self, best_index, x, y):
        return np.sqrt((best_index[0]-x)**2+(best_index[1]-y)**2)

    def update_neurons(self, vec):
        best_index = self.get_best_match(vec)
        """
        lr = self.lr_i*(self.lr_f/self.lr_i)#**(i/float(num_iterations))
        sigma = self.sigma_i*(self.sigma_f/self.sigma_i)#**(i/float(num_iterations))

        lr = self.lr_i*.1
        sigma = self.sigma_i*.1
        """

        self.lr = self.lr * .99
        self.sigma = self.sigma * .99

        self.lr = max(self.lr, self.lr_f)
        self.sigma = max(self.sigma, self.sigma_f)

        for x in range(self.n):
            for y in range(self.n):
                dist = self.calc_dist(best_index, x, y)
                #np.sqrt((best_index[0]-x)**2+(best_index[1]-y)**2)
                
                influence = np.exp(-dist**2/(self.sigma**2))
                
                self.neurons[y,x,:] += self.lr * influence * (vec - self.neurons[y,x,:])

    def update_html(self):
        scaled_neurons = self.neurons*256
        #values = self.neurons.astype('uint8')
        values = scaled_neurons.astype('uint8')
        png = Image.fromarray(values)
        buffer = cStringIO.StringIO()
        png.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue())
        self._nengo_html_ = '''
            <svg width="100%%" height="100%%" viewbox="0 0 %s %s">''' % (self.n, self.n)
        self._nengo_html_ += '''
            <image width="100%%" height="100%%"
                   xlink:href="data:image/png;base64,%s"
                   style="image-rendering: pixelated;">
            </svg>''' % (''.join(img_str))

    def __call__(self, t, x):
        self.update_neurons(x)
        self.update_html()

class CyclicSOM(SOM):

    def calc_dist(self, best_index, x, y):
        return min(np.sqrt((best_index[0]-x)**2+(best_index[1]-y)**2),
                   np.sqrt((best_index[0]-x-self.n)**2+(best_index[1]-y)**2),
                   np.sqrt((best_index[0]-x+self.n)**2+(best_index[1]-y)**2),
                   np.sqrt((best_index[0]-x)**2+(best_index[1]-y-self.n)**2),
                   np.sqrt((best_index[0]-x)**2+(best_index[1]-y+self.n)**2),
                   np.sqrt((best_index[0]-x+self.n)**2+(best_index[1]-y+self.n)**2),
                   np.sqrt((best_index[0]-x-self.n)**2+(best_index[1]-y+self.n)**2),
                   np.sqrt((best_index[0]-x+self.n)**2+(best_index[1]-y-self.n)**2),
                   np.sqrt((best_index[0]-x-self.n)**2+(best_index[1]-y-self.n)**2),
                  )

class DSOM(SOM):

    def __init__(self, elasticity=1.75, *args, **kwargs):

        super(DSOM, self).__init__(*args, **kwargs)

        self.elasticity = elasticity
        self.max = 0
    
    def update_neurons(self, vec):
        
        D = ((self.neurons - vec)**2).sum(axis=-1)
        
        best_index = self.get_best_match(vec)

        self.max = max(D.max(), self.max)
        d = np.sqrt(D/self.max)
        self.sigma = self.elasticity*d[best_index]
        
        for x in range(self.n):
            for y in range(self.n):
                dist = self.calc_dist(best_index, x, y)
                
                influence = np.exp(-dist**2/(self.sigma**2))
                
                self.neurons[y,x,:] += self.lr * d[y,x] * influence * (vec - self.neurons[y,x,:])

class CyclicDSOM(DSOM):
    
    def calc_dist(self, best_index, x, y):
        return min(np.sqrt((best_index[0]-x)**2+(best_index[1]-y)**2),
                   np.sqrt((best_index[0]-x-self.n)**2+(best_index[1]-y)**2),
                   np.sqrt((best_index[0]-x+self.n)**2+(best_index[1]-y)**2),
                   np.sqrt((best_index[0]-x)**2+(best_index[1]-y-self.n)**2),
                   np.sqrt((best_index[0]-x)**2+(best_index[1]-y+self.n)**2),
                   np.sqrt((best_index[0]-x+self.n)**2+(best_index[1]-y+self.n)**2),
                   np.sqrt((best_index[0]-x-self.n)**2+(best_index[1]-y+self.n)**2),
                   np.sqrt((best_index[0]-x+self.n)**2+(best_index[1]-y-self.n)**2),
                   np.sqrt((best_index[0]-x-self.n)**2+(best_index[1]-y-self.n)**2),
                  )
    

class ColourInput(object):

    def __init__(self, palette):

        self.palette = palette
        self.num_colours = len(palette)
        self.base_html = '''<svg width="100%" height="100%" viewbox="0 0 100 100">'''
        self.base_html += '<rect width="100" height="100" stroke-width="2.0" stroke="black" fill="rgb({0}, {1}, {2})" />'
        self.base_html += '</svg>'
        self.random_colour()
        self.count = 0

    def random_colour(self):

        self.colour = self.palette[np.random.randint(0,self.num_colours)]
        self._nengo_html_ = self.base_html.format(int(self.colour[0]*256), int(self.colour[1]*256), int(self.colour[2]*256))

    def __call__(self, t):

        #if t % 1 == 0:
        #if t % .01 == 0:
        if self.count >= 50:
            self.random_colour()
            self.count = 0
        self.count += 1

        return self.colour

palette = sns.color_palette()
#palette = sns.hls_palette(8, l=.3, s=.8)

model = nengo.Network()

with model:
    colour_input = nengo.Node(ColourInput(palette), size_in=0, size_out=3)

    #s = SOM(n=8)
    #s = SOM(n=8,sigma_f=2) # this one works nice
    #s = SOM(n=16,sigma_f=4, lr_f=0.01)
    #s = CyclicSOM(n=16,sigma_f=4, lr_f=0.01)
    #s = DSOM(n=16, lr_i=.1)
    s = CyclicDSOM(n=8, lr_i=.1)
    som = nengo.Node(s, size_in=3, size_out=0)

    nengo.Connection(colour_input, som)
