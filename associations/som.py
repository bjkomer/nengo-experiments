import numpy as np
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt

#TODO: try making a circular SOM (connections wrap around)

#map_shape = (10,10)

np.random.seed(13)

# this example does 3 channel colour, so it has 3 weights
n = 8
som = np.random.rand(n,n,3)
plt.imshow(som)

#num_samples = 6#30

# TODO: plot the colours by themselves to see what they look like (seaborn?)
#data = np.zeros((num_samples,3))
#data = np.random.rand(num_samples,3)
#data = np.array(sns.color_palette())
palette = sns.color_palette()
#palette = sns.hls_palette(16, l=.3, s=.8)
palette = sns.hls_palette(8, l=.3, s=.8)
num_samples = len(palette)
data = np.array(palette)

#sns.palplot(data)

print(data.shape)

num_iterations = 20000 #200000

# learning rate
const_lr = .1#.001
lr = .001
map_radius = 5

lr_i = .5 # initial learning rate
lr_f = .005 # final learning rate

sigma_i = 10#10 # initial neighborhood parameter
sigma_f = .01 # final neighborhood parameter

def dist_func(a,b):
    #return np.linalg.norm(a-b)
    diff = a-b
    #d = np.sqrt(diff[0]**2+diff[1]**2+diff[2]**2)
    d = diff[0]**2+diff[1]**2+diff[2]**2
    return d

def get_best_match(vec, som):
    d = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            d[i,j] = dist_func(vec, som[i,j])
    # this is the flattened index
    index = np.argmax(d)
    return (index/n, index%n)

"""
best_index = get_best_match(np.array([.5,.5,.5]),som)

print(som)
print(best_index)
print(som[best_index])
"""

#ssample = data
#print(ssample)
#I = np.random.randint(0, num_samples, num_iterations)
for i in range(num_iterations):
    # Update learning rate and neighborhood radius
    #time_constant = num_iterations/np.log(map_radius)
    #neighborhood_radius = map_radius * np.exp(-i/time_constant)

    #lr = const_lr * np.exp(-i/float(num_iterations))

    lr = lr_i*(lr_f/lr_i)**(i/float(num_iterations))
    sigma = sigma_i*(sigma_f/sigma_i)**(i/float(num_iterations))
    
    # Pick a sample at random
    s = np.random.randint(0, num_samples)
    #s = ssample[I[i]]

    best_index = get_best_match(data[s,:], som)
    #best_index = get_best_match(s, som)

    for x in range(n):
        for y in range(n):
            dist = np.sqrt((best_index[0]-x)**2+(best_index[1]-y)**2)
            
            influence = np.exp(-dist**2/(sigma**2))
            
            #som[x,y,:] += lr * influence * (data[s,:] - som[x,y,:])
            som[y,x,:] += lr * influence * (data[s,:] - som[y,x,:])
            #som[y,x,:] += lr * influence * (s - som[y,x,:])
            """
            if dist < neighborhood_radius:
                #TODO: make this more efficient by removing redundant squares
                #influence = np.exp(-dist**2/(2*neighborhood_radius**2))
                influence = np.exp(-dist**2/(sigma**2))

                som[x,y,:] += lr * influence * (data[s,:] - som[x,y,:])
                #som[x,y,:] += lr * influence * (som[x,y,:] - data[s,:])
            """
    if i % 1000 == 0:
        print("Iteration: {0}".format(i))
        #print(influence)

print(som)
plt.figure()
plt.imshow(som)
sns.palplot(palette)

plt.show()
