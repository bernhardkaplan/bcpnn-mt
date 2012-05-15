import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import PatchCollection
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

"""
Works only for square grids, i.e. n_x = n_y
"""
import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

N_RF_X = np.int(np.sqrt(params['N_RF']*np.sqrt(3)))
N_RF_Y = np.int(np.sqrt(params['N_RF']/np.sqrt(3)))
#N_RF_X = 8
#N_RF_Y = N_RF_X
n_cells = N_RF_X * N_RF_Y

RF = np.zeros((2, N_RF_X*N_RF_Y))
X, Y = np.mgrid[0:1:1j*(N_RF_X + 1), 0:1:1j*(N_RF_Y + 1)]
Y[::2, :] += (Y[0, 0] - Y[0, 1])/2
#X += (Y[0, 1] - Y[0, 0])/2

# It's a torus, so we remove the first row and column to avoid redundancy (would in principle not harm)
X, Y = X[1:, 1:], Y[1:, 1:]

RF[0, :] = X.ravel()
RF[1, :] = Y.ravel()
RF[0, :] *= np.sqrt(3.) # don't know why but it fits
scale =  (RF[1, 1] - RF[1, 0]) * 2./3#/ (np.sqrt(3))
#r_i = scale
r_i = (np.sqrt(3) / 2) * scale







angle_in_radian = 30 * np.pi / 180
patches = []
for i in xrange(N_RF_X * N_RF_Y):
    pos = (RF[0, i], RF[1, i])
#    print "i, pos", i, pos
    polygon = mpatches.RegularPolygon(pos, 6, r_i, orientation=angle_in_radian, lw=0.001)#, edgecolor='none')
    patches.append(polygon)

collection = PatchCollection(patches, cmap=matplotlib.cm.jet)#, alpha=0.4)
colors = np.random.rand(len(patches))
collection.set_array(colors)
fig = plt.figure(figsize=(5,5))
ax = plt.axes([0,0,1,1])
ax.add_collection(collection)
plt.show()

#plt.text(pos[0,3], pos[1,3]-0.15, "Polygon", ha="center",
#        family=font, size=14)
