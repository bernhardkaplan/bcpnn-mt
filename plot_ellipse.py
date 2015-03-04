import pylab
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

patches = []
fig = pylab.figure()
ax = fig.add_subplot(111)

patches_1 = []
patches_2 = []
ellipse1 = mpatches.Ellipse((.5, .5), .1, .1, alpha=0.1)
#ellipse.set_facecolor('r')
ellipse1.set_alpha(0.1)
patches_1.append(ellipse1)

ellipse2 = mpatches.Ellipse((.1, .1), .1, .1, alpha=0.8)
#ellipse.set_facecolor('b')
ellipse2.set_alpha(0.8)
patches_2.append(ellipse2)

#collection = PatchCollection(patches)
collection_1 = PatchCollection(patches_1, facecolor=['r'], alpha=0.1)
collection_2 = PatchCollection(patches_2, facecolor=['b'], alpha=0.8)
ax.add_collection(collection_1)
ax.add_collection(collection_2)

pylab.show()
