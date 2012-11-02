import sys
import numpy
import pylab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm

fn = sys.argv[1]
d = numpy.loadtxt(fn)


fig = pylab.figure()
ax = Axes3D(fig)

pylab.rcParams['lines.markeredgewidth'] = 0

colored = True
# map centroid numbers to different colors
if (colored):
    color_code_axis = 5
    code = d[:, color_code_axis]
    norm = matplotlib.mpl.colors.Normalize(vmin=code.min(), vmax=code.max())
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.jet)
    rgba_colors = m.to_rgba(code)
    p = ax.scatter(d[:,0], d[:,1], d[:,3], c=numpy.array(rgba_colors), marker='o', linewidth='5', edgecolor=rgba_colors)#, cmap='seismic')
    m.set_array(code)
    fig.colorbar(m)
else:
    p = ax.scatter(d[:,0], d[:,1], d[:,2], marker='o', linewidth='5')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
pylab.show()
