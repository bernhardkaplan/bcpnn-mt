import sys
import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

fn = sys.argv[1]
d = np.loadtxt(fn)


fig = pylab.figure()
ax = Axes3D(fig)

#colored = False
colored = True
# map centroid numbers to different colors
if (colored):
    color_code_axis = 4
#    color_code_axis = 3
    code = d[:, color_code_axis]
    colors = []
    h_values = []
    min_4d = np.min(code)
    max_4d = np.max(code)
    range_4d = float(max_4d - min_4d)
    norm = matplotlib.mpl.colors.Normalize(vmin=min_4d, vmax=max_4d)
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet)
    m.set_array(np.arange(min_4d, max_4d, 0.01))
    for i in xrange(len(code)):
        h = (code[i] - min_4d) / range_4d
        h_values.append(h)
        rgb = matplotlib.colors.hsv_to_rgb(np.array([[[h, 1.0, 1.0]]]))[0, 0,:]
        colors.append([rgb[0], rgb[1], rgb[2], 1.])

    cax = ax.scatter(d[:,0], d[:,1], d[:,2], c=np.array(colors), marker='o', linewidth='5', edgecolor=colors)
else:
    ax.scatter(d[:,0], d[:,1], d[:,2], marker='o')

ax.set_xlabel('dx between cells', fontsize=24)
ax.set_ylabel('$\\tau_{z_i}$', fontsize=24)
ax.set_zlabel('$w_{max}$', fontsize=24)
if colored: 
    cb = fig.colorbar(m, ax=ax, shrink=0.8)
    cb.set_label('$t_{max}$ [ms]', fontsize=24)
#    cb.set_label('$w_{end}$', fontsize=24)

pylab.show()
