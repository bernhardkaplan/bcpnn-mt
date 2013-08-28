import sys
import numpy as np
import pylab
import matplotlib

fn = sys.argv[1]
d = np.loadtxt(fn)

x_axis_idx = 0
y_axis_idx = 1
x_label = '$dx$'
y_label = '$dv$'

z_axis_idx = 6 # w_avg at the end

x_data = d[:, x_axis_idx]
y_data = d[:, y_axis_idx]
z_data = d[:, z_axis_idx]

#x_axis = np.unique(x_data, return_index=True, return_inverse=True)
#y_axis = np.unique(y_data)
print 'check xdata', np.unique(x_data, return_index=True, return_inverse=True)
#print 'check xaxis', x_axis
#color_matrix = np.zeros((x_axis.size, y_axis.size))
# as np.

#fig = pylab.figure()

#color_code_axis = z_axis_idx
#colorbar_label = '$w_{avg, end}$'
#code = d[:, color_code_axis]
#min_4d = np.min(code)
#max_4d = np.max(code)
#range_4d = float(max_4d - min_4d)
#norm = matplotlib.mpl.colors.Normalize(vmin=min_4d, vmax=max_4d)
#m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet)
#m.set_array(np.arange(min_4d, max_4d, 0.01))
#colors = m.to_rgba(code)

#ax.set_title(fn)
#ax.set_xlabel(x_label, fontsize=24)
#ax.set_ylabel(y_label, fontsize=24)
#ax.set_zlabel(z_label, fontsize=24)

#if colored: 
#    cb = fig.colorbar(m, ax=ax, shrink=0.8)
#    cb.set_label(colorbar_label, fontsize=24)

#pylab.show()
