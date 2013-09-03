import sys
import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

#fn = sys.argv[1]
#fn = 'TwoCellTauZiZjESweep_taup10000_vstim0.50_prex0.00_u0.50_dx0.50_0.00/tauzizje_sweep_taup10000_x00.00_u00.50_vstim0.50.dat'
fn = 'TwoCellTauZiZjESweep_taup50000_vstim0.50_prex0.00_u0.50_dx0.50_0.00/tauzizje_sweep_taup50000_x00.00_u00.50_vstim0.50.dat'
d = np.loadtxt(fn)

 # 0    1     2    3      4         5      6     7          8     9     10    11
#(dx, dv, v_stim, tau_zi, tau_zj, tau_e, tau_p, v_stim, w_max, w_end, w_avg, t_max)
x_axis_idx = 3
y_axis_idx = 4
z_axis_idx = 5 # w_avg at the end
x_label = '$\\tau_{z_i}$'
y_label = '$\\tau_{z_j}$'
z_label = '$\\tau_{e}$'
x_data = d[:, x_axis_idx]
y_data = d[:, y_axis_idx]
z_data = d[:, z_axis_idx]

if z_axis_idx == 4:
    z_label = '$w_{max}$'
elif z_axis_idx == 6:
    z_label = '$w_{avg, end}$'

fig = pylab.figure()
ax = Axes3D(fig)

colored = True

if (colored):
    color_code_axis = 10
    colorbar_label = '$w_{avg, end}$'
    code = d[:, color_code_axis]
    min_4d = np.min(code)
    max_4d = np.max(code)
    range_4d = float(max_4d - min_4d)
    norm = matplotlib.mpl.colors.Normalize(vmin=min_4d, vmax=max_4d)
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet)
    m.set_array(np.arange(min_4d, max_4d, 0.01))
    colors = m.to_rgba(code)
    cax = ax.scatter(x_data, y_data, z_data, c=colors, marker='o', linewidth='5', edgecolor=colors)

else:
    cax = ax.scatter(x_data, y_data, z_data, marker='o')

#ax.set_title(fn)
ax.set_xlabel(x_label, fontsize=24)
ax.set_ylabel(y_label, fontsize=24)
ax.set_zlabel(z_label, fontsize=24)

if colored: 
    cb = fig.colorbar(m, ax=ax, shrink=0.8)
#    cb.set_label('$t_{max}$ [ms]', fontsize=24)
    cb.set_label(colorbar_label, fontsize=24)

pylab.show()
