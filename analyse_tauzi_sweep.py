"""
    output_fn = 'sweep_dv_vstim_tauzj%d_preCellWellTuned.dat' % (bcpnn_params['tau_j'])
"""
import sys
import numpy as np
import pylab
import matplotlib

fn = sys.argv[1]

d = np.loadtxt(fn)


x_axis_idx = 2
y_axis_idx = 6
#z_axis_idx = 6 # w_avg at the end

x_data = d[:, x_axis_idx]
y_data = d[:, y_axis_idx]


#z_data = d[:, z_axis_idx]




#x_axis = np.around(np.arange(0.05, 0.75, 0.05), decimals=2)
#y_axis = np.zeros(x_axis.size)

for i_, dx in enumerate(x_axis):
    idx_dx = d[:, 0] == dx
    print 'idx_dx', idx_dx
    d_dx = d[idx_dx, :]
    print dx, d_dx.size#, idx_dx
    idx_wmax = d_dx[:, 2].argmax()
    tau_zi_wmax = d_dx[idx_wmax, 1]
    print dx, 'tau_zi_wmax:', tau_zi_wmax
    y_axis[i_] = tau_zi_wmax

fig = pylab.figure()
ax = fig.add_subplot(111)
ax.plot(x_axis, y_axis, lw=2)
ax.set_xlabel('Distance between cells', fontsize=24)
ax.set_ylabel('$\\tau_{z_i}(w_{max})$', fontsize=24)
ax.set_title('The $\\tau_{z_i}$ giving the maximum weight\ndepends on the distance between the cells', fontsize=24)

pylab.show()
