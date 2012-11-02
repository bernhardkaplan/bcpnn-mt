
"""

Based on the blur_X _V parameters, the network receives very different amounts of input spikes.
Hence, to compare the effect of different blur_ values we need to scale w_input_exc and f_max
in order to stimulate the network with the same (or very similar amount) of excitation.


After creating spike trains for different blur_X _V values (--> run_input_analysis.py) 
and writing the number of input spikes to a file (--> analyse_input.py),
we can now find a relation between blur_X _V and the scaling factors needed for w_input_exc and f_max 
to achieve a balanced input excitation for different blur_X _V parameters.
"""

import pylab
import numpy as np
import matplotlib.mlab as mlab
from scipy.optimize import leastsq


def residuals_function(p, x, y):
    """
    x: normal x coordinate
    p: parameters of the function to fit, e.g. a and b in y = a * x + b
    """
    y1 = peval_function(x, p)
    err = y - y1
    return err 


def peval_function(x, p):
#    return p[0] / x**p[1]
#    y = p[2] * mlab.normpdf(x, p[0], p[1]) + p[3]
#    y = p[3] + p[0] / (x - p[1])**p[2]
#    y = p[0] + p[1] / (x)**p[2] +  p[3] / x**p[4] + p[5] / x
#    y = p[0] + p[1] * np.exp((x - p[2])**2 / p[3]**2)
    y = p[0] + p[1] / x + p[2] / x**2 + p[3] / x**3 + p[4] / x**4 + p[5] / x**5
#    y = p[0] * np.exp(p[1] * (x - p[2])) + p[3]
    return y


input_fn = 'Figures_BlurSweep/nspikes_blur_sweep.dat'
d = np.loadtxt(input_fn)
# input_fn stores:
# (params['blur_X'], params['blur_V'], all_spikes.sum(), all_spikes.mean(), all_spikes.std(), input_spikes.mean(), input_spikes.std())
# input_spikes is the array of spikes for those cells that receive > 0 number of spikes
# all_spikes is for all cells

fig = pylab.figure()
ax = fig.add_subplot(111)

###############
# PLOT NSPIKES 
###############

blur_range = np.unique(d[:, 0])
n = blur_range.size

y_axis_idx = 2
m = 0

idx_0 = n * m
idx_1 = n * (m + 1)
x = d[idx_0:idx_1, 0]
blur_x = d[idx_0:idx_1, 0]
blur_v = d[idx_0:idx_1, 1]
ax.plot(x, d[idx_0:idx_1, y_axis_idx], 'o-', label='blur_v=%.3f' % blur_v.mean())
ax.legend(loc='upper left')
ax.set_ylabel('Average number of input spikes into one cell')
ax.set_xlabel('blur_x')


###############
# PLOT INVERSE
###############
# the desired value depends of course on what you try to balance (all spikes coming into the network, the average, ...)
#desired_value = 15 # desired value for the average number of input spikes
desired_value = 6000 # desired value for the average number of input spikes

fig = pylab.figure()
ax = fig.add_subplot(111)

idx_0 = n * m
idx_1 = n * (m + 1)
x = d[idx_0:idx_1, 0]
y = desired_value / d[idx_0:idx_1, y_axis_idx]
blur_x = d[idx_0:idx_1, 0]
blur_v = d[idx_0:idx_1, 1]
ax.plot(x, y, 'o-', label='inverse')

# fit a function on to the dependecy 
# x vs 1/y --- or ---- blur_ vs 1/nspikes ----> gives the function for the scaling factor 
#guess_params = [1., .01, .1, .01] # 1 / x**c
#guess_params = [1., .0, 2., 0.] # 1 / x**c
#guess_params = [0., .1, 3., 1., 2., 1.] # 1 / x**2 + 1 / x + c
guess_params = [0., 1., 1., 1., 1., 1.]
#guess_params = [0., .1, 1., 1., .1] # inverse gauss
#guess_params = [500., -100., .0, .01] # for exp(-tau/x)
opt_params = leastsq(residuals_function, guess_params, args=(x, y), maxfev=10000)
print 'opt_params', opt_params[0]
opt_func = peval_function(x, opt_params[0])
ax.plot(x, opt_func, lw=3, label='fitted function')

ax.set_title('Fitting a scaling function to %d / data,\ndata: average number of input spikes' % (desired_value))
ax.set_ylabel('desired_value / data')
ax.set_xlabel('blur_x')
#my_func = peval_function(x, guess_params)
#ax.plot(x, my_func, lw=3, label='initial guess')

ax.legend()

##########################################
# PLOT THE EXPECTED NUMBER OF INPUT SPIKES
##########################################
fig = pylab.figure()
ax = fig.add_subplot(111)
#f_max_0 = 5000
y_scaled = opt_func * d[idx_0:idx_1, y_axis_idx]
print 'debug', opt_func[0], d[idx_0:idx_0+1, y_axis_idx]
#print 'debug', opt_func[0] * 1./ d[idx_0:idx_0+1, y_axis_idx]
print 'y_scale', y_scaled
ax.plot(x, y_scaled, label='fit * data')
ax.set_title('Expected number of average input spikes after scaling\nshould be constant at desired_value=%d' % (desired_value))
ax.set_ylabel('Average number of input spikes')
ax.set_xlabel('blur_x')
ax.legend()


####################################################################################
# PLOT THE MAX NUMBER OF INPUT SPIKES (INTO THE BEST TUNED CELL)
####################################################################################
fig = pylab.figure()
ax = fig.add_subplot(111)
y_axis_idx = 6
idx_0 = n * m
idx_1 = n * (m + 1)
x = d[idx_0:idx_1, 0]
blur_x = d[idx_0:idx_1, 0]
blur_v = d[idx_0:idx_1, 1]
ax.plot(x, d[idx_0:idx_1, y_axis_idx], 'o-', label='blur_v=%.3f' % blur_v.mean())
ax.legend(loc='upper left')
ax.set_ylabel('Max number of input spikes')
ax.set_xlabel('blur_x')


pylab.show()
