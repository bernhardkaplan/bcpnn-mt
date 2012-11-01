
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
    y = p[3] + p[0] / (x - p[1])**p[2]
    return y



input_fn = 'Figures_BlurSweep/nspikes_blur_sweep.dat'
d = np.loadtxt(input_fn)
# input_fn stores:
# (params['blur_X'], params['blur_V'], all_spikes.sum(), all_spikes.mean(), all_spikes.std(), input_spikes.mean(), input_spikes.std())
# input_spikes is the array of spikes for those cells that receive > 0 number of spikes
# all_spikes is for all cells

fig = pylab.figure()
ax = fig.add_subplot(111)


blur_x_start = 0.025
blur_x_stop = 0.95
blur_x_step = 0.025
blur_range = np.arange(blur_x_start, blur_x_stop, blur_x_step)
n = blur_range.size
m = 6
idx_0 = n * m
idx_1 = n * (m + 1)
print 'idx:', idx_0, idx_1

x = d[idx_0:idx_1, 0]
y = 1. / d[idx_0:idx_1, 2]
print 'blur :', x
print 'y:', y
#x = d[::n, 0]
#y = d[::n, 2]
ax.plot(x, y, 'o-')

# fit a function on to the dependecy 
# x vs 1/y --- or ---- blur_ vs 1/nspikes ----> gives the function for the scaling factor 

#guess_params = [1., .1]
#guess_params = [1., -.1, .1, .1]
guess_params = [1., .01, .1, .1]
opt_params = leastsq(residuals_function, guess_params, args=(x, y), maxfev=10000)
print 'opt_params', opt_params[0]
opt_func = peval_function(x, opt_params[0])

#my_func = peval_function(x, [.0, .2, .1, .01])

ax.plot(x, opt_func, lw=3)
#ax.plot(x, my_func, lw=3)

pylab.show()
