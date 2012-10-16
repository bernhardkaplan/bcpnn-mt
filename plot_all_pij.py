import os
import simulation_parameters
import pylab
import numpy as np
import utils
import sys

network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

if len(sys.argv) > 1:
    fn = sys.argv[1]
else:
#    fn = raw_input('Filename to plot:\n')
    fn = params['bcpnntrace_folder'] + 'all_pij.dat'

d = np.loadtxt(fn)
pij_max = d[:, 2].max()
pij_min = d[:, 2].min()
print 'pij min max', pij_min, pij_max, d[:, 2].argmax()

tp_fn = params['tuning_prop_means_fn']
tp = np.loadtxt(tp_fn)

bg_color = 'blue'
ms = 5# markersize for scatterplots
pylab.rcParams['lines.markeredgewidth'] = 0

fig = pylab.figure()#facecolor=bg_color)
ax = fig.add_subplot(111, axisbg=bg_color)
for i in xrange(d[:, 0].size):
    src, tgt, pij = d[i, :]

    h = (1. - (pij - pij_min) / pij_max) * 240.
    l = 0.5
    s = 1. # saturation
    assert (0 <= h and h < 360)
    assert (0 <= l and l <= 1)
    assert (0 <= s and s <= 1)
    (r, g, b) = utils.convert_hsl_to_rgb(h, s, l)
    ax.plot(src, tgt, 's', c=(r,g,b), markersize=ms)
pylab.show()

#    x, y, u, v = tp[gid, :]
#    print 'tp[%d]:' % (gid), tp[gid, :]
#    ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1)#, headwidth=6)


    # plot stimulus
#    ax.quiver(mp[0], mp[1], mp[2], mp[3], angles='xy', scale_units='xy', scale=1, color='y', headwidth=6)
#    ax.annotate('Stimulus', (mp[0]+mp[2], mp[1]+0.1), fontsize=12)

#    ax.set_xlim((0, 1))
#    ax.set_ylim((0, 1))
    #output_fn_fig = 'delme_test.png'
    #print "Saving figure: ", output_fn_fig
    #pylab.savefig(output_fn_fig)#, facecolor=bg_color)

#if __name__ == '__main__':
#    return_plot(subplot_code=111)

