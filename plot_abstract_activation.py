import os
import simulation_parameters
import pylab
import numpy as np
import utils
import sys

# load simulation parameters
def return_plot(subplot_code, fig=None):
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

    fn = params['tuning_prop_means_fn']
    tp = np.loadtxt(fn)

    n_cells = tp[:, 0].size
    #n_cells = 10
    ms = 10 # markersize for scatterplots
    bg_color = 'w'
    pylab.rcParams['lines.markeredgewidth'] = 0

    input_sum = np.zeros(n_cells)
    for i in xrange(n_cells):
        input_fn = params['input_rate_fn_base'] + str(i) + '.dat'
        rate = np.loadtxt(input_fn)
        input_sum[i] = rate.sum()

    input_max = input_sum.max()
    if fig == None:
        fig = pylab.figure(facecolor=bg_color)
    ax = fig.add_subplot(subplot_code)
    for i in xrange(n_cells):
        h = 240.
        l = 1. - 0.5 * input_sum[i] / input_max
        s = 1. # saturation
        assert (0 <= h and h < 360)
        assert (0 <= l and l <= 1)
        assert (0 <= s and s <= 1)
        (r, g, b) = utils.convert_hsl_to_rgb(h, s, l)
        x, y, u, v = tp[i, :]
        ax.plot(x, y, 'o', c=(r,g,b), markersize=ms)
        if l < .75:
            ax.annotate('%d' % i, (x+0.01, y+0.01), fontsize=10)

    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    #output_fn_fig = 'delme_test.png'
    #print "Saving figure: ", output_fn_fig
    #pylab.savefig(output_fn_fig)#, facecolor=bg_color)
    return ax

if __name__ == '__main__':
    return_plot(111)
    pylab.show()

