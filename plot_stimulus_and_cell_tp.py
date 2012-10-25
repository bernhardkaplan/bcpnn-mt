import os
import simulation_parameters
import pylab
import numpy as np
import utils
import sys

# load simulation parameters
def return_plot(cell_gids=[], subplot_code=111, fig=None, input_fn_base=None, motion_params=None):
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
    if input_fn_base == None:
        input_fn_base = params['input_rate_fn_base']

    fn = params['tuning_prop_means_fn']
    if motion_params == None:
        mp = params['motion_params']
    else:
        mp = motion_params
    tp = np.loadtxt(fn)

    if len(cell_gids) == 0:
        n_cells =15 
        cell_gids = np.random.randint(0, params['n_exc'], n_cells)
    n_cells = len(cell_gids)
    ms = 5# markersize for scatterplots
    bg_color = 'w'
    pylab.rcParams['lines.markeredgewidth'] = 0

    input_sum = np.zeros(n_cells)
    for i, gid in enumerate(cell_gids):
        input_fn = input_fn_base + str(gid) + '.dat'
        rate = np.loadtxt(input_fn)
        input_sum[i] = rate.sum()

    input_max = input_sum.max()
    if fig == None:
        fig = pylab.figure(facecolor=bg_color)
    ax = fig.add_subplot(subplot_code)
    colors = ['b', 'g']
    for i, gid in enumerate(cell_gids):
        x, y, u, v = tp[gid, :]
#        print 'tp[%d]:' % (gid), tp[gid, :]

#        h = 240.
#        l = 1. - 0.5 * input_sum[i] / input_max
#        s = 1. # saturation
#        assert (0 <= h and h < 360)
#        assert (0 <= l and l <= 1)
#        assert (0 <= s and s <= 1)
#        (r, g, b) = utils.convert_hsl_to_rgb(h, s, l)
#        ax.plot(x, y, 'o', c=(r,g,b), markersize=ms)

        ax.plot(x, y, 'o', c=colors[i%len(colors)], markersize=ms)
        ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1)#, headwidth=6)


    # plot stimulus
    stim_color = 'y'
    ax.quiver(mp[0], mp[1], mp[2], mp[3], angles='xy', scale_units='xy', scale=1, color=stim_color, headwidth=6)
    ax.annotate('Stimulus', (mp[0]+.5*mp[2], mp[1]+0.1), fontsize=12, color=stim_color)

    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    #output_fn_fig = 'delme_test.png'
    #print "Saving figure: ", output_fn_fig
    #pylab.savefig(output_fn_fig)#, facecolor=bg_color)
    return ax

if __name__ == '__main__':
    return_plot(subplot_code=111)
    pylab.show()

