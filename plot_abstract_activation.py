import os
import simulation_parameters
import pylab
import numpy as np
import utils
import sys

# load simulation parameters
def return_plot(subplot_code, iteration=None, fig=None):
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
    if iteration == None:
        input_fn_base = params['input_rate_fn_base'] 
    else:
        input_fn_base = params['folder_name'] + 'TrainingInput_%d/abstract_input_' % iteration

    for i in xrange(n_cells):
        input_fn = input_fn_base + str(i) + '.dat'
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
#        if i >= 440:
            ax.annotate('%d' % i, (x+0.005, y+0.005), fontsize=10)

    stim_color = 'k'
    motion_params_fn = "%sTrainingInput_%d/input_params.txt" % (params['folder_name'], iteration)
    mp = np.loadtxt(motion_params_fn)
    ax.quiver(mp[0], mp[1], mp[2], mp[3], angles='xy', scale_units='xy', scale=1, color=stim_color, headwidth=4)
    ax.annotate('Stimulus', (mp[0]+.5*mp[2], mp[1]+0.1), fontsize=12, color=stim_color)


    ax.set_xlim((-.05, 1.05))
    ax.set_ylim((-.05, 1.05))
    ax.set_title(input_fn)
    output_fn = params['figures_folder'] + 'abstract_activation_%d.png' % (iteration)
    print "Saving figure: ", output_fn
    pylab.savefig(output_fn)#, facecolor=bg_color)
    return ax



if __name__ == '__main__':
    utils.sort_cells_by_distance_to_stimulus(448)

    if len(sys.argv) > 1:
        iteration = int(sys.argv[1])
    else:
        iteration = None

    return_plot(111, iteration=iteration)

#    pylab.show()

