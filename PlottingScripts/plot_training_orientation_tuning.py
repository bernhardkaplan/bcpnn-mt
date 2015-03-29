import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
print 'cmd_subfolder', cmd_subfolder
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import numpy as np
import pylab
import matplotlib
from matplotlib import mlab, cm
import utils
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from FigureCreator import plot_params

plot_params['figure.subplot.right'] = 0.95

pylab.rcParams.update(plot_params)

class Plotter(object):

    def __init__(self, params, it_max=None):
        self.params = params
        tp_fn = self.params['tuning_prop_exc_fn']
        print 'Loading', tp_fn
        self.tp = np.loadtxt(tp_fn)
        print 'Loading', self.params['receptive_fields_exc_fn']
        self.rfs = np.loadtxt(self.params['receptive_fields_exc_fn'])

    def plot_training_sample_space_with_orientation(self, mp):
        if mp == None:
            mp = np.loadtxt(self.params['training_stimuli_fn'])
            print 'Loading training stim from:', self.params['training_stimuli_fn']
        training_stim_durations = np.loadtxt(self.params['stim_durations_fn'])

        fig = pylab.figure()#figsize=(12, 12))
        ax1 = fig.add_subplot(111)
        patches = []
        for gid in xrange(self.params['n_exc']):
            ax1.plot(self.tp[gid, 0], self.tp[gid, 4], 'o', c='k', markersize=2)
            ellipse = mpatches.Ellipse((self.tp[gid, 0], self.tp[gid, 4]), self.rfs[gid, 0], self.rfs[gid, 4])
            patches.append(ellipse)
        collection = PatchCollection(patches, alpha=0.1, facecolor='b', edgecolor=None)
        ax1.add_collection(collection)

        for i_stim in xrange(mp[:, 0].size):
            ax1.plot(mp[i_stim, 0], mp[i_stim, 4], '*', markersize=20, color='y', markeredgewidth=1)#, zorder=100)
            x0 = mp[i_stim, 0]
            x1 = mp[i_stim, 0] + mp[i_stim, 2] * (training_stim_durations[i_stim] - params['t_stim_pause']) / 1000.
            ax1.plot((x0, x1), (mp[i_stim, 4], mp[i_stim, 4]), '--', color='r', lw=3)

        ax1.set_title('Training stimuli state space')
        ax1.set_xlabel('Stimulus position') 
        ax1.set_ylabel('Stimulus orientation') 
#        output_fig = self.params['figures_folder'] + 'stimulus_state_space_%.2f_%.2f.png' % (self.params['training_stim_noise_x'], self.params['training_stim_noise_v'])
#        print 'Saving to:', output_fig
#        pylab.savefig(output_fig, dpi=200)
        pylab.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        try:
            params = utils.load_params(sys.argv[1])
            print 'Case 1A'
        except: # its the 
            print 'Case 1B'
            import simulation_parameters
            param_tool = simulation_parameters.parameter_storage()
            params = param_tool.params
            print 'Loading the given training stim from:', sys.argv[1]
            training_stim = np.loadtxt(sys.argv[1])
    else:
        print 'Case 2'
        import simulation_parameters
        param_tool = simulation_parameters.parameter_storage()
        params = param_tool.params
        training_stim = None

    Plotter = Plotter(params)#, it_max=1)
    Plotter.plot_training_sample_space_with_orientation(training_stim)
    pylab.show()
