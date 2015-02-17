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


    def plot_training_sample_space(self, d=None, plot_process=False, stim_lim=None):
        if d == None:
            fn = self.params['training_stimuli_fn']
            print 'Loading training stimuli data from:', fn
            d = np.loadtxt(fn)
            stim_duration = np.loadtxt(self.params['stim_durations_fn'])
            n_stim = d[:, 0].size
        else:
            n_stim = d[:, 0].size
            stim_duration = np.zeros(n_stim)
            for i_ in xrange(n_stim):
                stim_params = d[i_, :]
                t_exit = utils.compute_stim_time(stim_params)
                stim_duration[i_] = min(t_exit, self.params['t_training_max']) + self.params['t_stim_pause']
        if stim_lim == None:
            stim_lim = (0, n_stim)

        fig = pylab.figure()#figsize=(12, 12))
        ax1 = fig.add_subplot(111)

        patches = []
            
        for gid in xrange(self.params['n_exc']):
            ax1.plot(self.tp[gid, 0], self.tp[gid, 2], 'o', c='k', markersize=2)
            ellipse = mpatches.Ellipse((self.tp[gid, 0], self.tp[gid, 2]), self.rfs[gid, 0], self.rfs[gid, 2], linewidth=0, alpha=0.1)
            ellipse.set_facecolor('b')
            patches.append(ellipse)
            ax1.add_artist(ellipse)

        # plot the stimulus start points
        for i_ in xrange(stim_lim[0], stim_lim[1]):
            if plot_process:
                mp = d[i_, :]
#                idx = i_ * self.params['n_iterations_per_stim']
#                mp = d[idx, :]
                ax1.plot(mp[0], mp[2], '*', markersize=20, color='y', markeredgewidth=1)#, zorder=100)
                ellipse = mpatches.Ellipse((mp[0], mp[2]), self.params['blur_X'], self.params['blur_V'], linewidth=0, alpha=0.2)
                ellipse.set_facecolor('r')
                patches.append(ellipse)
                ax1.add_artist(ellipse)

                # stop position
                x0 = mp[0]
                x1 = mp[0] + mp[2] * training_stim_duration[i_] / 1000.
#                x1 = mp[0] + mp[2] * self.params['t_training_stim'] / 1000.
                if (x1 < 0.):
#                    x_ = self.get_torus_trajectory(mp[0], mp[2])
                    ax1.plot((x0, 0), (mp[2], mp[2]), '--', color='r', lw=3)
                    # if wrap-around:
#                    x1 = x1 % 1.
#                    ax1.plot((1, x1), (mp[2], mp[2]), '--', color='r', lw=3)

                elif (x1 > 1.):
                    ax1.plot((x0, 1.), (mp[2], mp[2]), '--', color='r', lw=3)
                    # if wrap-around:
#                    x1 = x1 % 1.
#                    ax1.plot((0, x1), (mp[2], mp[2]), '--', color='r', lw=3)
                else:
                    ax1.plot((x0, x1), (mp[2], mp[2]), '--', color='r', lw=3)
#                idx_stop = (i_ + 1) * self.params['n_iterations_per_stim']
#                mps = d[idx:idx_stop, :]
                    ax1.plot(x1, mp[2], '*', markersize=10, color='y', markeredgewidth=1)#, zorder=100)
                    ellipse = mpatches.Ellipse((x1, mp[2]), self.params['blur_X'], self.params['blur_V'], linewidth=0, alpha=0.2)
                ellipse.set_facecolor('r')
                patches.append(ellipse)
                ax1.add_artist(ellipse)

            else:
                mp = d[i_, :]
                ax1.plot(mp[0], mp[2], '*', markersize=10, color='y', markeredgewidth=1)
                ellipse = mpatches.Ellipse((mp[0], mp[2]), self.params['blur_X'], self.params['blur_V'], linewidth=0, alpha=0.2)
                ellipse.set_facecolor('r')
                patches.append(ellipse)
                ax1.add_artist(ellipse)
            ax1.text(mp[0] + 0.02, mp[2] + 0.1, '%d' % i_, fontsize=16, color='k')
        collection = PatchCollection(patches)#, alpha=0.1)
        ax1.add_collection(collection)


        ax1.set_title('Training stimuli state space')
        ax1.set_xlabel('Stimulus position') 
        ax1.set_ylabel('Stimulus speed vx') 
        output_fig = self.params['figures_folder'] + 'stimulus_state_space_%.2f_%.2f.png' % (self.params['training_stim_noise_x'], self.params['training_stim_noise_v'])
        print 'Saving to:', output_fig
        pylab.savefig(output_fig, dpi=200)




if __name__ == '__main__':

    training_stim = None
    plot_process = False
#    stim_lim = None
    stim_lim = (0, 50)
    if len(sys.argv) > 1:
        try:
            params = utils.load_params(sys.argv[1])
            print 'Case 1A'
        except: # its the 
            print 'Case 1B'
            import simulation_parameters
            param_tool = simulation_parameters.parameter_storage()
            params = param_tool.params
            training_stim = np.loadtxt(sys.argv[1])
    else:
        print 'Case 2'
        import simulation_parameters
        param_tool = simulation_parameters.parameter_storage()
        params = param_tool.params
    
    Plotter = Plotter(params)#, it_max=1)
    Plotter.plot_training_sample_space(training_stim, plot_process=plot_process, stim_lim=stim_lim)
    pylab.show()
