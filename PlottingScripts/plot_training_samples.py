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
        tp_fn = self.params['tuning_prop_means_fn']
        print 'Loading', tp_fn
        self.tp = np.loadtxt(tp_fn)
        print 'Loading', self.params['receptive_fields_exc_fn']
        self.rfs = np.loadtxt(self.params['receptive_fields_exc_fn'])


    def plot_training_sample_space(self, plot_process=False):
        fn = self.params['training_sequence_fn']
        print 'Loading training stimuli data from:', fn
        d = np.loadtxt(fn)

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
        for i_ in xrange(self.params['n_stim']):
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
                x1 = mp[0] + mp[2] * self.params['t_training_stim'] / 1000.
                if (x1 < 0.):
#                    x_ = self.get_torus_trajectory(mp[0], mp[2])
                    ax1.plot((x0, 0), (mp[2], mp[2]), '--', color='r', lw=3)
                    x1 = x1 % 1.
                    ax1.plot((1, x1), (mp[2], mp[2]), '--', color='r', lw=3)

                elif (x1 > 1.):
                    ax1.plot((x0, 1.), (mp[2], mp[2]), '--', color='r', lw=3)
                    x1 = x1 % 1.
                    ax1.plot((0, x1), (mp[2], mp[2]), '--', color='r', lw=3)
                else:
                    ax1.plot((x0, x1), (mp[2], mp[2]), '--', color='r', lw=3)
#                idx_stop = (i_ + 1) * self.params['n_iterations_per_stim']
#                mps = d[idx:idx_stop, :]
                ax1.plot(x1, mp[2], '*', markersize=10, color='y', markeredgewidth=1)#, zorder=100)
                ellipse = mpatches.Ellipse((x1, mp[2]), self.params['blur_X'], self.params['blur_V'], linewidth=0)
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
        collection = PatchCollection(patches)#, alpha=0.1)
        ax1.add_collection(collection)


        ax1.set_title('Training stimuli state space')
        ax1.set_xlabel('Stimulus position') 
        ax1.set_ylabel('Stimulus speed vx') 
        output_fig = params['figures_folder'] + 'stimulus_state_space_%.2f_%.2f.png' % (self.params['training_stim_noise_x'], self.params['training_stim_noise_v'])
        print 'Saving to:', output_fig
        pylab.savefig(output_fig, dpi=200)



    def plot_precomputed_actions(self, plot_cells=True):

        supervisor_states = np.loadtxt(self.params['supervisor_states_fn'])
        action_indices = np.loadtxt(self.params['action_indices_fn'])
#        d = np.loadtxt(self.params['motion_params_precomputed_fn'])
        d = np.loadtxt(self.params['training_sequence_fn'])

        patches = []
        fig = pylab.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(111)

        # define the colormap
        cmap = matplotlib.cm.jet
        # extract all colors from the cmap
        cmaplist = [cmap(i) for i in xrange(cmap.N)]
        # force the first color entry to be grey #cmaplist[0] = (.5,.5,.5,1.0)
        # create the new map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

        # define the bins and normalize
        bounds = range(self.params['n_actions'])
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        m.set_array(np.arange(bounds[0], bounds[-1], 1.))
        cb = fig.colorbar(m)
        cb.set_label('Action indices')#, fontsize=24)

        if plot_cells:
            for gid in xrange(self.params['n_exc_mpn']):
                ax1.plot(self.tp[gid, 0], self.tp[gid, 2], 'o', c='k', markersize=2)
                ellipse = mpatches.Ellipse((self.tp[gid, 0], self.tp[gid, 2]), self.rfs[gid, 0], self.rfs[gid, 2], linewidth=0, alpha=0.1)
                ellipse.set_facecolor('b')
                patches.append(ellipse)
                ax1.add_artist(ellipse)

        colors = m.to_rgba(action_indices)
        for i_ in xrange(self.params['n_stim']):

            mp = d[i_, :]
            ax1.plot(mp[0], mp[2], '*', markersize=10, color=colors[i_], markeredgewidth=1)
#            ax1.scatter(mp[0], mp[2], marker='*', s=10, c=color)
#            ax1.plot(mp[0], mp[2], '*', markersize=10, color='y', markeredgewidth=1)
            ellipse = mpatches.Ellipse((mp[0], mp[2]), self.params['blur_X'], self.params['blur_V'], linewidth=0, alpha=0.2)
            ellipse.set_facecolor('r')
            patches.append(ellipse)
            ax1.add_artist(ellipse)

#        ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
#        cb = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
        collection = PatchCollection(patches)
        ax1.add_collection(collection)
        ax1.set_title('Training stimuli state space')
        ax1.set_xlabel('Stimulus position') 
        ax1.set_ylabel('Stimulus speed vx') 
        output_fig = params['figures_folder'] + 'stimulus_state_space_with_precomputed_actions_%.2f_%.2f.png' % (self.params['training_stim_noise_x'], self.params['training_stim_noise_v'])
        print 'Saving to:', output_fig
        pylab.savefig(output_fig, dpi=200)




if __name__ == '__main__':

    if len(sys.argv) > 1:
        param_fn = sys.argv[1]
        if os.path.isdir(param_fn):
            param_fn += '/Parameters/simulation_parameters.json'
        import json
        f = file(param_fn, 'r')
        print 'Loading parameters from', param_fn
        params = json.load(f)

    else:
        import simulation_parameters
        param_tool = simulation_parameters.parameter_storage()
        params = param_tool.params

    
    Plotter = Plotter(params)#, it_max=1)
    Plotter.plot_training_sample_space(plot_process=True)
#    Plotter.plot_precomputed_actions()
    pylab.show()
