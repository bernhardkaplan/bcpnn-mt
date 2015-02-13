import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import pylab
import numpy as np
import random
from PlottingScripts.plot_input import get_cell_gids_with_input_near_tp

class TrainingInputPlotter(object):

    def __init__(self, params):
        self.params = params
        if self.params['training_run']:
            self.motion_params = np.loadtxt(self.params['training_stimuli_fn'])
        else:
            self.motion_params = np.loadtxt(self.params['test_sequence_fn'])
        self.tuning_prop = np.loadtxt(self.params['tuning_prop_exc_fn'])
        self.color_list = ['b', 'g', 'r', 'k', 'y', 'c', 'm', '#00f80f', '#deff00', '#ff00e4', '#00ffe6']

    def plot_training_stimuli(self):
        """
        plots the 1D training sequence

         ^ stimulus
         | number
         |
         |      ->
         |    <----
         |  ->
         +---------------->
            x-start-pos
        """

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.set_ylabel('Training stimulus')
        ax.set_xlabel('Start x-position')

        n_stim = self.motion_params[:, 0].size
        mp = np.zeros((n_stim, 4))
        for i in xrange(n_stim):
#            mp[i, 0], mp[i, 1], mp[i, 2] = self.motion_params[i, :]
            mp[i, :] = self.motion_params[i, :]
            mp[i, 1] = i
            print 'Debug stim and motion_params', i, mp[i, :]
            ax.annotate('(%.2f, %.2f)' % (mp[i, 0], mp[i, 2]), (max(0, mp[i, 0] - .1), mp[i, 1] + .2))
        
        ax.quiver(mp[:, 0], mp[:, 1], mp[:, 2], mp[:, 3], \
                  angles='xy', scale_units='xy', scale=1, headwidth=4, pivot='tail')#, width=0.007)

        xmax = mp[np.argmax(mp[:, 0] + mp[:, 2]), 0] + mp[np.argmax(mp[:, 0] + mp[:, 2]), 2] 
        xmin = min(np.min(mp[:, 0]), 0)
#        xmin = mp[np.argmin(mp[:, 0] + mp[:, 2]), 0] + mp[np.argmin(mp[:, 0] + mp[:, 2]), 2] 
#        ax.set_xlim((mp[:, 0].min() + mp[mp[:, 0].argmin(), 2] * 1.1, mp[:, 0].max() + mp[mp[:, 0].argmax(), 2] * 1.1))
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((-.5, n_stim + .5))

        return fig

    def plot_training_input(self, gids_to_plot):

        n_stim = self.motion_params[:, 0].size

        n_cells_to_plot = len(gids_to_plot)
        tp = np.zeros((n_cells_to_plot, 4))

        fig = pylab.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        ax1.set_title('Tuning properties of example cells')
        ax1.set_xlabel('Cell x-pos')
        ax1.set_ylabel('GID')

        ax2.set_title('Input into example cells')
        ax2.set_xlabel('Time [ms]')
        ax2.set_ylabel('Input rate [Hz]')

        rate_curves = []
        rate_max = 0.
        t_stim_durations = np.loadtxt(self.params['training_stim_durations_fn'])
        for i in xrange(n_cells_to_plot):
            gid = gids_to_plot[i]
            for stim_idx in range(self.params['stim_range'][0], self.params['stim_range'][1]):

            # ax2 - input rates
#            input_rate = np.loadtxt(self.params['input_rate_fn_base'] + str(gid) + '.dat')
                input_fn = self.params['input_rate_fn_base'] + '%d_%d.dat' % (gid, stim_idx)
                if os.path.exists(input_fn):
                    input_rate = np.loadtxt(input_fn)
                else:
                    input_rate = np.zeros(t_stim_durations[stim_idx] / self.params['dt_rate'])
                rate_max = max(rate_max, input_rate.max())
                t_axis = np.arange(0, input_rate.size) * self.params['dt_rate']
                plot, = ax2.plot(t_axis, input_rate, lw=2, label=gid, c=self.color_list[i % len(self.color_list)])
                rate_curves.append(plot)

            # ax1 - cell tuning
            tp[i, :] = self.tuning_prop[gid, :]
            tp[i, 1] = i / 10.
            ax1.annotate('(%.2f, %.2f)' % (tp[i, 0], tp[i, 2]), (tp[i, 0] - .05, tp[i, 1] + .02))

#        handles, labels = ax2.get_legend_handles_labels()
        ax1.quiver(tp[:, 0], tp[:, 1], tp[:, 2], tp[:, 3], \
                  angles='xy', scale_units='xy', scale=1, \
                  headwidth=4, pivot='tail')#, width=0.007)
#                  color = self.color_list[:n_cells_to_plot], headwidth=4, pivot='tail')#, width=0.007)
        ax1.set_ylim((-.05, n_cells_to_plot / 10. + 0.05))
        ax1.set_yticks(tp[:, 1])
        ax1.set_yticklabels(gids_to_plot)
        xmax = tp[np.argmax(tp[:, 0] + tp[:, 2]), 0] + tp[np.argmax(tp[:, 0] + tp[:, 2]), 2] 
        ax1.set_xlim((-.05, xmax + .05))
#        ax1.set_xlim((0, tp[:, 0].max() + tp[tp[:, 0].argmax(), 2] * 1.1))


        ax2.legend(rate_curves, gids_to_plot)
        self.plot_stimulus_borders_vertically(ax2, ylim=ax2.get_ylim())
        return fig



    def plot_stimulus_borders_vertically(self, ax, ylim=None):

        if ylim == None:
            ylim = ax.get_ylim()
        n_stim = self.motion_params[:, 0].size

        training_stim_duration = np.loadtxt(self.params['training_stim_durations_fn'])
        for i_stim in xrange(n_stim):
            t0 = training_stim_duration[:i_stim].sum()
            t1 = training_stim_duration[:i_stim+1].sum()
            ax.plot((t0, t0), (0, ylim[1]), ls='--', c='k')
            ax.plot((t1, t1), (0, ylim[1]), ls='--', c='k')
            ax.text(t0 + .5 * (t1 - t0), 0.90 * ylim[1], '%d' % i_stim)






if __name__ == '__main__':
    if len(sys.argv) > 1:
        param_fn = sys.argv[1]
        if os.path.isdir(param_fn):
            param_fn += '/Parameters/simulation_parameters.json'
        print 'Loading parameters from', param_fn
        import json
        f = file(param_fn, 'r')
        params = json.load(f)

    else:
        print '\nPlotting the default parameters give in simulation_parameters.py\n'
        # load simulation parameters
        import simulation_parameters
        ps = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
        params = ps.load_params()                       # params stores cell numbers, etc as a dictionary

    rcParams = { 'axes.labelsize' : 18,
                'label.fontsize': 18,
                'xtick.labelsize' : 16, 
                'ytick.labelsize' : 16, 
                'axes.titlesize'  : 20,
                'legend.fontsize': 9, 
                'figure.subplot.hspace': .4, 
                'lines.markeredgewidth' : 0}
    pylab.rcParams.update(rcParams)

    TIP = TrainingInputPlotter(params)
    fig1 = TIP.plot_training_stimuli()

    random.seed(1)
    n_gids_to_plot = 3

    tp_params = (0.2, 0.5, 0.7, 0.)
    stim_range = params['stim_range']
    gids_to_plot = get_cell_gids_with_input_near_tp(params, tp_params, stim_range, n_cells=3)
    print 'gids_to_plot: ', gids_to_plot
    fig2 = TIP.plot_training_input(gids_to_plot)

    output_fn1 = params['figures_folder'] + 'training_stimuli_1D.png'
    fig1.savefig(output_fn1)

    pylab.show()


