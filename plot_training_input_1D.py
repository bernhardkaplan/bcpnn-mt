import sys
import os
import pylab
import numpy as np
import random

class TrainingInputPlotter(object):

    def __init__(self, params):
        self.params = params
        self.motion_params = np.loadtxt(self.params['training_sequence_fn'])
        self.tuning_prop = np.loadtxt(self.params['tuning_prop_means_fn'])
        self.color_list = ['b', 'g', 'r', 'k', 'y', 'c', 'm', '#00f80f', '#deff00', '#ff00e4', '#00ffe6']

    def plot_training_sequence(self):
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
            mp[i, 0], mp[i, 2] = self.motion_params[i, :]
            mp[i, 1] = i
            print 'debug', i, mp[i, :]
            ax.annotate('(%.2f, %.2f)' % (mp[i, 0], mp[i, 2]), (mp[i, 0] - .1, mp[i, 1] + .2))
        
        ax.quiver(mp[:, 0], mp[:, 1], mp[:, 2], mp[:, 3], \
                  angles='xy', scale_units='xy', scale=1, headwidth=4, pivot='tail')#, width=0.007)

        xmax = mp[np.argmax(mp[:, 0] + mp[:, 2]), 0] + mp[np.argmax(mp[:, 0] + mp[:, 2]), 2] 
        xmin = mp[np.argmin(mp[:, 0] + mp[:, 2]), 0] + mp[np.argmin(mp[:, 0] + mp[:, 2]), 2] 
#        ax.set_xlim((mp[:, 0].min() + mp[mp[:, 0].argmin(), 2] * 1.1, mp[:, 0].max() + mp[mp[:, 0].argmax(), 2] * 1.1))
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((-.5, n_stim + .5))

        return fig

    def plot_training_input(self, gids_to_plot):

        n_stim = self.motion_params[:, 0].size

        n_cells_to_plot = len(gids_to_plot)
        tp = np.zeros((n_cells_to_plot, 5))

        fig = pylab.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        ax1.set_title('Tuning and input into example cells')
        ax1.set_xlabel('Cell x-pos')
        ax1.set_ylabel('GID')

        ax2.set_xlabel('Time [ms]')
        ax2.set_ylabel('Input rate [Hz]')

        rate_curves = []
        rate_max = 0.
        for i in xrange(n_cells_to_plot):

            gid = gids_to_plot[i]

            # ax2 - input rates
            input_rate = np.load(self.params['input_rate_fn_base'] + str(gid) + '.npy')
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
        self.plot_stimulus_borders_vertically(ax2, ymax=rate_max)
        return fig



    def plot_stimulus_borders_vertically(self, ax, ymax=None):

        if ymax == None:
            ymax = ax.get_ylim()[1]
        n_stim = self.motion_params[:, 0].size
        for i in xrange(n_stim):
            t0 = i * self.params['t_training_stim']
            t1 = (i + 1) * self.params['t_training_stim']
            ax.plot((t0, t0), (0, ymax), ls='--', c='k')
            ax.plot((t1, t1), (0, ymax), ls='--', c='k')
            ax.annotate('%d' % i, (t0 + .5 * self.params['t_training_stim'], 0.90 * ymax))






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
                'figure.subplot.hspace': .3, 
                'lines.markeredgewidth' : 0}
    pylab.rcParams.update(rcParams)

    TIP = TrainingInputPlotter(params)
    fig1 = TIP.plot_training_sequence()

    random.seed(1)
    n_gids_to_plot = 10
    gids_to_plot = random.sample(np.loadtxt(params['gids_to_record_fn'], dtype=np.int), n_gids_to_plot)
    print 'gids_to_plot: ', gids_to_plot
    fig2 = TIP.plot_training_input(gids_to_plot)

    output_fn1 = params['figures_folder'] + 'training_sequence_1D.png'
    fig1.savefig(output_fn1)

    pylab.show()


