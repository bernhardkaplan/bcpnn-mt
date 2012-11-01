import simulation_parameters
import numpy as np
import pylab

class BasicPlotter(object):
    def __init__(self, params=None, **kwargs):

        print 'BasicPlotter'
        if params == None:
            self.network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
            self.params = self.network_params.load_params()                       # params stores cell numbers, etc as a dictionary
        else:
            self.params = params

        self.subfig_cnt = 1
        self.n_fig_x = kwargs.get('n_fig_x', 2)
        self.n_fig_y = kwargs.get('n_fig_y', 2)
        self.tuning_prop = np.loadtxt(self.params['tuning_prop_means_fn'])
        assert (self.tuning_prop[:, 0].size == self.params['n_exc']), 'Number of cells does not match in %s and simulation_parameters!\n Wrong tuning_prop file?' % self.params['tuning_prop_means_fn']

        # figure details
        fig_width_pt = 800.0  # Get this from LaTeX using \showthe\columnwidth
        inches_per_pt = 1.0/72.27               # Convert pt to inch
        golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fig_height = fig_width*golden_mean      # height in inches
#        fig_size =  [fig_width,fig_height]
        fig_size =  [fig_height,fig_width]
        params = {#'backend': 'png',
                  'titel.fontsize': 16,
                  'axes.labelsize' : 12,
                  'text.fontsize': 12,
                  'figure.figsize': fig_size}
        pylab.rcParams.update(params)

    def create_fig(self):
        self.fig = pylab.figure()
        pylab.subplots_adjust(left=0.2)
        pylab.subplots_adjust(hspace=0.35)
        pylab.subplots_adjust(wspace=0.35)
        self.subfig_cnt = 1


    def update_subfig_cnt(self):
        self.subfig_cnt += 1
        self.subfig_cnt = self.subfig_cnt % (self.n_fig_x * self.n_fig_y)
#    def say_hello(self):
#        print 'hello'
