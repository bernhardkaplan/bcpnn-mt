import sys
import numpy as np
import utils
import pylab
from Plotter import BasicPlotter


class PlotOutputActivity(BasicPlotter):
    """
    plot the ANN activity after training
    and the predicted resulting direction
    """

    def __init__(self, iteration =None, **kwargs):
        BasicPlotter.__init__(self, **kwargs)
        if iteration == None:
            iteration = 0
        activity_fn = self.params['activity_folder'] + 'output_activity_%d.dat' % (iteration)
        prediction_fn = self.params['activity_folder'] + 'prediction_%d.dat' % (iteration)
        prediction_error_fn = self.params['activity_folder'] + 'prediction_error_%d.dat' % (iteration)
        print 'activity_fn:', activity_fn
        print 'prediction_fn:', prediction_fn
        print 'prediction_error_fn:', prediction_error_fn
            
        self.activity = np.loadtxt(activity_fn)
        self.prediction = np.loadtxt(prediction_fn)
        self.prediction_error = np.loadtxt(prediction_error_fn)
        self.iteration = iteration
        rcParams = { 'axes.labelsize' : 16,
                    'label.fontsize': 20,
                    'legend.fontsize': 9}
        pylab.rcParams.update(rcParams)
        self.t_axis = self.prediction[:, 0]
        training_input_folder = "%sTrainingInput_%d/" % (self.params['folder_name'], iteration)
        self.stim_params = np.loadtxt(training_input_folder + 'input_params.txt')

#        self.vx_tuning = self.tuning_prop[:, 2]
#        self.vy_tuning = self.tuning_prop[:, 3]

#        vx_min, vx_max = self.vx_tuning.min(), self.vx_tuning.max()
#        vy_min, vy_max = self.vy_tuning.min(), self.vy_tuning.max()
#        n_vx_bins, n_vy_bins = 20, 20
#        vx_grid = np.linspace(vx_min, vx_max, n_vx_bins, endpoint=True)
#        vy_grid = np.linspace(vy_min, vy_max, n_vy_bins, endpoint=True)
#        self.calculate_v_predicted()

        self.create_fig()

    def plot_data_vs_time(self, data, **kwargs):
        xlabel = kwargs.get('xlabel', 'Time [ms]')
        ylabel = kwargs.get('ylabel', 'y')
        update_subfig_cnt = kwargs.get('update_subfig_cnt', True)

        label = kwargs.get('label', None)
        self.ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, self.subfig_cnt)
        self.ax.plot(self.t_axis, data, label=label)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        if update_subfig_cnt :
            self.update_subfig_cnt()

        if label != None:
            self.ax.legend()

    def set_title(self):
        title = 'Stimulus vx=%.2f, vy=%.2f' % (self.stim_params[2], self.stim_params[3])
        self.ax.set_title(title)

#    def calculate_v_predicted(self):
#        self.vx_pred = np.zeros(self.n_time_steps)
#        self.vy_pred = np.zeros(self.n_time_steps)
#        for t in xrange(self.n_time_steps):
#            normed_activity = self.activity[t, :] / self.activity[t, :].sum()
#            self.vx_pred[t] = np.dot(self.vx_tuning, normed_activity)
#            self.vy_pred[t] = np.dot(self.vy_tuning, normed_activity)


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        iteration = 0
    else:
        iteration = int(sys.argv[1])
    P1 = PlotOutputActivity(iteration, n_fig_x=1, n_fig_y=3)
    print 'debug vx:', P1.prediction[:, 1]
    P1.plot_data_vs_time(P1.prediction[:, 1], ylabel='$v_x$', update_subfig_cnt=False, label='Prediction')
    P1.plot_data_vs_time(P1.prediction_error[:, 1], ylabel='$v_x$', label='Error')
    P1.set_title()
    P1.plot_data_vs_time(P1.prediction[:, 2], ylabel='$v_y$', update_subfig_cnt=False, label='Prediction')
    P1.plot_data_vs_time(P1.prediction_error[:, 2], ylabel='$v_y$', label='Error')
    P1.plot_data_vs_time(P1.prediction_error[:, 3], ylabel='$|v_diff|$', label='Absolute prediction error')

    output_fn = P1.params['figures_folder'] + 'ann_prediction_%d.png' % (iteration)
    print 'Saving prediction figure to:', output_fn
    pylab.savefig(output_fn)

#    P2 = PlotOutputActivity(input_fn, n_fig_x=4, n_fig_y=2)
#    n_cells_to_plot = 32
#    idx = np.random.randint(0, P2.params['n_exc'], n_cells_to_plot)
#    for i in xrange(n_cells_to_plot):
#        cell = idx[i]
#        P2.plot_data_vs_time(P2.activity[:, cell], label='%d' % cell, ylabel='Activity')

#    pylab.show()


#n_fig_x = 2
#n_fig_y = 4
#n_plots = n_fig_x * n_fig_y
#fig = pylab.figure()

#np.random.seed(0)




#ax = fig.add_subplot(221)
#ax.plot(t_axis, vx_pred)

#ax = fig.add_subplot(222)
#ax.plot(t_axis, vy_pred)

#plot_grid_vs_time(vx_pred_binned)



#pylab.show()
