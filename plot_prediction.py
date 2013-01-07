import matplotlib
matplotlib.use('Agg')
import pylab
import PlotPrediction as P
import sys
import NeuroTools.parameters as ntp
import simulation_parameters

def plot_prediction(params=None, data_fn=None, inh_spikes = None):

    if params== None:
        network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
#        P = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
        params = network_params.params

    sim_cnt = 0
    if data_fn == None:
        data_fn = params['exc_spiketimes_fn_merged'] + '%d.ras' % (sim_cnt)

#    if inh_spikes == None:
#        inh_spikes = params['inh_spiketimes_fn_merged'] + '%d.ras' % (sim_cnt)

    plotter = P.PlotPrediction(params, data_fn)
    if plotter.no_spikes:
        return

    plotter.compute_v_estimates()
    plotter.compute_theta_estimates()

    # fig 1
    # neuronal level
    output_fn_base = '%s%s_wsigmaX_%.2f_wsigmaV%.2f_pthresh%.1e' % (params['prediction_fig_fn_base'], params['initial_connectivity'], \
            params['w_sigma_x'], params['w_sigma_v'], params['w_thresh_connection'])

    plotter.create_fig()  # create an empty figure
    plotter.plot_rasterplot('exc', 1)               # 1 
    plotter.plot_rasterplot('inh', 2)               # 2 
    plotter.plot_vx_grid_vs_time(3)              # 3 
    plotter.plot_vy_grid_vs_time(4)              # 4 
    output_fn = output_fn_base + '_0.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn)

    # poplation level, short time-scale
    # fig 3
    plotter.create_fig()
    plotter.plot_vx_estimates(1)                 # 1
    plotter.plot_vy_estimates(2)                 # 2
    plotter.plot_theta_estimates(3)              # 3
    plotter.plot_vdiff(4)                        # 4
    output_fn = output_fn_base + '_1.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn)

    # fig 3
    # population level, long time-scale
    plotter.n_fig_x = 1
    plotter.n_fig_y = 4
    pylab.rcParams['legend.fontsize'] = 10
    pylab.subplots_adjust(hspace=0.5)
    plotter.create_fig()
    plotter.plot_fullrun_estimates_vx(1)
    plotter.plot_fullrun_estimates_vy(2)
    plotter.plot_fullrun_estimates_theta(3)
    plotter.plot_nspike_histogram(4)
    output_fn = output_fn_base + '_2.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn)


    plotter.n_fig_x = 1
    plotter.n_fig_y = 1
    plotter.create_fig()
    weights = [plotter.nspikes_binned_normalized[i, :].sum() / plotter.n_bins for i in xrange(plotter.n_cells)]
    plotter.quiver_plot(weights, fig_cnt=1)
    output_fn = output_fn_base + '_quiver.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn)

#    plotter.n_fig_x = 1
#    plotter.n_fig_y = 1
#    time_binsize = plotter.time_binsize
#    for i in xrange(plotter.n_bins):
#        plotter.create_fig()
#        title = 'Predicted directions after spatial marginalization\nt=%.1f - %.1f [ms]' % (i*time_binsize, (i+1)*time_binsize)
#        plotter.quiver_plot(plotter.nspikes_binned_normalized[:, i], title=title, fig_cnt=1)
#        output_fn = params['figures_folder'] + 'quiver_%d.png' % i
#        print 'Saving figure to:', output_fn
#        pylab.savefig(output_fn)

#    plotter.make_infotextbox()

#    pylab.show()


#    plotter.plot_nspikes_binned()               # 1
#    plotter.plot_nspikes_binned_normalized()    # 2
#    plotter.plot_vx_confidence_binned()         # 3
#    plotter.plot_vy_confidence_binned()         # 4

if __name__ == '__main__':
    plot_prediction()

#folder = 'Data_inputstrength_swepng/NoColumns_winit_random_wsigmaX2.50e-01_wsigmaV2.50e-01_winput2.00e-03_finput2.00e+03pthresh1.0e-01_ptow1.0e-02/' 
#params_fn = folder + 'simulation_parameters.info'
#data_fn = folder + 'Spikes/exc_spikes_merged_0.ras'
#inh_spikes = folder + 'Spikes/inh_spikes_0.ras'
#tuning_prop_means_fn = folder + 'Parameters/tuning_prop_means.prm'
#output_fn = folder + 'Figures/prediction_0.png'

#params = ntp.ParameterSet(params_fn)
#params['exc_spiketimes_fn_merged'] = data_fn
#params['tuning_prop_means_fn'] = tuning_prop_means_fn

#new_params = { 'folder_name' : folder}
#PS = simulation_parameters.parameter_storage() # load the current parameters
#params = PS.params
#PS.update_values(new_params)
#print 'debug', PS.params['folder_name']
#data_fn = params['exc_spiketimes_fn_merged'] + '0.ras'
#print 'data_fn: ', data_fn
#exit(1)

#plot_prediction()#params, data_fn, inh_spikes, output_fn)

