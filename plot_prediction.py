import matplotlib
matplotlib.use('Agg')
import pylab
import PlotPrediction as P
import sys
import simulation_parameters
import os
import utils

def plot_prediction_2D(params=None, data_fn=None, inh_spikes = None):

    if params== None:
        network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
#        P = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
        params = network_params.params

    if data_fn == None:
        data_fn = params['exc_spiketimes_fn_merged'] + '.ras'

#    if inh_spikes == None:
#        inh_spikes = params['inh_spiketimes_fn_merged'] + '.ras'

#    params['t_sim'] = 1200

    plotter = P.PlotPrediction(params, data_fn)
    pylab.rcParams['axes.labelsize'] = 14
    pylab.rcParams['axes.titlesize'] = 16
    if plotter.no_spikes:
        return

    plotter.compute_v_estimates()
    plotter.compute_position_estimates()
    plotter.compute_theta_estimates()

    # fig 1
    # neuronal level
    output_fn_base = '%s%s_wsigmaX_%.2f_wsigmaV%.2f_delayScale%d_connRadius%.2f_wee%.2f' % (params['prediction_fig_fn_base'], params['connectivity_code'], \
            params['w_sigma_x'], params['w_sigma_v'], params['delay_scale'], params['connectivity_radius'], params['w_tgt_in_per_cell_ee'])


    plotter.create_fig()  # create an empty figure
    pylab.subplots_adjust(left=0.07, bottom=0.07, right=0.97, top=0.93, wspace=0.3, hspace=.2)
    plotter.n_fig_x = 2
    plotter.n_fig_y = 3
    plotter.plot_rasterplot('exc', 1)               # 1 
    plotter.plot_rasterplot('inh', 2)               # 2 
    plotter.plot_vx_grid_vs_time(3)              # 3 
    plotter.plot_vy_grid_vs_time(4)              # 4 
    plotter.plot_x_grid_vs_time(5)
    plotter.plot_y_grid_vs_time(6)
    output_fn = output_fn_base + '_0.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=200)
#    output_fn = output_fn_base + '_0.pdf'
#    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=200)
#    output_fn = output_fn_base + '_0.eps'
#    print 'Saving figure to:', output_fn
#    pylab.savefig(output_fn, dpi=200)



    # poplation level, short time-scale
    plotter.n_fig_x = 3
    plotter.n_fig_y = 2
    plotter.create_fig()
    pylab.rcParams['legend.fontsize'] = 12
    pylab.subplots_adjust(left=0.07, bottom=0.07, right=0.97, top=0.93, wspace=0.3, hspace=.3)
    plotter.plot_vx_estimates(1)
    plotter.plot_vy_estimates(2)
    plotter.plot_vdiff(3)
    plotter.plot_x_estimates(4)
    plotter.plot_y_estimates(5) 
    plotter.plot_xdiff(6)
    output_fn = output_fn_base + '_1.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=200)
#    output_fn = output_fn_base + '_1.pdf'
#    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=200)
#    output_fn = output_fn_base + '_1.eps'
#    print 'Saving figure to:', output_fn
#    pylab.savefig(output_fn, dpi=200)



#    plotter.plot_theta_estimates(5)


    # fig 3
    # population level, long time-scale

#    plotter.n_fig_x = 1
#    plotter.n_fig_y = 4
#    pylab.rcParams['legend.fontsize'] = 10
#    pylab.subplots_adjust(hspace=0.5)
#    plotter.create_fig()
#    plotter.plot_fullrun_estimates_vx(1)
#    plotter.plot_fullrun_estimates_vy(2)
#    plotter.plot_fullrun_estimates_theta(3)
#    plotter.plot_nspike_histogram(4)
#    output_fn = output_fn_base + '_2.png'
#    print 'Saving figure to:', output_fn
#    pylab.savefig(output_fn, dpi=200)
#    output_fn = output_fn_base + '_2.pdf'
#    print 'Saving figure to:', output_fn
#    pylab.savefig(output_fn, dpi=200)
#    output_fn = output_fn_base + '_2.eps'
#    print 'Saving figure to:', output_fn
#    pylab.savefig(output_fn, dpi=200)



    # fig 4
    plotter.n_fig_x = 1
    plotter.n_fig_y = 2
    plotter.create_fig()  # create an empty figure
    plotter.plot_network_activity('exc', 1)
    plotter.plot_network_activity('inh', 2)
    output_fn = output_fn_base + '_4.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=200)
#    output_fn = output_fn_base + '_4.pdf'
#    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=200)
#    output_fn = output_fn_base + '_4.eps'
#    print 'Saving figure to:', output_fn
#    pylab.savefig(output_fn, dpi=200)



#    plotter.n_fig_x = 1
#    plotter.n_fig_y = 1
#    plotter.create_fig()
#    weights = [plotter.nspikes_binned_normalized[i, :].sum() / plotter.n_bins for i in xrange(plotter.n_cells)]
#    plotter.quiver_plot(weights, fig_cnt=1)
#    output_fn = output_fn_base + '_quiver.png'
#    print 'Saving figure to:', output_fn
#    pylab.savefig(output_fn, dpi=200)



    plotter.save_data()
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

def plot_prediction_1D(params=None, data_fn=None, inh_spikes = None):

    if params== None:
        network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
        params = network_params.params

    if data_fn == None:
        data_fn = params['exc_spiketimes_fn_merged'] + '.ras'

    plotter = P.PlotPrediction(params, data_fn)
    pylab.rcParams['axes.labelsize'] = 14
    pylab.rcParams['axes.titlesize'] = 16
    if plotter.no_spikes:
        return

    plotter.compute_v_estimates()
    plotter.compute_position_estimates()
    plotter.compute_theta_estimates()

    # fig 1:  neuronal level
    output_fn_base = '%s%s_wsigmaX_%.2f_wsigmaV%.2f_delayScale%d_connRadius%.2f_wee%.2f' % (params['prediction_fig_fn_base'], params['connectivity_code'], \
            params['w_sigma_x'], params['w_sigma_v'], params['delay_scale'], params['connectivity_radius'], params['w_tgt_in_per_cell_ee'])
    plotter.create_fig()  # create an empty figure
    pylab.subplots_adjust(left=0.07, bottom=0.07, right=0.97, top=0.93, wspace=0.3, hspace=.2)
    plotter.n_fig_x = 2
    plotter.n_fig_y = 2
    plotter.plot_rasterplot('exc', 1)
    plotter.plot_rasterplot('inh', 2)
    plotter.plot_x_grid_vs_time(3)
    plotter.plot_vx_grid_vs_time(4)
    output_fn = output_fn_base + '_0.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=300)
    pylab.savefig(output_fn, dpi=300)

    # poplation level, short time-scale
    plotter.n_fig_x = 2
    plotter.n_fig_y = 2
    plotter.create_fig()
    pylab.rcParams['legend.fontsize'] = 12
    pylab.subplots_adjust(left=0.07, bottom=0.07, right=0.97, top=0.93, wspace=0.3, hspace=.3)
    plotter.plot_vx_estimates(1)
    plotter.plot_vdiff(2)
    plotter.plot_x_estimates(3)
    plotter.plot_xdiff(4)
    output_fn = output_fn_base + '_1.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=300)
    pylab.savefig(output_fn, dpi=300)

    # fig 4
    plotter.n_fig_x = 1
    plotter.n_fig_y = 2
    plotter.create_fig()  # create an empty figure
    plotter.plot_network_activity('exc', 1)
    plotter.plot_network_activity('inh', 2)
    output_fn = output_fn_base + '_4.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=300)
    plotter.save_data()


if __name__ == '__main__':


    if len(sys.argv) > 1:
        param_fn = sys.argv[1]
        if os.path.isdir(param_fn):
            param_fn += '/Parameters/simulation_parameters.json'

        import json
        f = file(param_fn, 'r')
        print 'Loading parameters from', param_fn
        params = json.load(f)
        if params['n_grid_dimensions'] == 2:
            plot_prediction_2D(params=params)
        else:
            plot_prediction_1D(params=params)

    else:
        print '\nPlotting the default parameters give in simulation_parameters.py\n'
        network_params = simulation_parameters.parameter_storage()
        params = network_params.params

        if params['n_grid_dimensions'] == 2:
            plot_prediction_2D()
        else:
            plot_prediction_1D()

