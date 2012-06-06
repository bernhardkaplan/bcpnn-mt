import matplotlib
matplotlib.use('Agg')
import pylab
import PlotPrediction as P
import sys
import NeuroTools.parameters as ntp

def plot_prediction(params=None):

    if params==None:
        import simulation_parameters
        network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
        params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

    plotter = P.PlotPrediction(params)
    if plotter.no_spikes:
        return
    # fig 1
    plotter.plot()

    plotter.compute_v_estimates()
    plotter.compute_theta_estimates()

    # neuronal level
    plotter.plot_nspikes_binned()               # 1
    plotter.plot_nspikes_binned_normalized()    # 2
    plotter.plot_vx_confidence_binned()         # 3
    plotter.plot_vy_confidence_binned()         # 4

    # poplation level, short time-scale
    plotter.plot_vx_estimates()                 # 5
    plotter.plot_vy_estimates()                 # 6
    plotter.plot_theta_estimates()              # 7

    output_fn = params['prediction_fig_fn_base'] + '0.png'
    output_fn = '%s%s_wsigmaX_%.2f_wsigmaV%.2f_0.png' % (params['prediction_fig_fn_base'], params['initial_connectivity'], \
            params['w_sigma_x'], params['w_sigma_v'])
    pylab.savefig(output_fn)
    # fig 2
    # population level, long time-scale
    plotter.plot_fullrun_estimates()
    plotter.plot_nspike_histogram()


    output_fn = '%s%s_wsigmaX_%.2f_wsigmaV%.2f_1.png' % (params['prediction_fig_fn_base'], params['initial_connectivity'], \
            params['w_sigma_x'], params['w_sigma_v'])
    print "Saving to:", output_fn
    pylab.savefig(output_fn)
    #pylab.show()

