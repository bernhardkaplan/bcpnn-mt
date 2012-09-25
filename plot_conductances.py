#import matplotlib
#matplotlib.use('Agg')
import pylab
import PlotConductances as P
import sys
import NeuroTools.parameters as ntp
import simulation_parameters
import time

def plot_conductances(params=None, data_fn=None, inh_spikes = None):

    t_start = time.time()
    if params== None:
        network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
#        P = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
        params = network_params.params

    sim_cnt = 0
    if data_fn == None:
        data_fn = params['exc_spiketimes_fn_merged'] + '%d.ras' % (sim_cnt)

#    if inh_spikes == None:
#        inh_spikes = params['inh_spiketimes_fn_merged'] + '%d.ras' % (sim_cnt)

    plotter = P.PlotConductances(params, data_fn)

    if plotter.no_spikes:
        return
    output_fn_base = '%s%s_wsigmaX_%.2f_wsigmaV%.2f_wthresh%.1e' % (params['grouped_actitivty_fig_fn_base'], params['initial_connectivity'], \
            params['w_sigma_x'], params['w_sigma_v'], params['w_thresh_connection'])

    # fig 1
    # neuronal level
#    plotter.create_fig()  # create an empty figure
#    plotter.plot_rasterplot('exc', 1)               # 1 
#    plotter.plot_rasterplot('inh', 2)               # 2 
#    plotter.plot_group_spikes_vs_time(3)            # 3
#    output_fn = output_fn_base + '_0.png'
#    print 'Saving figure to:', output_fn
#    pylab.savefig(output_fn)

#    output_fn = '%sgoodcell_connections_%s_wsigmaX_%.2f_wsigmaV%.2f_wthresh%.1e.png' % (params['figures_folder'], params['initial_connectivity'], \
#            params['w_sigma_x'], params['w_sigma_v'], params['w_thresh_connection'])
#    plotter.create_fig()  # create an empty figure
#    plotter.plot_good_cell_connections(1) # subplot 1 + 2
#    print 'Saving figure to:', output_fn
#    pylab.savefig(output_fn)

    # fig 2
    plotter.create_fig()  # create an empty figure
    plotter.plot_input_cond(1) # subplot 1 + 2
    plotter.plot_conductances()
    output_fn = output_fn_base + '_1.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn)

    t_stop = time.time()
    t_run = t_stop - t_start
    print "PlotConductance duration: %d sec or %.1f min for %d cells (%d exc, %d inh)" % (t_run, (t_run)/60., \
            params['n_cells'], params['n_exc'], params['n_inh'])
    # fig 3
#    pylab.show()

if __name__ == '__main__':
    plot_conductances(params=None)

