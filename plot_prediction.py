import pylab
import PlotPrediction as P
import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

plotter = P.PlotPrediction()
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
print "Saving to:", output_fn
pylab.savefig(output_fn)
# fig 2
# population level, long time-scale
plotter.plot_fullrun_estimates()
plotter.plot_nspike_histogram()


output_fn = params['prediction_fig_fn_base'] + '1.png'
print "Saving to:", output_fn
pylab.savefig(output_fn)
#pylab.show()
