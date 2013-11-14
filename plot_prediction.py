import matplotlib
matplotlib.use('Agg')
import pylab
import PlotPrediction as P
import sys
import simulation_parameters
import os
import utils

#folder = 'Data_inputstrength_swepng/NoColumns_winit_random_wsigmaX2.50e-01_wsigmaV2.50e-01_winput2.00e-03_finput2.00e+03pthresh1.0e-01_ptow1.0e-02/' 
#params_fn = folder + 'simulation_parameters.info'
#data_fn = folder + 'Spikes/exc_spikes_merged_.ras'
#inh_spikes = folder + 'Spikes/inh_spikes_.ras'
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
#data_fn = params['exc_spiketimes_fn_merged'] + '.ras'
#print 'data_fn: ', data_fn
#exit(1)

#plot_prediction()#params, data_fn, inh_spikes, output_fn)

