import os
import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

n_iterations = 80
exc_cell = 85
for iteration in xrange(48, n_iterations):
    matrix_fn = params['folder_name'] + 'TrainingResults_%d/' % iteration + 'wij_matrix_%d.dat' % (iteration)
    fig_fn = params['figures_folder'] + 'conn_profile_%d_%d.png' % (iteration, exc_cell)
    input_fn_base = params['folder_name'] + 'TrainingInput_%d/abstract_input_' % (iteration)
    os.system('python plot_all_pij.py %d' % iteration)
#    os.system('python plot_abstract_activation.py %d' % (iteration))
    os.system('python plot_connectivity_profile_abstract.py %d %s %s %d' % (exc_cell, matrix_fn, fig_fn, iteration))

