import os
import time
import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

n_iterations = 16
n_iterations = 8 * 5 * 2
exc_cells = [85, 161, 339]
for iteration in xrange(n_iterations):
    matrix_fn = params['folder_name'] + 'TrainingResults_%d/' % iteration + 'wij_matrix_%d.dat' % (iteration)
    input_fn_base = params['folder_name'] + 'TrainingInput_%d/abstract_input_' % (iteration)
    os.system('python plot_all_pij.py %d' % iteration)
#    os.system('python plot_abstract_activation.py %d' % (iteration))
#    os.system('python plot_ann_output_activity.py %d' % (iteration))
#    for exc_cell in exc_cells:
#        fig_fn = params['figures_folder'] + 'conn_profile_%d_%d.png' % (iteration, exc_cell)
#        os.system('python plot_connectivity_profile_abstract.py %d %s %s %d' % (exc_cell, matrix_fn, fig_fn, iteration))

