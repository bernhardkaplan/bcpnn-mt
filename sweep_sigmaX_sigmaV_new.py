import numpy as np
import os
import time
import simulation_parameters
import CreateConnections as CC

sigma_x_start = 0.05
sigma_x_step = 0.05
sigma_x_stop = sigma_x_start + 10 * sigma_x_step
sigma_x_range = np.arange(sigma_x_start, sigma_x_stop, sigma_x_step)

sigma_v_start = 0.05
sigma_v_step = 0.05
sigma_v_stop = sigma_v_start + 10 * sigma_v_step
sigma_v_range = np.arange(sigma_v_start, sigma_v_stop, sigma_v_step)

try:
    from mpi4py import MPI
    USE_MPI = True
    comm = MPI.COMM_WORLD
    pc_id, n_proc = comm.rank, comm.size

except:
    USE_MPI = False
    comm = None
    pc_id, n_proc = 0, 1

t_start = time.time()
print "USE_MPI:", USE_MPI

new_params = {  'initial_connectivity' : 'precomputed', 'w_sigma_x' : sigma_x_start, 'w_sigma_v' : sigma_v_start}
SP = simulation_parameters.parameter_storage()
SP.update_values(new_params)
sweep_independent_folders = []
sweep_independent_folders.append(str(SP.params['input_folder']))
sweep_independent_folders.append(str(SP.params['bias_folder']))
SP.create_folders()
SP.write_parameters_to_file()
param_file = SP.params['params_fn']

# create parameter independent input
if USE_MPI:
    create_input = 'mpirun -np %d python prepare_input.py %s' % (n_proc, param_file)
else:
    create_input = 'python prepare_input.py %s' % (param_file)

os.system(create_input)

# create folders, copy input_folder to all folders
for sigma_v in sigma_v_range:
    for sigma_x in sigma_x_range:
        # for precomputed connectivity
        new_params = {'initial_connectivity' : 'precomputed', 'w_sigma_x' : sigma_x, 'w_sigma_v' : sigma_v}
        SP.update_values(new_params)
        main_folder = SP.params['folder_name']
        SP.create_folders()
        for src_folder in sweep_independent_folders:
            copy = 'cp -r %s %s' % (src_folder, main_folder) # overwrite empty folders
            os.system(copy)
        SP.write_parameters_to_file()

        # for random connectivity
        new_params = { 'initial_connectivity' : 'random'}
        SP.update_values(new_params)
        main_folder = SP.params['folder_name']
        SP.create_folders()
        for src_folder in sweep_independent_folders:
            copy = 'cp -r %s %s' % (src_folder, main_folder)
            os.system(copy)
        SP.write_parameters_to_file()

# now all folders have the crucial data to run the simulations
i_ = 0
for sigma_v in sigma_v_range:
    for sigma_x in sigma_x_range:
        new_params = {'initial_connectivity' : 'precomputed', 'w_sigma_x' : sigma_x, 'w_sigma_v' : sigma_v}
        SP.update_values(new_params)
        print "Precompute connections for sigma_v", i_, sigma_v, ' sigma_x ', sigma_x
        tuning_prop = np.loadtxt(SP.params['tuning_prop_means_fn'])
        CC.compute_weights_from_tuning_prop(tuning_prop, SP.params)
        print "Running precomputed with sigma_v", i_, sigma_v, ' sigma_x ', sigma_x
        param_file = SP.params['params_fn']
        if USE_MPI:
            prepare = 'mpirun -np %d python prepare_tuning_prop.py %s' % (n_proc, param_file)
            run = 'mpirun -np %d python NetworkSimModuleNoColumns.py %s' % (n_proc, param_file)
        else:
            prepare = 'python prepare_tuning_prop.py %s' % (n_proc, param_file)
            run = 'python NetworkSimModuleNoColumns.py %s' % (param_file)
        os.system(prepare)
        os.system(run)

        if comm != None:
            comm.barrier()

        precomputed_weights = SP.params['conn_list_ee_fn_base'] + '0.dat'
        new_params = { 'initial_connectivity' : 'random'}
        SP.update_values(new_params)
        random_weights = SP.params['random_weight_list_fn'] + '0.dat'
        print "Randomize connections for sigma_v", i_, sigma_v, ' sigma_x ', sigma_x
        CC.compute_random_weight_list(precomputed_weights, random_weights, SP.params)
        print "Running with random connectivity sigma_v", i_, sigma_v, ' sigma_x ', sigma_x
        param_file = SP.params['params_fn']
        if USE_MPI:
            run = 'mpirun -np %d python NetworkSimModuleNoColumns.py %s' % (n_proc, param_file)
        else:
            run = 'python NetworkSimModuleNoColumns.py %s' % (param_file)
        os.system(run)

        if comm != None:
            comm.barrier()
        i_ += 1


# run the analysis scripts
i_ = 0
for sigma_v in sigma_v_range:
    for sigma_x in sigma_x_range:
        new_params = {'initial_connectivity' : 'precomputed', 'w_sigma_x' : sigma_x, 'w_sigma_v' : sigma_v}
        SP.update_values(new_params)
        param_file = SP.params['params_fn']
        if USE_MPI:
            plot = 'mpirun -np %d python plot_prediction.py %s' % (n_proc, param_file)
        else:
            plot = 'python plot_prediction.py %s' % (param_file)
        os.system(plot)

        if comm != None:
            comm.barrier()
        new_params = { 'initial_connectivity' : 'random'}
        SP.update_values(new_params)
        param_file = SP.params['params_fn']
        if USE_MPI:
            create_input = 'mpirun -np %d python plot_prediction.py %s' % (n_proc, param_file)
        else:
            create_input = 'python plot_prediction.py %s' % (param_file)
        os.system(plot)
t_stop = time.time()
t_run = t_stop - t_start
print "Full sweep duration: %d sec or %.1f min for %d cells (%d exc, %d inh)" % (t_run, (t_run)/60., \
        SP.params['n_cells'], SP.params['n_exc'], SP.params['n_inh'])
