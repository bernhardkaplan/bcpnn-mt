#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script needs to be run on a single core for a set of simulations individually.
It will create the folder structure and print the parameter file,
with which the simulation script NetworkSimModule is to be called.
"""
import os
import simulation_parameters

def clean_up_results_directory(params):
    filenames = [params['exc_nspikes_fn_merged'], \
                params['exc_spiketimes_fn_merged'], \
                params['inh_spiketimes_fn_merged'], \
                params['inh_nspikes_fn_merged'], \
                params['exc_spiketimes_fn_base'], \
                params['inh_spiketimes_fn_base'], \
                params['merged_conn_list_ee'], \
                params['merged_conn_list_ei'], \
                params['merged_conn_list_ie'], \
                params['merged_conn_list_ii']]
    for fn in filenames:
        cmd = 'rm %s*' % (fn)
        print 'Removing %s' % (cmd)
        os.system(cmd)

def prepare_simulation(folder_name, params, cleanup=False):
    if cleanup: clean_up_results_directory(params) # optional
    ps.set_filenames(folder_name)
    ps.create_folders()
    ps.write_parameters_to_file()
    print 'Ready for simulation:\n\t%s' % (ps.params['params_fn_json'])


def run_simulation(folder_name, params, USE_MPI):
    # specify your run command (mpirun -np X, python, ...)
    parameter_filename = params['params_fn_json']
    if USE_MPI:
        run_command = 'mpirun -np 8 python NetworkSimModule.py %s' % parameter_filename
    else:
        run_command = 'python NetworkSimModule.py %s' % parameter_filename
    print 'Running:\n\t%s' % (run_command)
    os.system(run_command)

if __name__ == '__main__':

    try:
        from mpi4py import MPI
        USE_MPI = True
        comm = MPI.COMM_WORLD
        pc_id, n_proc = comm.rank, comm.size
        print "USE_MPI:", USE_MPI, 'pc_id, n_proc:', pc_id, n_proc
    except:
        USE_MPI = False
        pc_id, n_proc, comm = 0, 1, None
        print "MPI not used"

    # define the parameter range you'd like to sweep
    import sys
    param_name = sys.argv[1] #'w_sigma_x' # must
    import numpy as np
    param_range = np.logspace(-1, 1, 5) # [0.01, 0.1, 0.2, 0.3, 0.4,  0.5, 1.0, 100.]

    ps = simulation_parameters.parameter_storage()
    main_folder = 'ParamSweep'
    if not os.path.exists(main_folder):
        os.system('mkdir %s' % main_folder)
    for i_, p in enumerate(param_range):
        # choose how you want to name your results folder
        folder_name = "%s/Data_for_%s_%.2f" % (main_folder, param_name, p)
        if folder_name[-1] != '/':
            folder_name += '/'
        params = ps.params
        params[param_name] = ps.params[param_name] * p
        prepare_simulation(folder_name, params)
        run_simulation(folder_name, params, USE_MPI)

