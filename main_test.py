"""
Simple network with a Poisson spike source projecting to populations of of IF_cond_exp neurons

on the cluster:
    frioul_batch -M "[['w_tgt_in_per_cell_ee', 'w_tgt_in_per_cell_ee', 'w_tgt_in_per_cell_ee'],[0.4, 0.8, 1.2]]" 'python NetworkSimModuleNoColumns.py'


"""
import time
t0 = time.time()
import numpy as np
import numpy.random as nprnd
import sys
#import NeuroTools.parameters as ntp
import os
import utils
import nest
import json
import simulation_parameters
from NetworkModelPyNest import NetworkModel



if __name__ == '__main__':

#    try: 
#        from mpi4py import MPI
#        USE_MPI = True
#        comm = MPI.COMM_WORLD
#        pc_id, n_proc = comm.rank, comm.size
#        print "USE_MPI:", USE_MPI, 'pc_id, n_proc:', pc_id, n_proc
#    except:
#        USE_MPI = False
#        pc_id, n_proc, comm = 0, 1, None
#        print "MPI not used"

    assert (len(sys.argv) > 2), 'Missing connection matrices folders as command line arguments'
    
    conn_fn_ampa = sys.argv[1]
    conn_fn_nmda = sys.argv[2]
    # conn_fn_ should be the filenames for the connection matrices on MC-MC basis
    w_ee = float(sys.argv[3])
    
    t_0 = time.time()
    ps = simulation_parameters.parameter_storage()
    params = ps.params
    assert (params['training_run'] == False), 'Wrong flag in simulation parameters. Set training_run = False.'
#    assert (params['n_mc'] == training_params['n_mc']), 'ERROR: Test and training params are differen wrt n_mc !\n\ttraining %d \t test %d' % (training_params['n_mc'], params['n_mc'])
    #assert (params['n_cells'] == training_params['n_cells']), 'ERROR: Test and training params are differen wrt n_cells!\n\ttraining %d \t test %d' % (training_params['n_cells'], params['n_cells'])
    # always call set_filenames to update the folder name and all depending filenames (if params are modified and folder names change due to that)!

    params['w_ee_global_max'] = w_ee
    folder_name = 'TestSim_%s_%d_nExcPerMc%d_wee%.2f_wei%.2f' % ( \
            params['sim_id'], params['n_test_stim'], 
            params['n_exc_per_mc'], params['w_ee_global_max'], \
            params['w_ei_unspec'])

    folder_name += '/'
    ps.set_filenames(folder_name) 
#    ps.set_filenames() 
    ps.create_folders()
    ps.write_parameters_to_file()

    load_files = False
    record = False
    save_input_files = True #not load_files
    NM = NetworkModel(params, iteration=0)
    NM.set_connection_matrices(conn_fn_ampa, conn_fn_nmda)
    pc_id, n_proc = NM.pc_id, NM.n_proc
    if pc_id == 0:
        utils.remove_files_from_folder(params['spiketimes_folder'])
    NM.setup()# training_params=training_params)
    NM.create()
    NM.connect()
#    NM.connect_recorder_neurons()

    if record:
        NM.record_v_exc()
        NM.record_v_inh_unspec()

    NM.run_sim()

    t_end = time.time()
    t_diff = t_end - t_0
    print "Simulating %d cells for %d ms took %.3f seconds or %.2f minutes" % (params['n_cells'], params["t_sim"], t_diff, t_diff / 60.)
    if pc_id == 0:
        os.system('python PlottingScripts/PlotPrediction.py')

