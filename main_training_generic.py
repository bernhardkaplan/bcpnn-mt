import sys
import os
import json
import simulation_parameters
import nest
import numpy as np
import time
import os
import utils
from copy import deepcopy
from NetworkModelPyNest import NetworkModel
import CreateInput

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



if __name__ == '__main__':


    t_0 = time.time()
    record = False

    t0 = time.time()
    old_params = None
    GP = simulation_parameters.parameter_storage()

    params = GP.params
    params['training_run'] = True
    #params['n_exc_per_mc'] = 4
    #params['training_run'] = True

    if len(sys.argv) > 1:
        params['taui_bcpnn'] = float(sys.argv[1])
        params['bcpnn_params']['tau_i'] = params['taui_bcpnn']
        folder_name = 'TrainingSim_%s_%dx%dx%d_%d-%d_taui%d_nHC%d_nMC%d_blurXV_%.2f_%.2f_pi%.1e/' % ( \
                params['sim_id'], params['n_training_cycles'], params['n_training_v'], params['n_training_x'], \
                params['stim_range'][0], params['stim_range'][1], \
                params['bcpnn_params']['tau_i'], \
                params['n_hc'], params['n_mc_per_hc'], params['blur_X'], params['blur_V'], \
                params['bcpnn_init_val'])
        GP.set_filenames(folder_name=folder_name)

#    trained_stimuli = []
#    info_text = "\nThere are different use cases:\n \
#    \tpython script_name [training_stimuli_fn] [training_stim_idx] \
#    \tpython script_name [folder_containing_connectivity] [training_stimuli_fn] [training_stim_idx] \
#    "
#    assert len(sys.argv) > 2, 'Missing training_stim_information and training_stim_idx!' + info_text
#    if len(sys.argv) == 3:
#        training_stimuli_fn = sys.argv[1]
#        training_stimuli = np.loadtxt(training_stimuli_fn)
#        continue_training_idx = int(sys.argv[2])
#        params = GP.params
#        continue_training_idx = 0
#    elif len(sys.argv) == 4: #load already trained stimuli
#        old_params_json = utils.load_params(os.path.abspath(sys.argv[1]))
#        old_params = utils.convert_to_NEST_conform_dict(old_params_json)
#        params = GP.params
#        trained_stimuli = old_params['trained_stimuli']
#        training_stimuli_fn = sys.argv[2]
#        continue_training_idx = int(sys.argv[3])
#    else:
#        print 'Wrong number of sys.argv!', info_txt
#        exit(1)
#    training_stimuli = np.loadtxt(training_stimuli_fn)
#    n_max = continue_training_idx + params['n_training_cycles'] * params['n_training_stim_per_cycle']
#    assert (training_stimuli[:, 0].size >= n_max), 'The expected number of training iterations (= %d) is too high for the given training_stimuli from file %s (contains %d training stim)' % \
#            (n_max, training_stimuli_fn, training_stimuli[:, 0].size)

    continue_training_idx = 0
    if params['n_stim'] == 1:
        params['stim_range'] = [continue_training_idx]
    else:
        params['stim_range'] = [continue_training_idx, continue_training_idx + params['n_stim']] 

    if pc_id == 0:
        GP.write_parameters_to_file(params['params_fn_json'], params) # write_parameters_to_file MUST be called before every simulation
        utils.remove_files_from_folder(params['spiketimes_folder'])
        utils.remove_files_from_folder(params['connections_folder'])
        utils.remove_files_from_folder(params['volt_folder'])
        if not params['load_input']:
            utils.remove_files_from_folder(params['input_folder'])

    if comm != None:
        comm.Barrier()
    t0 = time.time()

    NM = NetworkModel(params, iteration=0, comm=comm)
    NM.setup() # create stimuli
    #if params['debug']:
        #training_stimuli = np.array([[.1, .5, 1.0, .0], \
                            #[.9, .5, -.1, .0]])
        #NM.setup(training_stimuli=training_stimuli)
    NM.create()
    NM.connect()
    if old_params != None:
        NM.set_weights(old_params)
    if comm != None:
        comm.Barrier()
    if record:
        NM.record_v_exc()
        NM.record_v_inh_unspec()
    GP.write_parameters_to_file(params['params_fn_json'], NM.params) # write_parameters_to_file MUST be called before every simulation
    if comm != None:
        comm.Barrier()
    NM.run_sim()
    if comm != None:
        comm.Barrier()
    NM.trigger_spikes()
    if not params['debug']:
        t_start_get_weights = time.time()
        NM.get_weights_after_learning_cycle()
        t_start_get_weights_static = time.time()
        if params['with_inhibitory_neurons']:
            NM.get_weights_static()
        t_stop_get_weights = time.time()
        print "Getting weights took %d seconds for the bcpnn weights and %d seconds for the static on %d nodes" % (t_start_get_weights_static - t_start_get_weights, t_stop_get_weights - t_start_get_weights_static, n_proc)
    NM.merge_local_gid_files()
    t_end = time.time()
    t_diff = t_end - t_0
    if NM.pc_id == 0:
        print 'Removing empty files ...'
        utils.remove_empty_files(params['spiketimes_folder'])
    else:
        print 'Waiting for remove_empty_files to end ... '

    print "Simulating %d cells for %d ms took %.3f seconds or %.2f minutes" % (params['n_cells'], params["t_sim"], t_diff, t_diff / 60.)

