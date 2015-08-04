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

from PlottingScripts.PlotPrediction import plot_prediction
from PlottingScripts.PlotCurrents import run_plot_currents
#from PlottingScripts.plot_incoming_currents import plot_incoming_currents
import matplotlib
matplotlib.use('Agg')
from PlottingScripts.PlotAnticipation import plot_anticipation, plot_anticipation_cmap, plot_vmem_aligned

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

    use_command_line_arguments = False

    # set up the parameter storage class ps
    t_0 = time.time()
    ps = simulation_parameters.parameter_storage()
    params = ps.params
    # if training_run is set, you might end up with other wrong parameters (e.g. n_stim) --> assertion
    assert (params['training_run'] == False), 'Wrong flag in simulation parameters. Set training_run = False.'


    # Here, you should set the folder name that has been created during the previous training run, i.e. the folder that main_training_ has created
    # conn_fn_ should be the filenames for the connection matrices on MC-MC basis
    conn_fn_ampa = 'TrainingSim_asymmetricTauij__2x1_0-16_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_v0.40-0.8_fmax500/Connections/conn_matrix_mc.dat'
    conn_fn_nmda = 'TrainingSim_asymmetricTauij__2x1_0-16_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_v0.40-0.8_fmax500/Connections/conn_matrix_mc.dat'

    if use_command_line_arguments:
        taui_ampa = float(sys.argv[1])
        taui_nmda = float(sys.argv[2])
        g_exc_total_in = float(sys.argv[3])
        g_inh_total_in = float(sys.argv[4])
        ampa_nmda_ratio = float(sys.argv[5])
        w_noise_exc = float(sys.argv[6])
        # The filenames where to find the trained weights can also depend on the learning parameters for taui_ampa and taui_nmda
        # Here, you should set the folder name that has been created during the previous training run
        #conn_fn_ampa = 'TrainingSim_Cluster_symmetricTauij__50x2x1_0-400_taui%d_nHC20_nMC4_vtrain1.00-1.0/Connections/conn_matrix_mc.dat' % (params['taui_ampa'])
        #conn_fn_nmda = 'TrainingSim_Cluster_symmetricTauij__50x2x1_0-400_taui%d_nHC20_nMC4_vtrain1.00-1.0/Connections/conn_matrix_mc.dat' % (params['taui_nmda'])

        # depending on the type of parameter sweep that is done, use different parameters
        # e.g.
        #bcpnn_gain = float(sys.argv[3])
        #params['bcpnn_gain'] = bcpnn_gain
        #w_input_exc = float(sys.argv[4])
        #params['w_input_exc'] = w_input_exc

        # update the parameter dictionary to be passed to the NetworkModel with the given values 
        params['taui_ampa'] = taui_ampa
        params['taui_nmda'] = taui_nmda
        params['w_noise_exc'] = w_noise_exc
        params['w_noise_inh'] = -1.* w_noise_exc
        params['ampa_nmda_ratio'] = ampa_nmda_ratio
        params['w_in_ampa_pos_target'] = params['w_in_nmda_pos_target'] * params['ampa_nmda_ratio']
        params['w_in_ampa_neg_target'] = -1. * params['w_in_ampa_pos_target']
        if g_inh_total_in > 0:
            g_inh_total_in *= -1.
        assert (g_inh_total_in <= 0), 'g_inh_total_in must be < 0'
        params['g_exc_total_in'] = g_exc_total_in
        params['g_inh_total_in'] = g_inh_total_in
    else:
        ampa_nmda_ratio = params['ampa_nmda_ratio']

    w_ei = params['w_ei_unspec']
    #w_ie = float(sys.argv[5]) * w_ei
    w_ie = params['w_ie_unspec']
    w_ii = params['w_ii_unspec']

    if not params['debug']:
#        assert (bcpnn_gain > 0), 'BCPNN gain needs to be positive!'
        assert (w_ei > 0), 'Excitatory weights need to be positive!'
        assert (w_ie < 0), 'Inhibitory weights need to be negative!'
        #assert (w_ii < 0), 'Inhibitory weights need to be negative!'
        params['ampa_nmda_ratio'] = ampa_nmda_ratio
        params['w_ie_unspec'] = w_ie
        params['w_ei_unspec'] = w_ei
        params['w_ii_unspec'] = w_ii
        folder_name = 'TestSim_%s_tauiAMPA_%d_NMDA_%d_nExcPerMc%d_Etgt%.2f_Itgt%.2f_ratio%.1f_wei%.1f_wie%.1f_wii%.2f_wNoiseExc%.2f_Inh%.2f' % ( \
                params['sim_id'], params['taui_ampa'], params['taui_nmda'], \
                params['n_exc_per_mc'], params['g_exc_total_in'], params['g_inh_total_in'], params['ampa_nmda_ratio'], \
                params['w_ei_unspec'], params['w_ie_unspec'], params['w_ii_unspec'], params['w_noise_exc'], params['w_noise_inh'])
        folder_name += '/'
        ps.set_filenames(folder_name) 
    else:
        ps.set_filenames() 

    params['conn_fn_ampa'] = conn_fn_ampa
    params['conn_fn_nmda'] = conn_fn_nmda
    ps.create_folders()
    if pc_id == 0:
        ps.write_parameters_to_file()
        utils.remove_files_from_folder(params['spiketimes_folder'])
        utils.remove_files_from_folder(params['connections_folder'])
        utils.remove_files_from_folder(params['volt_folder'])
        if not params['load_input']:
            utils.remove_files_from_folder(params['input_folder'])
    if comm != None:
        comm.barrier()
    load_files = False
    save_input_files = True #not load_files
    NM = NetworkModel(params, iteration=0, comm=comm)
    if not params['debug']:
        NM.set_connection_matrices(conn_fn_ampa, conn_fn_nmda)
    pc_id, n_proc = NM.pc_id, NM.n_proc
    if comm != None:
        comm.barrier()
    NM.setup()
    if comm != None:
        comm.barrier()
    NM.create()
    if comm != None:
        comm.barrier()
    NM.connect()

#        NM.record_v_inh_unspec()

    # run_sim
    NM.run_test_approaching_stim()

    if pc_id == 0:
        ps.write_parameters_to_file(params['params_fn_json'], NM.params) # write_parameters_to_file MUST be called after the simulation, as t_sim is computed and updated
    if comm != None:
        comm.barrier()

    NM.collect_spikes()
    NM.collect_vmem_data()
#    NM.get_weights_static()

    t_end = time.time()
    t_diff = t_end - t_0
    print "Simulating %d cells for %d ms took %.3f seconds or %.2f minutes on proc %d (%d)" % (params['n_cells'], params["t_sim"], t_diff, t_diff / 60., NM.pc_id, NM.n_proc)
    #if pc_id == 0 and not params['Cluster']:
    if params['Cluster']:
        show = False
    else:
        show = True
    
    if params['Cluster']:
        if pc_id == 0:
            plot_anticipation_cmap(params)
        elif pc_id == 1:
            plot_anticipation(params, show) 
        elif pc_id == 2:
            plot_vmem_aligned(params)
        elif pc_id == 3:
            run_plot_currents(NM.params)
        elif pc_id == 4:
            plot_prediction(params=NM.params, stim_range=params['stim_range'])
    else:
        if pc_id == 0:
            plot_anticipation_cmap(params)
            plot_anticipation(params, show) 
            plot_vmem_aligned(params)
            run_plot_currents(NM.params)
            plot_prediction(params=NM.params, stim_range=params['stim_range'])
#        plot_incoming_currents(NM.params)
#        os.system('python PlottingScripts/PlotCurrents.py %s' % (params['folder_name']))
#        display_cmd = 'ristretto $(find %s -name prediction_stim0.png)' % params['folder_name']
#        os.system(display_cmd)

    if comm != None:
        comm.barrier()
