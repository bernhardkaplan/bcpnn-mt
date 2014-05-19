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

def plot_traces(t_axis, pi, pj, pij, wij_nest, output_fn=None):

    import pylab
    wij = np.log(pij / (pi * pj))
    bias = np.log(pj)
    fig = pylab.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    ax1.set_title('Traces retrieved from NEST module')
    plots = []
    p1, = ax1.plot(t_axis, pi)
    p2, = ax1.plot(t_axis, pj)
    p3, = ax1.plot(t_axis, pij)
    plots += [p1, p2, p3]
    labels = ['$p_i$', '$p_j$', '$p_{ij}$']
    ax1.legend(plots, labels, loc='upper left')
    ax1.set_ylabel('p values')


    plots = []
    p1, = ax2.plot(t_axis, wij)
    p2, = ax2.plot(t_axis, wij_nest)
    plots += [p1, p2]
    labels = ['$w=\log(p_{ij} / (p_i \cdot p_j))$', '$w_{NEST}$']
    ax2.set_ylabel('Weight')
    ax2.legend(plots, labels, loc='upper left')


    plots = []
    p1, = ax3.plot(t_axis, bias)
    plots += [p1]
    ax3.set_ylabel('Bias')
    ax3.set_xlabel('Time [ms]')
    ax3.legend(plots, ['Bias'], loc='upper left')

    if output_fn == None:
        output_fn = params['figures_folder'] + 'nest_traces.png'
    print 'Saving to:', output_fn
    pylab.savefig(output_fn, dpi=300)


def run_tracking(params, NM):
    pre_gid = 62
    post_gid = 81
    hc_idx, mc_idx_in_hc, idx_in_mc = NM.get_indices_for_gid(pre_gid)
    pre_neuron = NM.list_of_exc_pop[hc_idx][mc_idx_in_hc][idx_in_mc]
    hc_idx, mc_idx_in_hc, idx_in_mc = NM.get_indices_for_gid(post_gid)
    post_neuron = NM.list_of_exc_pop[hc_idx][mc_idx_in_hc][idx_in_mc]
    on_node = NM.get_p_values([pre_neuron, post_neuron])
    t = 0
    t_step = 10.
    t_axis = np.arange(0, params['t_sim'], t_step)
    pi_nest = np.ones(t_axis.size) * params['bcpnn_init_val']
    pj_nest = np.ones(t_axis.size) * params['bcpnn_init_val']
    pij_nest = np.ones(t_axis.size) * params['bcpnn_init_val'] ** 2
    wij_nest = np.log(pij_nest / (pi_nest * pj_nest))
    for i_, t in enumerate(t_axis):
        NM.run_sim(t_step)
        if on_node != False:
            pi, pj, pij, wij = NM.get_p_values([pre_neuron, post_neuron])
            pi_nest[i_] = pi
            pj_nest[i_] = pj
            pij_nest[i_] = pij
            wij_nest[i_] = wij
    if on_node != False:
        output_fn = params['figures_folder'] + 'p_traces_%d_%d.png' % (pre_gid, post_gid)
        plot_traces(t_axis, pi_nest, pj_nest, pij_nest, wij_nest, output_fn)
        output_fn = params['connections_folder'] + 'p_traces_%d_%d.dat' % (pre_gid, post_gid)
        print 'Saving traces to', output_fn
        np.savetxt(output_fn, np.array((t_axis, pi_nest, pj_nest, pij_nest, wij_nest)).transpose())


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

    assert (len(sys.argv) > 1), 'Missing training folder as command line argument'
    training_folder = os.path.abspath(sys.argv[1]) # contains the EPTH and OB activity of simple patterns
    print 'Training folder:', training_folder
    training_params_fn = os.path.abspath(training_folder) + '/Parameters/simulation_parameters.json'
    training_param_tool = simulation_parameters.parameter_storage(params_fn=training_params_fn)
    training_params = training_param_tool.params

    t_0 = time.time()
    ps = simulation_parameters.parameter_storage()
    params = ps.params
    if params['training_run']:
        print 'Wrong flag in simulation parameters. Set training_run = False.'
        exit(1)

    assert (params['n_cells'] == training_params['n_cells']), 'ERROR: Test and training params are differen wrt n_cells!\n\ttraining %d \t test %d' % (trainin_params['n_cells'], params['n_cells'])
    # always call set_filenames to update the folder name and all depending filenames (if params are modified and folder names change due to that)!
    ps.set_filenames() 
    ps.create_folders()
    ps.write_parameters_to_file()

    load_files = False
    record = False
    save_input_files = True #not load_files
    NM = NetworkModel(ps, iteration=0)
    pc_id, n_proc = NM.pc_id, NM.n_proc
    if pc_id == 0:
        utils.remove_files_from_folder(params['spiketimes_folder'])
    NM.setup()
    NM.training_params = training_params
    NM.create()
#    NM.create_test_input(load_files=load_files, save_output=save_input_files, with_blank=True)
    NM.create_test_input(load_files=load_files, save_output=save_input_files, with_blank=True, training_params=training_params)
    NM.connect()
    NM.connect_recorder_neurons()

    if record:
        NM.record_v_exc()
        NM.record_v_inh_unspec()

    tracking = False
    if tracking:
        run_tracking(params, NM)
    else:
        NM.run_sim(params['t_sim'])

    t_end = time.time()
    t_diff = t_end - t_0
    print "Simulating %d cells for %d ms took %.3f seconds or %.2f minutes" % (params['n_cells'], params["t_sim"], t_diff, t_diff / 60.)
    if pc_id == 0:
        os.system('python PlottingScripts/PlotPrediction.py')

