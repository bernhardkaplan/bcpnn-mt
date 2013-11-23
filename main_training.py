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

def plot_traces(t_axis, pi, pj, pij, wij_nest, output_fn=None, title=''):

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
    ax1.set_title(title)
    
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

    neuron_gid_pairs = [(37, 204), (204, 37), (106, 178), (106, 155), (106, 224), \
            (3, 24), (24, 3), (58, 79), (58, 144), (79, 144), (79, 128), (58, 128), \
            (90, 210), (90, 137), (137, 90), (210, 137), (137, 210)]
    n_conn = len(neuron_gid_pairs)
    on_node = np.zeros(n_conn)
    for i_, (pre_gid, post_gid) in enumerate(neuron_gid_pairs):
        on_node[i_] = NM.check_if_conn_on_node(pre_gid, post_gid)

    t = 0
    t_step = 50.
    t_axis = np.arange(0, params['t_sim'], t_step)
    # set default values for all connections 
    pi_nest = np.ones((n_conn, t_axis.size)) * params['bcpnn_init_val']
    pj_nest = np.ones((n_conn, t_axis.size)) * params['bcpnn_init_val']
    pij_nest = np.ones((n_conn, t_axis.size)) * params['bcpnn_init_val'] ** 2
    wij_nest = np.zeros((n_conn, t_axis.size))
    for j_ in xrange(n_conn):
        wij_nest[j_, :] = np.log(pij_nest[j_, :] / (pi_nest[j_, :] * pj_nest[j_, :]))

    for i_, t in enumerate(t_axis):
        NM.run_sim(t_step)
        # iterate over all connections on the node
        for j_, (pre_gid, post_gid) in enumerate(neuron_gid_pairs):
            if on_node[j_] != False:
                pre_gid_nest = pre_gid + 1
                post_gid_nest = post_gid + 1
                pi, pj, pij, wij = NM.get_p_values([pre_gid_nest, post_gid_nest])
                pi_nest[j_, i_] = pi
                pj_nest[j_, i_] = pj
                pij_nest[j_, i_] = pij
                wij_nest[j_, i_] = wij

    # ---- end sim ---------

    # plot traces
    for j_ in xrange(n_conn):
        if on_node[j_] != False:
            pre_gid, post_gid = neuron_gid_pairs[j_]
            output_fn = params['figures_folder'] + 'p_traces_%d_%d.png' % (pre_gid, post_gid)
            title = 'Traces for GIDs %d - %d' % (pre_gid, post_gid)
            plot_traces(t_axis, pi_nest[j_], pj_nest[j_], pij_nest[j_], wij_nest[j_], output_fn=output_fn, title=title)
            output_fn = params['connections_folder'] + 'p_traces_%d_%d.dat' % (pre_gid, post_gid)
            print 'Saving traces to', output_fn
            np.savetxt(output_fn, np.array((t_axis, pi_nest[j_], pj_nest[j_], pij_nest[j_], wij_nest[j_])).transpose())


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

    t_0 = time.time()
    ps = simulation_parameters.parameter_storage()
    params = ps.params
    if not params['training_run']:
        print 'Wrong flag in simulation parameters. Set training_run = True.'
        exit(1)

#    if pc_id == 0:
#        ok = raw_input('\nNew folder: %s\nContinue to create this folder structure? Parameters therein will be overwritten\n\ty / Y / blank = OK; anything else --> exit\n' % ps.params['folder_name'])
#        if not ((ok == '') or (ok.capitalize() == 'Y')):
#            print 'quit'
#            exit(1)
#    if comm != None:
#        comm.Barrier()

    # always call set_filenames to update the folder name and all depending filenames (if params are modified and folder names change due to that)!
    ps.set_filenames() 
    ps.create_folders()
    ps.write_parameters_to_file()

    load_files = False
    record = False
    save_input_files = not load_files

    NM = NetworkModel(ps, iteration=0)
    pc_id, n_proc = NM.pc_id, NM.n_proc
    if pc_id == 0:
        utils.remove_files_from_folder(params['spiketimes_folder'])
    NM.setup()
    NM.create()
    NM.create_training_input(load_files=load_files, save_output=save_input_files, with_blank=(not params['training_run']))
    NM.connect()

    if record:
        NM.record_v_exc()
        NM.record_v_inh_unspec()

    tracking = True
    if tracking:
        run_tracking(params, NM)
    else:
        NM.run_sim(params['t_sim'])
        NM.get_weights_after_learning_cycle()

    t_end = time.time()
    t_diff = t_end - t_0
    print "Simulating %d cells for %d ms took %.3f seconds or %.2f minutes" % (params['n_cells'], params["t_sim"], t_diff, t_diff / 60.)

