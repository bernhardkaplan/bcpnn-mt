import simulation_parameters
import numpy as np
import utils
import Bcpnn
import os
import sys
import time
import simulation_parameters

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


def prepare_input(tp, params, my_units=None):
    n_units = tp.shape[0]
    dt = params['dt_rate'] # [ms] time step for the non-homogenous Poisson process 

    time = np.arange(0, params['t_stimulus'], dt)

    if (my_units == None):
        my_units = xrange(n_units)
    else:
        my_units = xrange(my_units[0], my_units[1])

    n_cells = len(my_units)
    L_input = np.empty((n_cells, time.shape[0]))
#    offset = 100
    for i_time, time_ in enumerate(time):
        if (i_time % 100 == 0):
            print "t:", time_
#        i_time += offset
#        i_time = min(i_time, max(i_time, len(time)-1))
        L_input[:, i_time] = utils.get_input(tuning_prop[my_units, :], params, time_/params['t_sim'])

    for i_, unit in enumerate(my_units):
        rate_of_t = np.array(L_input[i_, :]) 
        output_fn = params['input_rate_fn_base'] + str(unit) + '.dat'
        print 'output_fn:', output_fn
        np.savetxt(output_fn, rate_of_t)

# C O M P U T E    P_IJ
def compute_my_pijs(conns, output_fn, tau_dict):
    """
    conns = list of connections, i.e. tuples: (src, tgt)
    """
    print 'pc_id computes pijs for %d connections' % (len(conns))
    my_traces = {}
    p_ij_string = '#pre_id\tpost_id\tpij[-1]\tpij_max\tw_ij[-1]\tbias\n'
    for i in xrange(len(conns)):
    #for i in xrange(100):
        print "Pc %d conn: \t%d - %d; \t%d / %d\t%.4f percent complete" % (pc_id, conns[i][0], conns[i][1], i, len(conns), i * 100./len(conns))
        pre_id = conns[i][0]
        post_id = conns[i][1]
        if not my_traces.has_key(pre_id): # compute traces
            fn = params['input_rate_fn_base'] + str(pre_id) + '.dat'
            print 'PID %d loads pre trace %s' % (pc_id, fn)
            pre_trace = np.loadtxt(fn)
            zi, ei, pi = Bcpnn.compute_traces(pre_trace, tau_dict['tau_zi'], tau_dict['tau_ei'], tau_dict['tau_pi'])
            my_traces[pre_id] = (zi, pi)
        else: # load the traces from the dict
            zi, pi = my_traces[pre_id]

        if not my_traces.has_key(post_id): # compute traces
            fn = params['input_rate_fn_base'] + str(post_id) + '.dat'
            print 'PID %d loads post trace %s' % (pc_id, fn)
            post_trace = np.loadtxt(fn)
            zj, ej, pj = Bcpnn.compute_traces(pre_trace, tau_dict['tau_zj'], tau_dict['tau_ej'], tau_dict['tau_pj'])
            my_traces[post_id] = (zj, pj)
        else: # load the traces from the dict
            zj, pj = my_traces[post_id]
        pij, pij_max, w_ij, bias = Bcpnn.compute_pij(zi, zj, pi, pj, tau_dict['tau_eij'], tau_dict['tau_pij'])
        p_ij_string += '%d\t%d\t%.6e\t%.6e\t%.6e\t%.6e\n' % (pre_id, post_id, pij, pij_max, w_ij, bias)

    print 'Writing p_ij output to:', output_fn
    f = file(output_fn, 'w')
    f.write(p_ij_string)
    f.close()
    if comm != None:
        comm.barrier()
    if n_proc > 1 and pc_id == 0:
        tmp_fn = params['bcpnntrace_folder'] + 'all_pij.dat'
        cat_cmd = 'cat %s* > %s' % (params['bcpnntrace_folder'] + 'pij_', tmp_fn)
        print cat_cmd
        os.system(cat_cmd)

def compute_traces(conns, verbose=True):
    p_ij_string = ''
    bcpnn_trace_len = params['t_stimulus'] / params['dt_rate']
    trace_len = params['t_stimulus'] / params['dt_rate']


    for i in xrange(len(conns)):
    #for i in xrange(100):
        print "Pc %d conn: \t%d - %d; \t%d / %d\t%.4f percent complete" % (pc_id, conns[i][0], conns[i][1], i + 1, len(conns), i * 100./len(conns))
        pre_id = conns[i][0]
        post_id = conns[i][1]

        pre_trace = np.zeros(bcpnn_trace_len)
        d = np.loadtxt(params['input_rate_fn_base'] + str(pre_id) + '.dat')
        pre_trace[:trace_len] = d
        post_trace = np.zeros(bcpnn_trace_len)
        d = np.loadtxt(params['input_rate_fn_base'] + str(post_id) + '.dat')
        post_trace[:trace_len] = d
        post_trace[trace_len:] = 0.
        post_trace.resize(bcpnn_trace_len)

        # compute
        wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = Bcpnn.get_spiking_weight_and_bias(pre_trace, post_trace, get_traces=True, \
                tau_dict=tau_dict, f_max=1000.)

        p_ij_string += '%d\t%d\t%.6e\n' % (pre_id, post_id, pij[-1])

        weight_fn = params['weights_fn_base'] + '%d_%d.dat' % (pre_id, post_id)
        if verbose:
            print 'Saving to ', weight_fn
        np.savetxt(weight_fn, wij)
        output_fn = params['bias_fn_base'] + "%d_%d.dat" % (pre_id, post_id)
        if verbose:
            print 'Saving to ', output_fn
        np.savetxt(output_fn, bias)

        output_fn = params['ztrace_fn_base'] + "%d.dat" % pre_id
        if verbose:
            print 'Saving to ', output_fn
        np.savetxt(output_fn, zi)
        output_fn = params['ztrace_fn_base'] + "%d.dat" % post_id
        if verbose:
            print 'Saving to ', output_fn
        np.savetxt(output_fn, zj)

        output_fn = params['etrace_fn_base'] + "%d.dat" % pre_id
        if verbose:
            print 'Saving to ', output_fn
        np.savetxt(output_fn, ei)
        output_fn = params['etrace_fn_base'] + "%d.dat" % post_id
        if verbose:
            print 'Saving to ', output_fn
        np.savetxt(output_fn, ej)
        output_fn = params['etrace_fn_base'] + "%d_%d.dat" % (pre_id, post_id)
        if verbose:
            print 'Saving to ', output_fn
        np.savetxt(output_fn, eij)

        output_fn = params['ptrace_fn_base'] + "%d.dat" % pre_id
        if verbose:
            print 'Saving to ', output_fn
        np.savetxt(output_fn, pi)
        output_fn = params['ptrace_fn_base'] + "%d.dat" % post_id
        if verbose:
            print 'Saving to ', output_fn
        np.savetxt(output_fn, pj)
        output_fn = params['ptrace_fn_base'] + "%d_%d.dat" % (pre_id, post_id)
        if verbose:
            print 'Saving to ', output_fn
        np.savetxt(output_fn, pij)

        output_fn = params['weights_fn_base'] + "%d_%d.dat" % (pre_id, post_id)
        if verbose:
            print 'Saving to ', output_fn
        np.savetxt(output_fn, wij)


times = []
times.append(time.time())

full_system = False
if len(sys.argv) > 1:
    if sys.argv[1] == 'full':
        full_system = True

tau_dict = {'tau_zi' : 400,    'tau_zj' : 10, 
            'tau_ei' : 100,   'tau_ej' : 100, 'tau_eij' : 100,
            'tau_pi' : 100000,  'tau_pj' : 100000, 'tau_pij' : 100000,
            }

PS = simulation_parameters.parameter_storage()
params = PS.params
PS.create_folders()
PS.write_parameters_to_file()

n_cells = params['n_exc']
my_units = utils.distribute_n(n_cells, n_proc, pc_id)

mp = params['motion_params']

# P R E P A R E     T U N I N G    P R O P E R T I E S
#tuning_prop = utils.set_tuning_prop(params, mode='hexgrid', v_max=params['v_max'])
#np.savetxt(params['tuning_prop_means_fn'],tuning_prop)

# load
tuning_prop = np.loadtxt(params['tuning_prop_means_fn'])

# P R E P A R E     I N P U T 
prepare_input(tuning_prop, params, my_units)
#exit(1)
times.append(time.time())


# S E L E C T    C E L L S    T O     C O M P U T E    B C P N N    F O R 
#input_sum = np.zeros(n_cells) # summed input signal
#for i in xrange(n_cells):
#    input_fn = params['input_rate_fn_base'] + str(i) + '.dat'
#    rate = np.loadtxt(input_fn)
#    input_sum[i] = rate.sum()
#sorted_idx = input_sum.argsort()
#n_selected = int(round(.25 * n_cells))
#selected_idx = sorted_idx[-n_selected:]
#print 'chosen cells:', sorted_idx[selected_idx], input_sum[selected_idx]
#print 'mean selected, mean all', input_sum[selected_idx].mean(), input_sum.mean()
#print 'std selected, std all', input_sum[selected_idx].std(), input_sum.std()
#print 'max selected, max all', input_sum[selected_idx].max(), input_sum.max()
#print 'min selected, min all', input_sum[selected_idx].min(), input_sum.min()



if full_system: 
    # distribute the connections
    all_conns = []
    for i in xrange(n_cells):
        for j in xrange(i):
            if i != j:
                all_conns.append((i, j))

    my_conns = utils.distribute_list(all_conns, n_proc, pc_id)
    output_fn = params['bcpnntrace_folder'] + 'pij_%d.dat' % (pc_id)
    compute_my_pijs(my_conns, output_fn, tau_dict)
    times.append(time.time())
    t_comp = times[-1] - times[0]
    print 'Computation time: %d sec = %.1f min' % (t_comp, t_comp / 60.)
else:
    my_conns = [(145, 125)]
#    my_conns = [(145, 177), (65, 61)]
    compute_traces(my_conns)

    # debug 
    pre_trace = np.loadtxt(params['input_rate_fn_base'] + '%d.dat' % my_conns[0][0])
    zi, ei, pi = Bcpnn.compute_traces(pre_trace, tau_dict['tau_zi'], tau_dict['tau_ei'], tau_dict['tau_pi'])
    zj, ej, pj = Bcpnn.compute_traces(pre_trace, tau_dict['tau_zj'], tau_dict['tau_ej'], tau_dict['tau_pj'])
    pij, pij_max, w_ij, bias = Bcpnn.compute_pij(zi, zj, pi, pj, tau_dict['tau_eij'], tau_dict['tau_pij'])

    zi_debug = np.loadtxt(params['ztrace_fn_base'] + "%d.dat" % my_conns[0][0])
    bias_debug = np.loadtxt(params['ztrace_fn_base'] + "%d.dat" % my_conns[0][0])
    assert np.any(zi == zi_debug)


    # P L O T T I N G
    text = ''
    text += 'pre_id %d post_id %d\n' % (my_conns[0][0], my_conns[0][1])
    text += 'tau_zi = %d\n' % tau_dict['tau_zi']
    text += 'tau_zj = %d\n' % tau_dict['tau_zj']
    text += 'tau_e = %d\n' % tau_dict['tau_ei']
    text += 'tau_p = %d\n' % tau_dict['tau_pi']
    text += 'x_stim(t) = (%.1f, %.1f) + (%.1f, %.1f) * t\n'% (mp[0], mp[1], mp[2], mp[3])
    import plot_bcpnn_traces as trace_plotter
    output_fig_fn = params['figures_folder'] + 'bcpnn_trace_%d_%d_%d.png' % (tau_dict['tau_zi'], tau_dict['tau_ei'], tau_dict['tau_pi'])
    times.append(time.time())
    t_comp = times[-1] - times[0]
    print 'Computation time: %d sec = %.1f min' % (t_comp, t_comp / 60.)
    trace_plotter.plot_all(my_conns[0][0], my_conns[0][1], fig=None, text=text, output_fn=output_fig_fn, show=True)

