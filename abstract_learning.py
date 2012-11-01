import simulation_parameters
import numpy as np
import utils
import Bcpnn
import os
import sys
import time

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
    L_input = np.zeros((n_cells, time.shape[0]))
#    offset = 100
    for i_time, time_ in enumerate(time):
        if (i_time % 100 == 0):
            print "t:", time_
#        i_time += offset
#        i_time = min(i_time, max(i_time, len(time)-1))
        L_input[:, i_time] = utils.get_input(tuning_prop[my_units, :], params, time_/params['t_sim'])

    for i_, unit in enumerate(my_units):
        output_fn = params['input_rate_fn_base'] + str(unit) + '.dat'
        print 'output_fn:', output_fn
        np.savetxt(output_fn, L_input[i_, :])


def normalize_input(params):
    if pc_id == 0:
        print 'normalize_input'
        dt = params['dt_rate'] # [ms] time step for the non-homogenous Poisson process 
        L_input = np.zeros((params['n_exc'], params['t_stimulus']/dt))

        v_max = params['v_max']
        if params['log_scale']==1:
            v_rho = np.linspace(v_max/params['N_V'], v_max, num=params['N_V'], endpoint=True)
        else:
            v_rho = np.logspace(np.log(v_max/params['N_V'])/np.log(params['log_scale']),
                            np.log(v_max)/np.log(params['log_scale']), num=params['N_V'],
                            endpoint=True, base=params['log_scale'])
        v_theta = np.linspace(0, 2*np.pi, params['N_theta'], endpoint=False)
        index = 0
        for i_RF in xrange(params['N_RF_X']*params['N_RF_Y']):
            index_start = index
            for i_v_rho, rho in enumerate(v_rho):
                for i_theta, theta in enumerate(v_theta):
                    fn = params['input_rate_fn_base'] + str(index) + '.dat'
                    L_input[index, :] = np.loadtxt(fn)
                    print 'debug', fn
                    index += 1
            index_stop = index
            print 'before', i_RF, L_input[index_start:index_stop, :].sum()
            if (L_input[index_start:index_stop, :].sum() > 1):
                L_input[index_start:index_stop, :] /= L_input[index_start:index_stop, :].sum()
            print 'after', i_RF, L_input[index_start:index_stop, :].sum()

        for i in xrange(params['n_exc']):
            output_fn = params['input_rate_fn_base'] + str(i) + '.dat'
            print 'output_fn:', output_fn
            np.savetxt(output_fn, L_input[i, :])
    if comm != None:
        comm.barrier()



def compute_pre_post_traces(conns, verbose=True):
    bcpnn_trace_len = params['t_stimulus'] / params['dt_rate']
    trace_len = params['t_stimulus'] / params['dt_rate']


    for i in xrange(len(conns)):
    #for i in xrange(100):
        print "Pc %d conn: \t%d - %d; \t%d / %d\t%.4f percent complete" % (pc_id, conns[i][0], conns[i][1], i + 1, len(conns), i * 100./len(conns))
        pre_id = conns[i][0]
        post_id = conns[i][1]

        pre_trace = np.loadtxt(params['input_rate_fn_base'] + str(pre_id) + '.dat')
        post_trace = np.loadtxt(params['input_rate_fn_base'] + str(post_id) + '.dat')

#        pre_trace = np.zeros(bcpnn_trace_len)
#        d = np.loadtxt(params['input_rate_fn_base'] + str(pre_id) + '.dat')
#        pre_trace[:trace_len] = d
#        post_trace = np.zeros(bcpnn_trace_len)
#        d = np.loadtxt(params['input_rate_fn_base'] + str(post_id) + '.dat')
#        post_trace[:trace_len] = d
#        post_trace[trace_len:] = 0.
#        post_trace.resize(bcpnn_trace_len)

        # compute
        wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = Bcpnn.get_spiking_weight_and_bias(pre_trace, post_trace, get_traces=True, \
                tau_dict=tau_dict, f_max=1000.)

        weight_fn = params['weights_fn_base'] + '%d_%d.dat' % (pre_id, post_id)
        if verbose:
            print 'Saving to ', weight_fn
        np.savetxt(weight_fn, wij)
        output_fn = params['bias_fn_base'] + "%d.dat" % (post_id)
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



# C O M P U T E    P_IJ
def compute_my_pijs(conns, output_fn, tau_dict, input_fn_base):
    """
    conns = list of connections, i.e. tuples: (src, tgt)
    """

    dt = 1
    print 'pc_id computes pijs for %d connections' % (len(conns))
    my_traces_pre = {}
    my_traces_post = {}
    p_ij_string = '#pre_id\tpost_id\tpij[-1]\tw_ij[-1]\tbias\n'
    for i in xrange(len(conns)):
        if (i % 500) == 0:
            print "Pc %d conn: \t%d - %d; \t%d / %d\t%.4f percent complete" % (pc_id, conns[i][0], conns[i][1], i, len(conns), i * 100./len(conns))
        pre_id = conns[i][0]
        post_id = conns[i][1]
        if my_traces_pre.has_key(pre_id):
            (zi, pi) = my_traces_pre[pre_id]
        else:
            pre_trace = np.loadtxt(input_fn_base + str(pre_id) + '.dat')
            zi, ei, pi = Bcpnn.compute_traces(pre_trace, tau_dict['tau_zi'], tau_dict['tau_ei'], tau_dict['tau_pi'], eps=dt/tau_dict['tau_pi'])
            my_traces_pre[pre_id] = (zi, pi)

        if my_traces_post.has_key(post_id):
            (zj, pj) = my_traces_post[post_id]
        else: 
            post_trace = np.loadtxt(input_fn_base  + str(post_id) + '.dat')
            zj, ej, pj = Bcpnn.compute_traces(post_trace, tau_dict['tau_zj'], tau_dict['tau_ej'], tau_dict['tau_pj'], eps=dt/tau_dict['tau_pj']) # actually should be eps=dt/tau_dict['tau_pi']
            my_traces_post[post_id] = (zj, pj)

        pij, w_ij, bias = Bcpnn.compute_pij(zi, zj, pi, pj, tau_dict['tau_eij'], tau_dict['tau_pij'])
        p_ij_string += '%d\t%d\t%.8e\t%.8e\t%.8e\n' % (pre_id, post_id, pij, w_ij, bias)

    if comm != None:
        comm.barrier()

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

    return my_traces_pre, my_traces_post
    


times = []
times.append(time.time())

full_system = False
if len(sys.argv) > 1:
    if sys.argv[1] == 'full':
        full_system = True

tau_dict = {'tau_zi' : 50.,    'tau_zj' : 5., 
            'tau_ei' : 50.,   'tau_ej' : 50., 'tau_eij' : 50.,
            'tau_pi' : 500.,  'tau_pj' : 500., 'tau_pij' : 500.,
            }

PS = simulation_parameters.parameter_storage()
params = PS.params
PS.create_folders()
PS.write_parameters_to_file()

n_cells = params['n_exc']
my_units = utils.distribute_n(n_cells, n_proc, pc_id)

mp = params['motion_params']

# P R E P A R E     T U N I N G    P R O P E R T I E S
tuning_prop = utils.set_tuning_prop(params, mode='hexgrid', v_max=params['v_max'])
np.savetxt(params['tuning_prop_means_fn'],tuning_prop)
#exit(1)

# load
#tuning_prop = np.loadtxt(params['tuning_prop_means_fn'])

# P R E P A R E     I N P U T 
#prepare_input(tuning_prop, params, my_units)
#if comm != None:
#    comm.barrier()
#normalize_input(params)
#times.append(time.time())

#exit(1)


if full_system: 
    # distribute the connections
    all_conns = []
    for i in xrange(n_cells):
        for j in xrange(n_cells):
            if i != j:
                all_conns.append((i, j))

    my_conns = utils.distribute_list(all_conns, n_proc, pc_id)
    output_fn = params['bcpnntrace_folder'] + 'pij_%d.dat' % (pc_id)
    input_fn_base = params['input_rate_fn_base']
    compute_my_pijs(my_conns, output_fn, tau_dict, input_fn_base)
    times.append(time.time())
    t_comp = times[-1] - times[0]
    print 'Computation time: %d sec = %.1f min' % (t_comp, t_comp / 60.)

else:
    my_conns = [(25, 81)]
#    my_conns = [(145, 300)]
    compute_pre_post_traces(my_conns)

    # traces computed above by Bcpnn.get_spiking_weight_and_bias
    zi_debug = np.loadtxt(params['ztrace_fn_base'] + "%d.dat" % my_conns[0][0])
    zj_debug = np.loadtxt(params['ztrace_fn_base'] + "%d.dat" % my_conns[0][1])
    ei_debug = np.loadtxt(params['etrace_fn_base'] + "%d.dat" % my_conns[0][0])
    ej_debug = np.loadtxt(params['etrace_fn_base'] + "%d.dat" % my_conns[0][1])
    pi_debug = np.loadtxt(params['ptrace_fn_base'] + "%d.dat" % my_conns[0][0])
    pj_debug = np.loadtxt(params['ptrace_fn_base'] + "%d.dat" % my_conns[0][1])
    bias_j_debug = np.loadtxt(params['bias_fn_base'] + "%d.dat" % my_conns[0][1])

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

    # seperate computation of traces and weights
    pre_trace = np.loadtxt(params['input_rate_fn_base'] + '%d.dat' % my_conns[0][0])
    post_trace = np.loadtxt(params['input_rate_fn_base'] + '%d.dat' % my_conns[0][1])
    dt = 1.
    zi = np.zeros(pre_trace.size)
    zi, ei, pi = Bcpnn.compute_traces(pre_trace, tau_dict['tau_zi'], tau_dict['tau_ei'], tau_dict['tau_pi'], eps=dt/tau_dict['tau_pi'])
    pre_id = my_conns[0][0]
    post_id = my_conns[0][1]
    # print the seperately computed traces 
    np.savetxt('debug_zi_trace_%d.dat' % pre_id, zi)
    np.savetxt('debug_ei_trace_%d.dat' % pre_id, ei)
    np.savetxt('debug_pi_trace_%d.dat' % pre_id, pi)

    zj, ej, pj = Bcpnn.compute_traces(post_trace, tau_dict['tau_zj'], tau_dict['tau_ej'], tau_dict['tau_pj'], eps=dt/tau_dict['tau_pj'])
    np.savetxt('debug_zj_trace_%d.dat' % post_id, zj)
    np.savetxt('debug_ej_trace_%d.dat' % post_id, ej)
    np.savetxt('debug_pj_trace_%d.dat' % post_id, pj)

    pij, w_ij, bias = Bcpnn.compute_pij(zi, zj, pi, pj, tau_dict['tau_eij'], tau_dict['tau_pij'])
    print 'debug w_ij', pre_id, post_id, w_ij
    w_ij_trace, bias, pij, eij = Bcpnn.compute_pij(zi, zj, pi, pj, tau_dict['tau_eij'], tau_dict['tau_pij'], get_traces=True)
    print 'debug w_ij_trace[-1]', w_ij_trace[-1]

    print 'debug compare:'
    print 'zi zi_debug', zi[-1], zi_debug[-1], zi[-1] == zi_debug[-1], np.all(zi == zi_debug)
    print 'zj zj_debug', zj[-1], zj_debug[-1], zj[-1] == zj_debug[-1], np.all(zj == zj_debug)
    print 'ei ei_debug', ei[-1], ei_debug[-1], ei[-1] == ei_debug[-1], np.all(ei == ei_debug)
    print 'ej ej_debug', ej[-1], ej_debug[-1], ej[-1] == ej_debug[-1], np.all(ej == ej_debug)
    print 'pi pi_debug', pi[-1], pi_debug[-1], pi[-1] == pi_debug[-1], np.all(pi == pi_debug)
    print 'pj pj_debug', pj[-1], pj_debug[-1], pj[-1] == pj_debug[-1], np.all(pj == pj_debug)
    print 'bias, bias_j_debug[-1]', bias[-1], bias_j_debug[-1]
    assert np.all(bias[-1] == bias_j_debug[-1])

    trace_plotter.plot_all(my_conns[0][0], my_conns[0][1], fig=None, text=text, output_fn=output_fig_fn, show=True)

