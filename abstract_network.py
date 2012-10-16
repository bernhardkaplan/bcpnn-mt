import numpy as np
import utils
import Bcpnn
import os

def prepare_input(tp, params, my_units=None):
    n_units = tp.shape[0]
    n_cells = params['n_exc'] # each unit / column can contain several cells
    dt = params['dt_rate'] # [ms] time step for the non-homogenous Poisson process 

    time = np.arange(0, params['t_stimulus'], dt)

    if (my_units == None):
        my_units = xrange(n_units)
    else:
        my_units = xrange(my_units[0], my_units[1])

    L_input = np.empty((n_cells, time.shape[0]))
    for i_time, time_ in enumerate(time):
        if (i_time % 100 == 0):
            print "t:", time_
        L_input[:, i_time] = utils.get_input(tuning_prop, params, time_/params['t_sim'])

    for unit in my_units:
        rate_of_t = np.array(L_input[unit, :]) 
        output_fn = params['input_rate_fn_base'] + str(unit) + '.dat'
        print 'output_fn:', output_fn
        np.savetxt(output_fn, rate_of_t)


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

import simulation_parameters
PS = simulation_parameters.parameter_storage()
params = PS.params
PS.create_folders()
PS.write_parameters_to_file()

n_cells = params['n_exc']
my_units = utils.distribute_n(n_cells, n_proc, pc_id)

# P R E P A R E     T U N I N G    P R O P E R T I E S
tp_fn = params['tuning_prop_means_fn']
tuning_prop = np.loadtxt(tp_fn)
#tuning_prop = utils.set_tuning_prop(params, mode='hexgrid', v_max=params['v_max'])
#np.savetxt(params['tuning_prop_means_fn'],tuning_prop)

# P R E P A R E     I N P U T 
#prepare_input(tuning_prop, params, my_units)

#input_sum = np.zeros(n_cells)
#for i in xrange(n_cells)
#    input_fn = params['input_rate_fn_base'] + str(i) + '.dat'
#    rate = np.loadtxt(input_fn)
#    input_sum[i] = rate.sum()
#np.savetxt(params['input_sum_fn'], input_sum)




# distribute the connections
conns = []
for i in xrange(n_cells):
    for j in xrange(i):
        if i != j:
            conns.append((i, j))
my_conns = utils.distribute_list(conns, n_proc, pc_id)
my_conns = [(145, 177), (65, 61)]

save_all = True
bcpnn_trace_len = params['t_stimulus'] * 2 / params['dt_rate']

p_ij_string = ''
for i in xrange(len(my_conns)):
#for i in xrange(100):
    print "Pc %d conn: \t%d - %d; \t%d / %d\t%.4f percent complete" % (pc_id, my_conns[i][0], my_conns[i][1], i, len(my_conns), i * 100./len(my_conns))
    pre_id = my_conns[i][0]
    post_id = my_conns[i][1]

    trace_len = params['t_stimulus'] / params['dt_rate']
    pre_trace = np.zeros(bcpnn_trace_len)
    d = np.loadtxt(params['input_rate_fn_base'] + str(pre_id) + '.dat')
    pre_trace[:trace_len] = d
    post_trace = np.zeros(bcpnn_trace_len)
    d = np.loadtxt(params['input_rate_fn_base'] + str(post_id) + '.dat')
    post_trace[:trace_len] = d
    post_trace[trace_len:] = 0.
    post_trace.resize(bcpnn_trace_len)

    # compute

    wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = Bcpnn.get_spiking_weight_and_bias(pre_trace, post_trace, get_traces=True, tau_z=100, f_max=1000.)

    p_ij_string += '%d\t%d\t%.6e\n' % (pre_id, post_id, pij[-1])

    if (save_all):
        weight_fn = params['weights_fn_base'] + '%d_%d.dat' % (pre_id, post_id)
        print 'Saving to ', weight_fn
        np.savetxt(weight_fn, wij)
        output_fn = params['bias_fn_base'] + "%d_%d.dat" % (pre_id, post_id)
        np.savetxt(output_fn, bias)

        output_fn = params['ztrace_fn_base'] + "%d.dat" % pre_id
        np.savetxt(output_fn, zi)
        output_fn = params['ztrace_fn_base'] + "%d.dat" % post_id
        np.savetxt(output_fn, zj)

        output_fn = params['etrace_fn_base'] + "%d.dat" % pre_id
        np.savetxt(output_fn, ei)
        output_fn = params['etrace_fn_base'] + "%d.dat" % post_id
        np.savetxt(output_fn, ej)
        output_fn = params['etrace_fn_base'] + "%d_%d.dat" % (pre_id, post_id)
        np.savetxt(output_fn, eij)

        output_fn = params['ptrace_fn_base'] + "%d.dat" % pre_id
        np.savetxt(output_fn, pi)
        output_fn = params['ptrace_fn_base'] + "%d.dat" % post_id
        np.savetxt(output_fn, pj)
        output_fn = params['ptrace_fn_base'] + "%d_%d.dat" % (pre_id, post_id)
        np.savetxt(output_fn, pij)

output_fn = params['bcpnntrace_folder'] + 'pij_%d.dat' % (pc_id)
print 'Writing p_ij output to:', output_fn
f = file(output_fn, 'w')
f.write(p_ij_string)
f.close()

if comm != None:
    comm.barrier()


if pc_id == 0:
#    tmp_fn = 'delme_tmp%d' % np.random.randint(0, 10000)
    tmp_fn = params['bcpnntrace_folder'] + 'all_pij.dat'
    cat_cmd = 'cat %s* > %s' % (params['bcpnntrace_folder'] + 'pij_', tmp_fn)
    print cat_cmd
    os.system(cat_cmd)
