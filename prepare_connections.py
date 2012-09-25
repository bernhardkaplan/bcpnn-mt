import numpy as np
import utils
import sys

import simulation_parameters
PS = simulation_parameters.parameter_storage()
params = PS.load_params()                       # params stores cell numbers, etc as a dictionary
#params['w_sigma_x'], params['w_sigma_v'] = float(sys.argv[1]), float(sys.argv[2])
PS.set_filenames()
PS.create_folders()
PS.write_parameters_to_file()
print 'n_cells=%d\tn_exc=%d\tn_inh=%d' % (params['n_cells'], params['n_exc'], params['n_inh'])
print 'w_sigma_x, v', params['w_sigma_x'], params['w_sigma_v']

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

#try:
tuning_prop = np.loadtxt(params['tuning_prop_means_fn'])
#except:
#    print 'File with tuning properties missing: %s\nPlease run: \nmpirun -np [N] python prepare_tuning_prop.py\nOR\npython prepare_tuning_prop.py' % params['tuning_prop_means_fn']
#    exit(1)

import CreateConnections as CC
#try:
CC.compute_weights_convergence_constrained(tuning_prop, params, comm)
#except:
#    print 'File with tuning properties contains wrong number of cells %s\nPlease re-run: \nmpirun -np [N] python prepare_tuning_prop.py\nOR\npython prepare_tuning_prop.py' % params['tuning_prop_means_fn']
#    exit(1)
output_fn = params['conn_list_ee_conv_constr_fn_base'] + 'merged.dat'
if pc_id == 0:
    print 'Merged connections file:', output_fn
    utils.merge_files('%spid*.dat' % params['conn_list_ee_conv_constr_fn_base'], output_fn)
