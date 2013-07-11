import os
import sys
import simulation_parameters
import Bcpnn
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
pc_id, n_proc = comm.rank, comm.size


sim_cnt = int(sys.argv[1])

network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
conn_list = np.loadtxt(params['conn_list_ee_fn_base'] + str(sim_cnt) + '.dat')

params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
Bcpnn.bcpnn_offline_noColumns(params, conn_list, sim_cnt, True, comm)
