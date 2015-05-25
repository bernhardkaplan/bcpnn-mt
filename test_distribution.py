import numpy as np
import utils


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


    n_x = 10
    n_y = 4
    n_total = n_x * n_y
    # distribute the elements of the (n_y x n_x) matrix across n_proc processes
    x_elements = range(n_x)
    y_elements = range(n_y)
    all_elements = [(x, y) for x in x_elements for y in y_elements]
    
    local_elements = utils.distribute_list(all_elements, n_proc, pc_id)

#    x_local = utils.distribute_list(x_elements, n_proc, pc_id)
#    y_local = utils.distribute_list(y_elements, n_proc, pc_id)
    print 'pc_id %d x_local:' % (pc_id), local_elements

