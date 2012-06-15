import sys
import utils
import simulation_parameters
import numpy as np
import CreateConnections as CC


class Preparer(object):
    def __init__(self, comm=None):
        if comm != None:
            self.pc_id, self.n_proc = comm.rank, comm.size
        else:
            self.pc_id, self.n_proc = 0, 1
        self.comm = comm


    def prepare_tuning_prop(self, params):
        tuning_prop = utils.set_tuning_prop(params, mode='hexgrid', v_max=params['v_max'])        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
        if self.pc_id == 0:
            print "Creating tuning properties", self.pc_id, params['tuning_prop_means_fn']
            print "Saving tuning_prop to file:", self.pc_id, params['tuning_prop_means_fn']
            np.savetxt(params['tuning_prop_means_fn'], tuning_prop)
            # write inital bias values to file
            np.savetxt(params['bias_values_fn_base']+'0.dat', np.zeros(params['n_exc']))
        if self.comm != None:
            print 'Pid %d at Barrier in Preparer.prepare_tuning_params' % self.pc_id
            sys.stdout.flush()
            self.comm.barrier()
        

    def prepare_spiketrains(self, params, tuning_prop):
        if (type(tuning_prop) == type('')):
            try:
                tp = np.loadtxt(tuning_prop)
            except:
                print 'Pid %d fails to load the file ...' % self.pc_id
                self.prepare_tuning_prop(params)

                    
        elif (type(tuning_prop) == type(np.array)):
            tp = tuning_prop

        my_units = utils.distribute_n(params['n_exc'], self.n_proc, self.pc_id)

        input_spike_trains = utils.create_spike_trains_for_motion(tp, params, contrast=.9, my_units=my_units) # write to paths defined in the params dictionary

        if self.comm != None:
            self.comm.barrier() # 
        




