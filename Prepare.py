import sys
import utils
import simulation_parameters
import numpy as np
import CreateConnections as CC
from ParallelObject import PObject

class Preparer(PObject):
    def __init__(self, parameter_storage, comm=None):
        PObject.__init__(self, parameter_storage, comm)

    def prepare_tuning_prop(self):
        tuning_prop = utils.set_tuning_prop(self.params, mode='hexgrid', v_max=self.params['v_max'])        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
        if self.pc_id == 0:
            print "Creating tuning properties", self.pc_id, self.params['tuning_prop_means_fn']
            print "Saving tuning_prop to file:", self.pc_id, self.params['tuning_prop_means_fn']
            np.savetxt(self.params['tuning_prop_means_fn'], tuning_prop)
            # write inital bias values to file
            np.savetxt(self.params['bias_values_fn_base']+'0.dat', np.zeros(self.params['n_exc']))
        if self.comm != None:
            print 'Pid %d at Barrier in Preparer.prepare_tuning_params' % self.pc_id
            sys.stdout.flush()
            self.comm.barrier()
        return tuning_prop
        

    def prepare_spiketrains(self, tuning_prop):
        if (type(tuning_prop) == type('')):
#            try:
            tp = np.loadtxt(tuning_prop)
#            except:
#                print 'Pid %d fails to load the file ...' % self.pc_id
#                tp = self.prepare_tuning_prop(self.params)
                    
        elif (type(tuning_prop) == type(np.array([]))):
            tp = tuning_prop
        else:
            raise TypeError, 'Only filename or numpy array accepted for tuning_prop, given %s' % (str(type(tuning_prop)))

        my_units = utils.distribute_n(self.params['n_exc'], self.n_proc, self.pc_id)

        input_spike_trains = utils.create_spike_trains_for_motion(tp, self.params, contrast=.9, my_units=my_units) # write to paths defined in the params dictionary

        if self.comm != None:
            self.comm.barrier() # 
        

    def sort_gids(self):
        self.tuning_prop = np.loadtxt(self.params['tuning_prop_means_fn'])
        # sort the cells by their proximity to the stimulus into 'good_gids' and the 'rest'
        # cell in 'good_gids' should have the highest response to the stimulus
        all_gids, all_distances = utils.sort_gids_by_distance_to_stimulus(self.tuning_prop, self.params['motion_params']) 
        n_good = self.params['n_exc'] * 0.02
        self.good_gids, self.good_distances = all_gids[0:n_good], all_distances[0:n_good]
        self.rest_gids = range(self.params['n_exc'])
        for gid in self.good_gids:
            self.rest_gids.remove(gid)

        print 'Saving gids to record to', self.params['gids_to_record_fn']
        np.savetxt(self.params['gids_to_record_fn'], np.array(self.good_gids), fmt='%d')
        return self.good_gids, self.rest_gids


