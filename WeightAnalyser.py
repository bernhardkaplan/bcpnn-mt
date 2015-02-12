import sys
import os
import numpy as np
import re
import json
import utils
from PlottingScripts import plot_conn_list_as_colormap
import pylab

class WeightAnalyser(object):

    def __init__(self, params, iteration=0):
        self.params = params
        self.iteration = iteration


    def load_spikes(self):
        fn_match = self.params['spiketimes_folder'] + 'exc_spikes-'
        fn_out = self.params['spiketimes_folder'] + 'exc_spikes_merged.dat'
        utils.merge_and_sort_files(fn_match, fn_out)
        print 'Getting spikes from:', fn_out
        self.spiketrains = utils.get_spiketrains(fn_out)
        self.nspikes, self.spiketimes = utils.get_nspikes(fn_out, get_spiketrains=True)


    def get_weight_matrix_mc_mc(self):

        conn_type = 'ee'
        conn_list_fn = params['merged_conn_list_%s' % conn_type]
        if not os.path.exists(conn_list_fn):
            print 'Merging connection files...'
            utils.merge_connection_files(params, conn_type, iteration=None)

#        conn_list_fn = 'dummy_connlist.txt'
        print 'Loading ', conn_list_fn, 
        conn_list = np.loadtxt(conn_list_fn)
        print 'finished'

        M = np.zeros((self.params['n_mc'], self.params['n_mc']))
        count_conn = np.zeros((self.params['n_mc'], self.params['n_mc']))
        src_gids = np.array(conn_list[:, 0], dtype=np.int)
        hc_idx, mc_idx_in_hc, idx_in_mc = utils.get_indices_for_gid(self.params, src_gids)
        print 'Filling weight matrix', 
        for i_ in xrange(conn_list[:, 0].size):
            mc_idx_src = (conn_list[i_, 0] - 1) / self.params['n_exc_per_mc']
            mc_idx_tgt = (conn_list[i_, 1] - 1) / self.params['n_exc_per_mc']
            count_conn[mc_idx_src, mc_idx_tgt] += 1
            M[mc_idx_src, mc_idx_tgt] += conn_list[i_, 2]

        print 'finished'
        M /= count_conn
        try:
            output_fn = self.params['conn_matrix_mc_fn']
        except:
            output_fn = self.params['connections_folder'] + 'conn_matrix_mc.dat'
        print 'Saving connection matrix to:', output_fn
        np.savetxt(output_fn, M)
        print 'Plotting'
        plot_conn_list_as_colormap.plot_matrix(M)



if __name__ == '__main__':

    if len(sys.argv) > 1:
        param_fn = sys.argv[1]
        if os.path.isdir(param_fn):
            param_fn += '/Parameters/simulation_parameters.json'
        import json
        f = file(param_fn, 'r')
        print 'Loading parameters from', param_fn
        params = json.load(f)

    else:
        import simulation_parameters
        param_tool = simulation_parameters.parameter_storage()
        params = param_tool.params

    iteration = 0
    WA = WeightAnalyser(params, iteration=iteration)
    WA.get_weight_matrix_mc_mc()

    pylab.show()

#    WA.load_adj_lists(src_tgt='tgt', verbose=True)

#    WA.load_spikes()
#    WA.get_weights_to_cell(295)
#    gids = np.loadtxt(params['gids_to_record_fn'])
#    gids = [49, 91, 201, 203]
#    for i_ in xrange(len(gids)):
#        for j_ in xrange(len(gids)):
#            if i_ != j_:
#                WA.get_weight(gids[i_], gids[j_])
#        print 'nspikes %d' % gids[i_], WA.nspikes[gids[i_]]
#    WA.plot_nspikes_histogram()

#    output_fn = params['conn_mat_fn_base'] + 'ee_' + str(iteration) + '.dat'
#    if params['Cluster']:
#        WA.get_weight_matrix(plot=False, output_fn=output_fn)
#    else:
#        WA.get_weight_matrix(plot=True, output_fn=output_fn)

#    utils.convert_adjacency_lists(params)
#    WA2 = WeightAnalyser(params, iteration=iteration)
#    WA2.load_adj_lists(src_tgt='src', verbose=True)
#    output_fn = params['conn_mat_fn_base'] + 'ee_src_' + str(iteration) + '.dat'
#    WA2.get_weight_matrix_src_index(plot=True, output_fn=output_fn)
#    pylab.show()

