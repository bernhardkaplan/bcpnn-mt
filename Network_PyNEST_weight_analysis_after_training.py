import sys
import os
import numpy as np
import re
import json
import utils
import MergeSpikefiles
import pylab



class WeightAnalyser(object):

    def __init__(self, params, iteration=0):
        self.params = params
        self.iteration = iteration


    def get_filenames(self, iteration=None, src_tgt='tgt'):
        """
        As connections are stored as adjacency in separate files (for each processing node a separate file),
        the connections_folder is filtered by the adjacency list filename 'adj_list_tgt_fn_base'
        """
        if iteration == None:
            iteration = self.iteration
        fns = []
        folder = self.params['connections_folder']

        # find files written by different processes
        conn_mat_fn = self.params['adj_list_tgt_fn_base'] + 'AS_(\d)_(\d+).json'
        to_match = conn_mat_fn.rsplit('/')[-1]
        for fn in os.listdir(os.path.abspath(folder)):
            m = re.match(to_match, fn)
            if m:
                it_ = int(m.groups()[0])
                if it_ == iteration:
                    fn_abs_path = self.params['connections_folder'] +  fn
                    fns.append(fn_abs_path)
        print 'Found files:', fns
        return fns
                
        
    def load_adj_lists(self, src_tgt='tgt'):
        if src_tgt == 'tgt':
            fns = self.get_filenames()
            self.adj_list = {}
            for fn in fns:
                f = file(fn, 'r')
                print 'Loading weights:', fn
                d = json.load(f)
                self.adj_list.update(d)
        if src_tgt == 'src':
            self.adj_list = utils.convert_adjacency_lists(self.params)
        return self.adj_list

    def get_weights_to_cell(self, tgt_gid):

        print 'Cells projecting to tgt:', tgt_gid
        d = np.array(self.adj_list[str(tgt_gid)])
        src_gids = d[:, 0] # source gids projecting to the gid
        print 'src_gids', src_gids


    def get_weight(self, src_gid, tgt_gid):

        if self.adj_list.has_key(str(tgt_gid)):
            d = np.array(self.adj_list[str(tgt_gid)])
        else:
#            print 'Could not find tgt_gid %d in the connection files' % (tgt_gid)
            return False
        src_gids = d[:, 0] # source gids projecting to the gid
        idx = (src_gid == src_gids).nonzero()[0]
        if len(idx) > 0:
            w = d[(src_gid == src_gids).nonzero()[0], 1]
#            print 'w (%d - %d) = %f' % (src_gid, tgt_gid, w)
            return w
        else:
#            print 'Could not find a projection from %d to %d in the connection files' % (src_gid, tgt_gid)
            return False
#            for tgt in tgt_gids:


    
    def load_spikes(self):
        fn_match = self.params['spiketimes_folder'] + 'exc_spikes-'
        fn_out = self.params['spiketimes_folder'] + 'exc_spikes_merged.dat'
        utils.merge_and_sort_files(fn_match, fn_out)
        print 'Getting spikes from:', fn_out
        self.spiketrains = utils.get_spiketrains(fn_out)
        self.nspikes, self.spiketimes = utils.get_nspikes(fn_out, get_spiketrains=True)


    def plot_nspikes_histogram(self):

        idx_sorted = np.argsort(self.nspikes)
        print 'idx_sorted', idx_sorted[-40:]
        print 'nspikes sorted', self.nspikes[idx_sorted[-40:]]
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        gids = range(len(self.nspikes))
        ax.bar(gids, self.nspikes)

        ax.set_xlabel('GID')
        ax.set_ylabel('# spikes')


    def get_weight_matrix_src_index(self, plot=True, output_fn=None):
        w = np.zeros((self.params['n_exc'], self.params['n_exc']))
        for src in xrange(self.params['n_exc']):
            if self.adj_list.has_key(src + 1):
                tgt_weight_array = np.array(self.adj_list[src + 1])
                for i_ in xrange(tgt_weight_array[:, 0].size):
                    tgt, w_ij = tgt_weight_array[i_, :]
                    w_ij = float(w_ij)
                    w[src, int(tgt) - 1] = w_ij

        if output_fn != None:
            print 'Saving connection matrix to:', output_fn
            np.savetxt(output_fn, w)
        if plot:
            fig = pylab.figure()
            ax = fig.add_subplot(111)
            print "plotting ...."
            cax = ax.pcolormesh(w)#, edgecolor='k', linewidths='1')
            ax.set_ylim(0, w.shape[0])
            ax.set_xlim(0, w.shape[1])
            pylab.colorbar(cax)
            ax.set_title('Matrix from source indexed adjacency lists')
        return w
#            output_fig = self.params['connection_matrix_fig'] + str(self.iteration) + '.png'
#            print 'Savig fig to:', output_fig
#            pylab.savefig(output_fig, dpi=400)




    def get_weight_matrix(self, plot=True, output_fn=None):
        w = np.zeros((self.params['n_exc'], self.params['n_exc']))
        for tgt in xrange(self.params['n_exc']):
            if self.adj_list.has_key(str(tgt + 1)):
                src_weight_array = np.array(self.adj_list[str(tgt + 1)])
                for i_ in xrange(src_weight_array[:, 0].size):
                    src, w_ij = src_weight_array[i_, :]
                    w[src - 1, tgt] = w_ij


        if output_fn != None:
            print 'Saving connection matrix to:', output_fn
            np.savetxt(output_fn, w)

        if plot:
            fig = pylab.figure()
            ax = fig.add_subplot(111)
            print "plotting ...."

            cax = ax.pcolormesh(w)#, edgecolor='k', linewidths='1')
            ax.set_ylim(0, w.shape[0])
            ax.set_xlim(0, w.shape[1])
            pylab.colorbar(cax)

            output_fig = self.params['connection_matrix_fig'] + str(self.iteration) + '.png'
            print 'Savig fig to:', output_fig
            pylab.savefig(output_fig, dpi=400)


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
    WA.load_adj_lists(src_tgt='tgt')

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

    output_fn = params['conn_mat_fn_base'] + 'ee_' + str(iteration) + '.dat'
    WA.get_weight_matrix(plot=True, output_fn=output_fn)



#    utils.convert_adjacency_lists(params)
    WA2 = WeightAnalyser(params, iteration=iteration)
    WA2.load_adj_lists(src_tgt='src')
    output_fn = params['conn_mat_fn_base'] + 'ee_src_' + str(iteration) + '.dat'
    WA2.get_weight_matrix_src_index(plot=True, output_fn=output_fn)
    pylab.show()
