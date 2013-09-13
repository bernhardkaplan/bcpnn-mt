import sys
import os
import numpy as np
import re
import json
import utils
import MergeSpikefiles
import pylab



class WeightAnalyser(object):

    def __init__(self, params):
        self.params = params


    def get_filenames(self, iteration=0):
        fns = []
        folder = self.params['connections_folder']

        # find files written by different processes
        conn_mat_fn = self.params['adj_list_tgt_fn_base'] + 'AS_(\d)_(\d).json'
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
                
        
    def load_weights(self):

        fns = self.get_filenames()

        self.adj_list = {}
        for fn in fns:
            f = file(fn, 'r')
            print 'Debug loading weights:', fn
            d = json.load(f)
            self.adj_list.update(d)


    def get_weights_to_cell(self, tgt_gid):

        print 'Cells projecting to tgt:', tgt_gid
        d = np.array(self.adj_list[str(tgt_gid)])
        src_gids = d[:, 0] # source gids projecting to the gid


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


    def get_weight_matrix(self, plot=True):
        w = np.zeros((self.params['n_exc'], self.params['n_exc']))
        for tgt in xrange(self.params['n_exc']):
            if self.adj_list.has_key(str(tgt)):
                src_weight_array = np.array(self.adj_list[str(tgt)])
#                exit(1)
#                srcs, weights = self.adj_list[str(tgt)]
                for i_ in xrange(src_weight_array[:, 0].size):
                    src, w_ij = src_weight_array[i_, :]
                    w[src - 1, tgt] = w_ij

        if plot:
            fig = pylab.figure()
            ax = fig.add_subplot(111)
            print "plotting ...."

            cax = ax.pcolor(w)#, edgecolor='k', linewidths='1')
            ax.set_ylim(0, w.shape[0])
            ax.set_xlim(0, w.shape[1])
            pylab.colorbar(cax)
            pylab.show()



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

    WA = WeightAnalyser(params)
    WA.load_weights()
    WA.load_spikes()
#    WA.get_weights_to_cell(201)
#    gids = [49, 91, 201, 203]
#    for i_ in xrange(len(gids)):
#        for j_ in xrange(len(gids)):
#            if i_ != j_:
#                WA.get_weight(gids[i_], gids[j_])
#        gid_ = gids[i_]
#        print 'nspikes %d' % gid_, WA.nspikes[gid_]
#    WA.plot_nspikes_histogram()
    WA.get_weight_matrix()

    pylab.show()
#    print 'WROOOOOOOOONG!!!!!!!!!'


