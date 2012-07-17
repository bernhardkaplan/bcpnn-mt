
"""
Calculate the following conductances:
g_input g_gg    g_gr    g_rg    g_rr    g_ee    g_gi    g_ie    g_ri    g_ir    g_ii

conductance = weight * nspikes
[no temporal structure considered]
g_input:  input stimulus ---> cell
_g:       well-tuned ('good') cells
_r:       rest of the population
_e:       excitatory cells
_i:       inhibitory cells
"""

import numpy as np
import simulation_parameters
import pylab
import utils
import sys

class ConductanceCalculator(object):
    def __init__(self, params=None):
        if params == None:
            network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    #        P = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
            self.params = network_params.params
        else:
            self.params = params

        self.n_exc = self.params['n_exc']
        self.output = []
        self.g_in_histograms = []
        self.output_fig = self.params['conductances_fig_fn_base']
        self.n_good = self.params['n_exc'] * .05 # fraction of 'good' (well-tuned) cells

        self.no_spikes = False

        self.load_nspikes()

        if self.params['initial_connectivity'] == 'precomputed':
            conn_fn = self.params['conn_list_ee_fn_base'] + '0.dat'
        else:
            conn_fn = self.params['random_weight_list_fn'] + '0.dat'
        print 'Calling utils.get_conn_dict(..., %s)' % conn_fn
        self.conn_dict_exc = utils.get_conn_dict(self.params, conn_fn)

        conn_fn = self.params['conn_list_ei_fn']
        print 'Calling utils.get_conn_dict(..., %s)' % conn_fn
        self.conn_dict_exc_inh = utils.get_conn_dict(self.params, conn_fn)

        conn_fn = self.params['conn_list_ie_fn']
        print 'Calling utils.get_conn_dict(..., %s)' % conn_fn
        self.conn_dict_inh_exc = utils.get_conn_dict(self.params, conn_fn)

        conn_fn = self.params['conn_list_ii_fn']
        print 'Calling utils.get_conn_dict(..., %s)' % conn_fn
        self.conn_dict_inh_inh = utils.get_conn_dict(self.params, conn_fn)


        fig_width_pt = 800.0  # Get this from LaTeX using \showthe\columnwidth
        inches_per_pt = 1.0/72.27               # Convert pt to inch
        golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fig_height = fig_width*golden_mean      # height in inches
        fig_size =  [fig_width,fig_height]
        params = {#'backend': 'png',
                  'axes.labelsize': 12,
#                  'text.fontsize': 14,
#                  'legend.fontsize': 10,
#                  'xtick.labelsize': 8,
#                  'ytick.labelsize': 8,
#                  'text.usetex': True,
                  'figure.figsize': fig_size}
        pylab.rcParams.update(params)
        pylab.subplots_adjust(bottom=0.30)

    def load_nspikes(self): 
        fn = self.params['exc_spiketimes_fn_merged'] + '0.ras'
        try:
            self.nspikes_exc = utils.get_nspikes(fn, self.params['n_exc'])
        except:
            print 'No spikes found in ', fn
            self.nspikes_exc = np.zeros(self.params['n_exc'])
            self.no_spikes = True

        fn = self.params['inh_spiketimes_fn_merged'] + '0.ras'
        try:
            self.nspikes_inh = utils.get_nspikes(fn, self.params['n_inh'])
        except:
            print 'No spikes found in ', fn
            self.nspikes_inh = np.zeros(self.params['n_inh'])


    def sort_gids(self):
        print 'ConductanceCalculator.sort_gids'
        self.tuning_prop = np.loadtxt(self.params['tuning_prop_means_fn'])
        # sort the cells by their proximity to the stimulus into 'good_gids' and the 'rest'
        # cell in 'good_gids' should have the highest response to the stimulus
        all_gids, all_distances = utils.sort_gids_by_distance_to_stimulus(self.tuning_prop, self.params['motion_params']) 
        self.good_gids, self.good_distances = all_gids[0:self.n_good], all_distances[0:self.n_good]
        self.rest_gids = range(self.n_exc)
        for gid in self.good_gids:
            self.rest_gids.remove(gid)


    def get_input_spikes(self, gids=None, label=None):
        print 'ConductanceCalculator.get_input_spikes', label

        if gids == None:
            gids = range(self.n_exc)
        n_cells = len(gids)
        self.n_input_spikes = np.zeros(n_cells)
        self.g_input = np.zeros(n_cells)
        fn_base = self.params['input_st_fn_base']
        w = self.params['w_input_exc'] * 1000. # uS --> nS
        for i in xrange(n_cells):
            tgt = gids[i]
            fn = fn_base + '%d.npy' % (tgt)
            d = np.load(fn)
            self.n_input_spikes[i] = d.size
            self.g_input[i] = w * d.size
        s, mean, std = self.g_input.sum(), self.g_input.mean(), self.g_input.std()
        if label == None:
            label = 'G_input_all'
        print '%s: Sum: %.3e; Average: %.3e +- %.3e' % (label, s, mean, std)
        self.output.append((mean, std, label))

        # make a histogram
        n_bins = 40
        count, bins = np.histogram(self.g_input, bins=n_bins)
        self.g_in_histograms.append((count, bins, label))



    def get_cond(self, conn_dict, src_gids, tgt_gids, nspikes, label):

        g_in = np.zeros(len(tgt_gids))

        for i_, tgt in enumerate(tgt_gids):
            all_srcs = conn_dict[tgt]['sources']
            w_in = conn_dict[tgt]['w_in']
            srcs = set(all_srcs).intersection(set(src_gids))
            for j_, src in enumerate(srcs):
                w_index = all_srcs.index(src)
                g_in[i_] += w_in[w_index] * nspikes[src]

        mean, std = g_in.mean(), g_in.std()
        print '%s = %.3e +- %.3e' % (label, mean, std)
        self.output.append((mean, std, label))

        # make a histogram
        n_bins = 40
        count, bins = np.histogram(g_in, bins=n_bins)
        self.g_in_histograms.append((count, bins, label))


    def get_cond_to_inh(self, src_gids):
        pass
    

    def plot(self):

        """
        Plots all elements stored in self.output
        self.output[0] : mean
        self.output[1] : std
        self.output[2] : label
        """

        fig = pylab.figure()
        ax = fig.add_subplot(111)

        width = .5
        spacing = 4.
        x_axis = np.arange(0, len(self.output) * spacing, spacing) - width
        xlabels = []
        bars = []
        for i in xrange(len(self.output)):
            bars.append(ax.bar(x_axis[i] - width, self.output[i][0], width, yerr=self.output[i][1]))#, error_kw=dict(elinewidth=4, ecolor='green')))
            xlabels.append(self.output[i][2])
        
        xtickNames = pylab.setp(ax, xticklabels=xlabels)
        pylab.setp(xtickNames, rotation=45, fontsize=12)
        ax.set_ylabel('Conductance [nS]')
        ax.set_xticks(x_axis)
        ax.set_xticklabels(xlabels)

        def autolabel(rects):
            # attach some text labels
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%.1e'%(height), ha='center', va='bottom',fontsize=10)

        for rect in bars:
            autolabel(rect)
        print '\nSaving to ', self.output_fig
        pylab.savefig(self.output_fig)


    
    def plot_g_in_histograms(self):

        fig_width_pt = 800.0  # Get this from LaTeX using \showthe\columnwidth
        inches_per_pt = 1.0/72.27               # Convert pt to inch
        golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fig_height = fig_width*golden_mean      # height in inches
        fig_size =  [fig_height, fig_width]
        params = {#'backend': 'png',
                  'axes.labelsize': 12,
#                  'text.fontsize': 14,
#                  'legend.fontsize': 10,
#                  'xtick.labelsize': 8,
#                  'ytick.labelsize': 8,
#                  'text.usetex': True,
                  'figure.figsize': fig_size}
        pylab.rcParams.update(params)
        pylab.subplots_adjust(hspace=1.00)

        fig = pylab.figure()
        n_fig_y = len(self.g_in_histograms)
        n_fig_y_per_page = 4
        fig_on_page = 1
        fig_fn_base = self.params['conductances_hist_fig_fn_base']
        fig_cnt = 0
        for i in xrange(n_fig_y):
            count, bins, label = self.g_in_histograms[i][0], self.g_in_histograms[i][1], self.g_in_histograms[i][2]
            bin_width = bins[1] - bins[0]
            y_ticks = [np.min(count), .4 * np.max(count), .8 * np.max(count)]

            if fig_on_page == n_fig_y_per_page:
                fig_fn = fig_fn_base + '%d.png' % fig_cnt
                print 'Saving conductance histograms to:', fig_fn
                pylab.savefig(fig_fn)
                fig = pylab.figure()
                fig_on_page = 1
                fig_cnt += 1
            ax = fig.add_subplot(n_fig_y_per_page, 1, fig_on_page)
            ax.bar(bins[:-1], count, width=bin_width, label=label)
            ax.set_yticks(y_ticks)
            ax.legend()
            fig_on_page += 1

        fig_fn = fig_fn_base + '%d.png' % fig_cnt
        print 'Saving conductance histograms to:', fig_fn
        pylab.savefig(fig_fn)


#sort_cells --> good_gids, rest_gids


#get_sources(cell)

#g_input[cell] = n_input_spikes[cell] * w_input_exc

#for tgt in good_gids:
#    sources = get_sources(tgt)
#    for src in sources:
#        g_gg[gid] += w_src-tgt* nspikes[src]

def run_all(params=None):
    C = ConductanceCalculator(params)
    C.sort_gids()
    C.get_input_spikes(C.good_gids, label='G_input_good')
    C.get_input_spikes(C.rest_gids, label='G_input_rest')
    if C.no_spikes:
        print 'No spikes found, will quit'
        return
    C.get_cond(C.conn_dict_exc, C.good_gids, C.good_gids, C.nspikes_exc, label='G_good_good')
    C.get_cond(C.conn_dict_exc, C.good_gids, C.rest_gids, C.nspikes_exc, label='G_good_rest')
    C.get_cond(C.conn_dict_exc, C.rest_gids, C.good_gids, C.nspikes_exc, label='G_rest_good')
    C.get_cond(C.conn_dict_exc, C.rest_gids, C.rest_gids, C.nspikes_exc, label='G_rest_rest')
    C.get_cond(C.conn_dict_exc_inh, C.good_gids, range(C.params['n_inh']), C.nspikes_exc, label='G_good_inh')
    C.get_cond(C.conn_dict_exc_inh, C.rest_gids, range(C.params['n_inh']), C.nspikes_exc, label='G_rest_inh')
    C.get_cond(C.conn_dict_inh_exc, range(C.params['n_inh']), C.good_gids, C.nspikes_inh, label='G_inh_good')
    C.get_cond(C.conn_dict_inh_exc, range(C.params['n_inh']), C.rest_gids, C.nspikes_inh, label='G_inh_rest')
    C.get_cond(C.conn_dict_inh_exc, range(C.params['n_inh']), range(C.params['n_exc']), C.nspikes_inh, label='G_inh_exc')
    C.get_cond(C.conn_dict_inh_inh, range(C.params['n_inh']), range(C.params['n_inh']), C.nspikes_inh, label='G_inh_inh')
    C.plot()
    C.plot_g_in_histograms()


if __name__ == '__main__':
    run_all()


#get_nspikes_exc
#get_nspikes_inh


    

