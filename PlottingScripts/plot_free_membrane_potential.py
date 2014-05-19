import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
print 'cmd_subfolder', cmd_subfolder
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import numpy as np
import pylab
import utils
import json


class Plotter(object):
    def __init__(self, params):

        self.params = params
        self.cell_cnt = 0
        self.colorlist = utils.get_colorlist()
        self.traces = {}  # store traces for each GID
        self.tp = np.loadtxt(self.params['tuning_prop_means_fn'])
        self.t_axis = None
        self.y_lim_global = [np.inf, -np.inf]

    def plot_volt(self, fn, gid=None, n=1):
        print 'loading', fn
        d = np.loadtxt(fn)

        if gid == None:
            recorded_gids = np.unique(d[:, 0])
            gids = random.sample(recorded_gids, n)
            print 'plotting random gids:', gids
        elif gid == 'all':
            gids = np.unique(d[:, 0])
        elif type(gid) == type([]):
            gids = gid
        else:
            gids = [gid]
        
        for gid in gids:
            time_axis, volt = utils.extract_trace(d, gid)
            pylab.plot(time_axis, volt, label='%d' % gid, lw=2, c=self.colorlist[self.cell_cnt % len(self.colorlist)])
            self.cell_cnt += 1

        parts = fn.rsplit('.')
        output_fn = "" 
        for i in xrange(len(parts)-1):
            output_fn += "%s" % parts[i] + '.'
        output_fn += 'png'
        pylab.legend()
        pylab.title(fn)
    #    print 'Saving to', output_fn
    #    pylab.savefig(output_fn)


#    def plot_average_volt(fn, gid=None, n=1):
#        print 'Plotting average voltage; loading', fn
#        d = np.loadtxt(fn)
#        if gid == None:
#            gid_range = np.unique(d[:, 0])
#            gids = np.random.randint(np.min(gid_range), np.max(gid_range) + 1, n)
#            print 'plotting random gids:', gids
#        elif gid == 'all':
#            gids = np.unique(d[:, 0])
#        elif type(gid) == type([]):
#            gids = gid
#        else:
#            gids = [gid]
#        time_axis, volt = utils.extract_trace(d, gids[0])
#        all_volt = np.zeros((time_axis.size, len(gids)))
#        for i_, gid in enumerate(gids):
#            time_axis, volt = utils.extract_trace(d, gid)
#            print 'gid %d v_mean, std = %.2f +- %.2f; min %.2f max %.2f, diff %.2f ' % (gid, volt.mean(), volt.std(), volt.min(), volt.max(), volt.max() - volt.min())
#            all_volt[:, i_] = volt
#        avg_volt = np.zeros((time_axis.size, 2))
#        for t in xrange(time_axis.size):
#            avg_volt[t, 0] = all_volt[t, :].mean()
#            avg_volt[t, 1] = all_volt[t, :].std()
#        print 'Average voltage and std: %.2e +- %.2e (%.2e)' % (avg_volt[:, 0].mean(), avg_volt[:, 0].std(), avg_volt[:, 1].mean())
#        pylab.errorbar(time_axis, avg_volt[:, 0], yerr=avg_volt[:, 1], lw=3, c='k') 

    def collect_data(self, fn):

        print 'plot_free_membrane_potential plotter loads:', fn
        d = np.loadtxt(fn)
        gids = np.array(np.unique(d[:, 0]), dtype=int)
        for gid in gids:
            time_axis, volt = utils.extract_trace(d, gid)
            self.traces[gid] = volt
            self.y_lim_global[0] = min(self.y_lim_global[0], np.min(volt))
            self.y_lim_global[1] = max(self.y_lim_global[1], np.max(volt))
            self.t_axis = time_axis


    def plot_cells_sorted(self, ax, sort_idx=2):
        """
        sort_idx = 0 means sorting according to x-pos
        sort_idx = 1 means sorting according to y-pos
        sort_idx = 2 means sorting according to v_x
        sort_idx = 3 means sorting according to v_y
        """
        (gids_recorders, tp_recorders) = self.get_recorder_tuning_properties()
        # convert the dictionary to an array that can be sorted
        n_cells = len(gids_recorders)
        tp_rec_unsrtd = np.zeros((n_cells, self.tp.shape[1]))
        for i_, gid in enumerate(gids_recorders):
            tp_rec_unsrtd[i_, :] = tp_recorders[gid]

        idx_sorted = np.argsort(tp_rec_unsrtd[:, sort_idx])
        # idx_sorted holds the order of gids for gids_recorders
        for i_, gid in enumerate(gids_recorders):
            gid_sorted = gids_recorders[idx_sorted[i_]]
#            print 'gid', gid, tp_recorders[gids_recorders[idx_sorted[i_]]], 'gid_sorted', gid_sorted
            v_x = tp_recorders[gids_recorders[idx_sorted[i_]]][sort_idx]
            rf_x = tp_recorders[gids_recorders[idx_sorted[i_]]][0]

            ax = fig.add_subplot(n_cells, 1, i_ + 1)
            ax.plot(self.t_axis, self.traces[gid_sorted])
#            print 'debug', ax.get_yticks()
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_ylim(self.y_lim_global)
            ax.set_yticks([.5 * (self.y_lim_global[0] - self.y_lim_global[1])])
#            print 'debug', .5 * (self.y_lim_global[1] - self.y_lim_global[0]), self.traces[gid_sorted].mean(), self.y_lim_global
            print 'debug', v_x, rf_x
            ax.set_yticklabels(['%.2f %.2f' % (rf_x, v_x)])





    def get_recorder_tuning_properties(self):

        assert (len(self.traces.keys()) > 0), 'No data loaded!\nRun collect data on the filenames gathered by utils.find_files'

        print 'Loading:', self.params['recorder_neurons_gid_mapping']
        f = file(self.params['recorder_neurons_gid_mapping'], 'r')
        gid_map = json.load(f)
        gids_recorders = np.array(gid_map.keys(), dtype=int)
        n_cells = len(gids_recorders)
        tp_recorders = {}


        for i_ in xrange(n_cells):
            # get the position for the cell
            gid_rec = self.traces.keys()[i_]
            gid_real = gid_map[str(gid_rec)]
            tp_recorders[gid_rec] = self.tp[gid_real, :]
#            print 'GID recorder', gid_rec, gid_real, tp_recorders[gid_rec]

        return (gids_recorders, tp_recorders)






    
if __name__ == '__main__':

    if len(sys.argv) == 1:
        import simulation_parameters
        ps = simulation_parameters.parameter_storage()
        params = ps.params
    else:
        # folder / parameter file option
        param_fn = sys.argv[1]
        params = utils.load_params(param_fn)

    fns = utils.find_files(params['spiketimes_folder'], params['free_vmem_fn_base'])
    print 'fns', fns

    fig = pylab.figure()
    
    P = Plotter(params)
    for fn in fns:
        path = params['spiketimes_folder'] + fn

        P.collect_data(path)

    P.plot_cells_sorted(fig, sort_idx=0)

        # SIMPLE PLOTTING, plot without any sorting or other stuff
#        P.plot_volt(path, gid='all')

    pylab.show()
