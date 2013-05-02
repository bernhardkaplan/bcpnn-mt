import matplotlib
matplotlib.use('Agg')
import numpy as np

import utils
import pylab
import sys
import re
import os
import simulation_parameters
import CreateConnections as CC

class ConnectivityAnalyser(object):

    def __init__(self, params=None, comm=None):

        if params == None:
            network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
            params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
            print 'Merging connlists ...'
            os.system('python merge_connlists.py')
        else:
            self.params = params
            print 'Assuming that \n\tpython merge_connlists.py \nhas been called before in the directory %s' % params['folder_name']

        self.comm = comm
        if comm != None:
            self.pc_id, self.n_proc = comm.rank, comm.size
        self.conn_lists = {}
        self.n_fig_x = 1
        self.n_fig_y = 1

        # cell markers
        self.markersize_cell = 10
        self.markersize_min = 3
        self.markersize_max = 12
        self.shaft_width = 0.005
        self.conn_type_dict = {'e' : 'excitatory', 'i' : 'inhibitory'}

    def load_tuning_prop(self):
        print 'Loading tuning properties'
        self.tp_exc = np.loadtxt(self.params['tuning_prop_means_fn'])
        self.tp_inh = np.loadtxt(self.params['tuning_prop_inh_fn'])


    def load_connlist(self, conn_type):
        fn = self.params['merged_conn_list_%s' % conn_type]
        print 'Loading:', fn
        if not os.path.exists(fn):
            print 'Merging connlists ...'
            cmd = 'python merge_connlists.py %s' % self.params['params_fn']
            os.system(cmd)

        self.conn_lists[conn_type] = np.loadtxt(fn)


    def get_tp(self, conn_type):
        if conn_type == 'ee':
            return (self.tp_exc, self.tp_exc)
        elif conn_type == 'ei':
            return (self.tp_exc, self.tp_inh)
        elif conn_type == 'ie':
            return (self.tp_inh, self.tp_exc)
        elif conn_type == 'ii':
            return (self.tp_inh, self.tp_inh)


    def plot_num_outgoing_connections(self, conn_type, fig_cnt=1):

        fn = self.params['merged_conn_list_%s' % conn_type]
        print 'Loading:', fn
        if not os.path.exists(fn):
            print 'Merging connlists ...'
            cmd = 'python merge_connlists.py %s' % self.params['params_fn']
            os.system(cmd)

        if not self.conn_lists.has_key(conn_type):
            self.load_connlist(conn_type)
        conn_list = self.conn_lists[conn_type]

        (n_src, n_tgt, syn_type) = utils.resolve_src_tgt(conn_type, self.params)
        n_tgts = np.zeros(n_src)
        w_out = np.zeros(n_src)
        n_srcs = np.zeros(n_tgt)
        w_in = np.zeros(n_tgt)
        for i in xrange(conn_list[:, 0].size):
            src, tgt, w, delay = conn_list[i, :4]
            n_tgts[src] += 1 # count how often src connects to some other cell
            n_srcs[tgt] += 1 # count how often tgt is the target cell
            w_out[src] += w
            w_in[tgt] += w

        n_out_mean = n_tgts.mean()
        n_out_sem = n_tgts.std() / np.sqrt(n_src)
        n_in_mean = n_srcs.mean()
        n_in_sem = n_srcs.std() / np.sqrt(n_tgt)
        print '\nConvergence:\nNumber of %s cells that get no %s input: %d = %.2f percent' % (self.conn_type_dict[conn_type[1]], self.conn_type_dict[conn_type[0]], (n_srcs == 0).nonzero()[0].size, (n_srcs==0).nonzero()[0].size / n_tgt * 100.)
        print 'Divergence: Number of %s cells that have no %s target: %d = %.2f percent\n' % (self.conn_type_dict[conn_type[0]], self.conn_type_dict[conn_type[1]], (n_tgts == 0).nonzero()[0].size, (n_tgts==0).nonzero()[0].size/float(n_src)*100.)
        print '%s cells that do not connect to other %s cells:' % (self.conn_type_dict[conn_type[0]], self.conn_type_dict[conn_type[1]]), (n_tgts == 0).nonzero()[0]
        print 'Weight in %.2e +- %.2e' % (w_in.mean(), w_in.std())
        print 'Weight out %.2e +- %.2e' % (w_out.mean(), w_out.std())
#        print 'debug n_tgts', n_tgts
#        print 'debug w_out', w_out

        # OUTGOING CONNECTIONS
        # plot number of outgoing connections
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        ax.bar(range(n_src), n_tgts, width=1)
        ax.set_xlim((0, n_src))
        ax.set_xlabel('Source index')
        ax.set_ylabel('Number of outgoing connections')
        title = 'Every %s cell connects on average to $%.2f\pm%.2f \, (%.1f\pm%.2f\, \%% $ of the) %s cells' % (self.conn_type_dict[conn_type[0]], \
                n_out_mean, n_out_sem, n_out_mean / n_tgt * 100., n_out_sem / n_tgt * 100.,  self.conn_type_dict[conn_type[1]])
        print title
        ax.set_title(title)

        # INCOMING CONNECTIONS
        # plot number of outgoing connections
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt + 1)
        ax.bar(range(n_tgt), n_srcs, width=1)
        ax.set_xlim((0, n_tgt))
        ax.set_xlabel('Target index')
        ax.set_ylabel('Number of incoming connections')
        title = 'Every %s cell receives on average input from $ %.2f \pm %.2f \,(%.1f \pm %.2f \, \%% $ of the) %s cells' % (self.conn_type_dict[conn_type[1]], \
                n_in_mean, n_in_sem, n_in_mean / n_tgt * 100., n_in_sem / n_tgt * 100.,  self.conn_type_dict[conn_type[0]])
        print title
        ax.set_title(title)


        self.fig = self.create_fig()
        ax = self.fig.add_subplot(221)
        ax.bar(range(n_src), w_out, width=1)
        ax.set_xlim((0, n_src))
        ax.set_xlabel('Source neuron')
        ax.set_ylabel('Sum of outgoing weights')

        ax = self.fig.add_subplot(222)
        ax.bar(range(n_tgt), w_in, width=1)
        ax.set_xlim((0, n_tgt))
        ax.set_xlabel('Target neuron')
        ax.set_ylabel('Sum of incoming weights')

        # plot the sorted weights
        w_out_srt = w_out.copy()
        w_out_srt.sort()
        ax = self.fig.add_subplot(223)
        ax.bar(range(n_src), w_out_srt, width=1)
        ax.set_xlim((0, n_src))
        ax.set_xlabel('Source neuron')
        ax.set_ylabel('Sum of outgoing weights')

        w_in_srt = w_in.copy()
        w_in_srt.sort()
        ax = self.fig.add_subplot(224)
        ax.bar(range(n_tgt), w_in_srt, width=1)
        ax.set_xlim((0, n_tgt))
        ax.set_xlabel('Source neuron')
        ax.set_ylabel('Sum of incoming weights')




    def plot_tgt_connections(self, conn_type, gids_to_plot=None, fig_cnt=1):
        """
        For all gids_to_plot all outgoing connections and the centroid / center of gravitiy is plotted.
        conn_type = ['ee', 'ei', 'ie', 'ii']
        """

        tp_src, tp_tgt = self.get_tp(conn_type)
        if gids_to_plot == None:
            if conn_type[0] == 'e':
                gids_to_plot = np.loadtxt(self.params['gids_to_record_fn'], dtype=np.int)
                gids_to_plot = [gids_to_plot[0]]
            else:
                gids_to_plot = np.random.randint(0, tp_src[:, 0].size, 1)

        fn = self.params['merged_conn_list_%s' % conn_type]
        print 'Loading:', fn
        if not os.path.exists(fn):
            print 'Merging connlists ...'
            cmd = 'python merge_connlists.py %s' % self.params['params_fn']
            os.system(cmd)

        if not self.conn_lists.has_key(conn_type):
            self.load_connlist(conn_type)
        conn_list = self.conn_lists[conn_type]

        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        for i_, src_gid in enumerate(gids_to_plot):
            (x, y, u, v) = tp_src[src_gid, :]

            tgts = utils.get_targets(conn_list, src_gid)
            tgt_ids = np.array(tgts[:, 1], dtype=np.int)
            weights = tgts[:, 2]
            delays = tgts[:, 3]
            print 'weights size', weights.size
            if weights.size > 0:
                c_x, c_v = self.get_cg_vec(tp_src[src_gid, :], tp_tgt[tgt_ids, :], weights)
                markersizes = utils.linear_transformation(weights, self.markersize_min, self.markersize_max)
            else:
                print '\n WARNING: Cell %d has no outgoing connections!\n' % src_gid
                c_x, c_v = [(0, 0), (0, 0)]
                markersizes = []

            vector_conn_centroid_x_minus_vsrc = (c_x[0] - u, c_x[1] - v)
#            c_x *= 100.
            for j_, tgt_gid in enumerate(tgts[:, 1]):
                (x_tgt, y_tgt, u_tgt, v_tgt) = tp_tgt[tgt_gid, :]
                xdiff = (x_tgt - x)
                ydiff = (y_tgt - y)
                ax.plot(xdiff, ydiff, 'o', markersize=markersizes[j_], color='r')
#                ax.quiver(
            preferred_direction = ax.quiver(0, 0, u, v, angles='xy', scale_units='xy', scale=1, color='r', headwidth=3, width=self.shaft_width * 2, linewidths=(1,), edgecolors=('k'), zorder=100000)
            connection_centroid = ax.quiver(0, 0, c_x[0], c_x[1], angles='xy', scale_units='xy', scale=1, color='k', headwidth=3, width=self.shaft_width * 2, linewidths=(1,), edgecolors=('k'), zorder=100000)
            diff_v = ax.quiver(0, 0, vector_conn_centroid_x_minus_vsrc[0], vector_conn_centroid_x_minus_vsrc[1], angles='xy', scale_units='xy', scale=1, color='y', headwidth=3, width=self.shaft_width * 2, linewidths=(1,), edgecolors=('k'), zorder=100000)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.plot((0, 0), (ylim[0], ylim[1]), 'k--')
            ax.plot((xlim[0], xlim[1]), (0, 0), 'k--')
            quiverkey_length = .05 * (xlim[1] - xlim[0] + ylim[1] - ylim[0])
            ax.quiverkey(preferred_direction, .1, .85, quiverkey_length, 'Preferred direction')
            ax.quiverkey(connection_centroid, .1, .75, quiverkey_length, 'Connection centroid')
            ax.quiverkey(diff_v, .8, .95, quiverkey_length, 'Difference vector')
#            ax.plot((0, 0), (-.2, .2), 'k--')
#            ax.plot((-.2, .2), (0, 0), 'k--')


    def create_fig(self):
        print "Creating fig..."
        self.fig_size = (14, 10)
        self.fig = pylab.figure(figsize=self.fig_size)
        pylab.subplots_adjust(hspace=0.4)
        pylab.subplots_adjust(wspace=0.35)
        return self.fig



    def plot_tuning_vs_conn_cg(self, conn_type, show=False):
        """
        For each source cell, loop through all target connections and compute the
        scalar (dot) product between the preferred direction of the source cell and the center of gravity of the connection vector
        (both in the spatial domain and the direction domain)
        c_x_i = sum_j w_ij * (x_i - x_j) # x_ are position vectors of the cell
        c_v_i = sum_j w_ij * (v_i - v_j) # v_ are preferred directions
        """
        (n_src, n_tgt, tp_src, tp_tgt) = utils.resolve_src_tgt_with_tp(conn_type, self.params)
        fn = self.params['merged_conn_list_%s' % conn_type]
        print 'Loading:', fn
        if not os.path.exists(fn):
            print 'Merging connlists ...'
            cmd = 'python merge_connlists.py %s' % self.params['params_fn']
            os.system(cmd)

        conn_list = np.loadtxt(fn)

#        conn_mat_fn = self.params['conn_mat_fn_base'] + '%s.dat' % (conn_type)
#        if os.path.exists(conn_mat_fn):
#            print 'Loading', conn_mat_fn
#            w = np.loadtxt(conn_mat_fn)
#        else:
#            w, delays = utils.convert_connlist_to_matrix(params['merged_conn_list_%s' % conn_type], n_src, n_tgt)
#            print 'Saving:', conn_mat_fn
#            np.savetxt(conn_mat_fn, w)

        # for all source cells store the length of the vector:
        # (connection centroid - preferred direction)
        diff_conn_centroid_x_vsrc = np.zeros(n_src)
        diff_conn_centroid_v_vsrc = np.zeros(n_src)
        angles_x = np.zeros(n_src)
        angles_v = np.zeros(n_src)
        # deprecated
#        cx_ = np.zeros(n_src) # stores the scalar products
#        cv_ = np.zeros(n_src) # stores the scalar products
        for i in xrange(n_src):
            src_gid = i
            targets = utils.get_targets(conn_list, src_gid)
            weights = targets[:, 2]
            targets = np.array(targets[:, 1], dtype=np.int)
#            weights = w[src_gid, targets]
            if weights.size > 0:
                c_x, c_v = self.get_cg_vec(tp_src[src_gid, :], tp_tgt[targets, :], weights)
            else:
                c_x, c_v = [(0, 0), (0, 0)]

            (x_src, y_src, vx_src, vy_src) = tp_src[src_gid, :]
#            cx_[i] = np.abs(np.dot(c_x, (vx_src, vy_src)))
#            cv_[i] = np.abs(np.dot(c_v, (vx_src, vy_src)))

            vector_conn_centroid_x_minus_vsrc = (c_x[0] - vx_src, c_x[1] - vy_src)
            vector_conn_centroid_v_minus_vsrc = (c_v[0] - vx_src, c_v[1] - vy_src)
#            angles_x[i] = np.arc((c_x[
            diff_conn_centroid_x_vsrc[i] = np.sqrt(np.dot(vector_conn_centroid_x_minus_vsrc, vector_conn_centroid_x_minus_vsrc))
            diff_conn_centroid_v_vsrc[i] = np.sqrt(np.dot(vector_conn_centroid_v_minus_vsrc, vector_conn_centroid_v_minus_vsrc))


        print 'diff_conn_centroid_x_vsrc mean %.2e +- %.2e' % (diff_conn_centroid_x_vsrc.mean(), diff_conn_centroid_x_vsrc.std())
        print 'diff_conn_centroid_v_vsrc mean %.2e +- %.2e' % (diff_conn_centroid_v_vsrc.mean(), diff_conn_centroid_v_vsrc.std())
#        cx_mean = cx_.mean()
#        cx_sem = cx_.std() / np.sqrt(cx_.size)
#        cv_mean = cv_.mean()
#        cv_sem = cv_.std() / np.sqrt(cv_.size)
        cx_mean = diff_conn_centroid_x_vsrc.mean()
        cx_sem = diff_conn_centroid_x_vsrc.std() / np.sqrt(n_src)
        cv_mean = diff_conn_centroid_v_vsrc.mean()
        cv_sem = diff_conn_centroid_v_vsrc.std() / np.sqrt(n_src)

        output_fn = self.params['data_folder'] + 'mean_length_of_vector_diff_tuning_prop_minus_cgxv.dat'
        output_data = np.array((diff_conn_centroid_x_vsrc, diff_conn_centroid_v_vsrc)).transpose()
#        output_data = np.array((diff_conn_centroid_x_vsrc, diff_conn_centroid_v_vsrc, cx_, cv_)).transpose()
        print 'Saving to:', output_fn
        np.savetxt(output_fn, output_data)

        fig = pylab.figure(figsize=(12, 10))
        pylab.subplots_adjust(hspace=0.35)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        x = range(n_src)
        ax1.set_xlabel('source cell')
        ax1.set_ylabel('$|\\vec{v}_i - \\vec{c}_i^X|$')
        title = '$\langle|\\vec{v}_i - \\vec{c}_i^X| \\rangle = %.2e \pm %.1e$' % (cx_mean, cx_sem)
        ax1.bar(x, diff_conn_centroid_x_vsrc)
        ax1.set_title('Length of difference vector: preferred direction $\\vec{v}_i$ and connection centroid $\\vec{c}_i^x$\n%s' % title)
        ax1.set_xlim((0, n_src))
#        ax1.legend()

        ax2.bar(x, diff_conn_centroid_v_vsrc)
        ax2.set_xlabel('source cell')
        ax1.set_ylabel('$|\\vec{v}_i - \\vec{c}_i^V|$')
        title = '$\langle|\\vec{v}_i - \\vec{c}_i^V| \\rangle = %.2e \pm %.1e$' % (cv_mean, cv_sem)
        ax2.set_title(title)
        ax2.set_xlim((0, n_src))
#        ax2.legend()
        output_fig = self.params['figures_folder'] + 'mean_length_of_vector_diff_tuning_prop_minus_cgxv.png'
        print 'Saving to:', output_fig
        pylab.savefig(output_fig)
        if show:
            pylab.show()



    def get_cg_vec(self, tp_src, tp_tgt, weights):
        """
        Computes the center of gravity connection vector in the spatial and direction domain
        for one source cell and all its targets.
        c_x_i = sum_j w_ij * (x_i - x_j) # x_ are position vectors of the cell
        c_v_i = sum_j w_ij * (v_i - v_j) # v_ are preferred directions

        tp_src = 4-tuple of the source's tuning properties
        tp_tgt = 4 x n_tgt array with all the target's tuning properties
        """

        c_x = np.zeros(2)
        c_v = np.zeros(2)
        weights /= weights.max()
        (x_src, y_src, vx_src, vy_src) = tp_src

        n_tgt = tp_tgt[:, 0].size
        for tgt in xrange(n_tgt):
            (x_tgt, y_tgt, vx_tgt, vy_tgt) = tp_tgt[tgt, :]
            c_x += weights[tgt] * np.array( (x_tgt - x_src) % 1., (y_tgt - y_src) % 1.)
            c_v += weights[tgt] * np.array(vx_tgt - vx_src, vy_tgt - vy_src)

        c_x /= n_tgt
        c_v /= n_tgt
#        c_x *= self.params['connectivity_radius']
#        c_v *= self.params['connectivity_radius']
        return c_x, c_v
#        n_tgt =


    def create_connectivity(self, conn_type):
        """
        This function (re-) creates the network connectivity.
        """

        # distribute the cell ids among involved processes
        (n_src, n_tgt, self.tp_src, self.tp_tgt) = utils.resolve_src_tgt_with_tp(conn_type, self.params)

        print 'Connect anisotropic %s - %s' % (conn_type[0].capitalize(), conn_type[1].capitalize())

        gid_tgt_min, gid_tgt_max = utils.distribute_n(n_tgt, self.n_proc, self.pc_id)
        print 'Process %d deals with target GIDS %d - %d' % (self.pc_id, gid_tgt_min, gid_tgt_max)
        gid_src_min, gid_src_max = utils.distribute_n(n_src, self.n_proc, self.pc_id)
        print 'Process %d deals with source GIDS %d - %d' % (self.pc_id, gid_src_min, gid_src_max)
        n_my_tgts = gid_tgt_max - gid_tgt_min

        # data structure for connection storage
        self.target_adj_list = [ [] for i in xrange(n_my_tgts)]

        n_src_cells_per_neuron = int(round(self.params['p_%s' % conn_type] * n_src))

        # compute all pairwise connection probabilities
        for i_, tgt in enumerate(range(gid_tgt_min, gid_tgt_max)):
            if (i_ % 20) == 0:
                print '%.2f percent complete' % (i_ / float(n_my_tgts) * 100.)
            p = np.zeros(n_src)
            latency = np.zeros(n_src)
            for src in xrange(n_src):
                if conn_type[0] == conn_type[1]: # no self-connection
                    if (src != tgt):
                        p[src], latency[src] = CC.get_p_conn(self.tp_src[src, :], self.tp_tgt[tgt, :], params['w_sigma_x'], params['w_sigma_v'], params['connectivity_radius'])
                else: # different populations --> same indices mean different cells, no check for src != tgt
                    p[src], latency[src] = CC.get_p_conn(self.tp_src[src, :], self.tp_tgt[tgt, :], params['w_sigma_x'], params['w_sigma_v'], params['connectivity_radius'])
            # sort connection probabilities and select remaining connections
            sorted_indices = np.argsort(p)
            if conn_type[0] == 'e':
                sources = sorted_indices[-n_src_cells_per_neuron:]
            else:
                if conn_type == 'ii':
                    sources = sorted_indices[1:n_src_cells_per_neuron+1]  # shift indices to avoid self-connection, because p_ii = .0
                else:
                    sources = sorted_indices[:n_src_cells_per_neuron]
            w = (self.params['w_tgt_in_per_cell_%s' % conn_type] / p[sources].sum()) * p[sources]
            for i in xrange(len(sources)):
                if w[i] > self.params['w_thresh_connection']:
                    delay = min(max(latency[sources[i]] * self.params['delay_scale'], self.params['delay_range'][0]), self.params['delay_range'][1])  # map the delay into the valid range
                    # create adjacency list for all local cells and store connection in class container
                    self.target_adj_list[i_].append(sources[i])


        # communicate the resulting target_adj_list to the root process
        self.send_list_to_root(self.target_adj_list)


    def send_list_to_root(self, list_to_be_sent):
        pass
        #


    def plot_src_tgt_position_scatter(self, conn_type):
        pass

#        for i_

    def print_well_tuned_cell_connectivity(self):
        """
        This function prints the sources and targets for the 'well-tuned' cells,
        and prints additional information, like
        cos(x_tgt - x_src, v_tgt)
        cos(x_tgt - x_src, v_tgt) / sigma_x**2
        etc...
        """
        gids = np.loadtxt(self.params['gids_to_record_fn'], dtype=int)



if __name__ == '__main__':


    # IMPORT MPI
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


    conn_types = ['ee', 'ei', 'ie', 'ii']

    # CHECK IF PARAMETER FILE WAS PASSED
    conn_type = None
    if len(sys.argv) > 1:
        if len(sys.argv[1]) == 2:
            conn_type = sys.argv[1]
            assert (conn_type in conn_types), 'Non-existant conn_type %s' % conn_type
            try:
                param_fn = sys.argv[2]
            except:
                network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
                params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
        else:
            param_fn = sys.argv[1]
            if os.path.isdir(param_fn):
                param_fn += '/Parameters/simulation_parameters.info'
            print 'Trying to load parameters from', param_fn
            import NeuroTools.parameters as NTP
            params = NTP.ParameterSet(utils.convert_to_url(param_fn))
    else:
        print '\nLoading the parameters currently in simulation_parameters.py\n'
        network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
        params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

    # get the connection type either from sys.argv[1] or [2]
    if conn_type == None:
        conn_types = ['ee', 'ei', 'ie', 'ii']
    else:
        conn_types = [conn_type]
    print 'Processing conn_types', conn_types
    CA = ConnectivityAnalyser(params, comm)


    def plot_outgoing_connections(conn_type):
        CA.load_tuning_prop()
        CA.n_fig_x = 1
        CA.n_fig_y = 3
        CA.create_fig()
#        CA.plot_tgt_connections(conn_type, fig_cnt=1)
        CA.plot_num_outgoing_connections(conn_type, fig_cnt=2)

    for conn_type in conn_types:
        plot_outgoing_connections(conn_type)
        output_fn = params['figures_folder'] + 'connectivity_analysis_%s.png' % conn_type
        print 'Saving to', output_fn
        pylab.savefig(output_fn)

#    pylab.show()


#    CA.create_connectivity(conn_type)
#    CA.plot_src_tgt_position_scatter(conn_type)


#    CA.plot_tuning_vs_conn_cg(conn_type, show=False)





