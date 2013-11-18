import numpy as np
import os
import simulation_parameters
import NeuroTools.parameters as NTP
import pylab
import utils
import json

try:
    from mpi4py import MPI
    USE_MPI = True
    comm = MPI.COMM_WORLD
    pc_id, n_proc = comm.rank, comm.size

except:
    USE_MPI = False
    comm = None
    pc_id, n_proc = 0, 1


class ResultsCollector(object):

    def __init__(self, params):
        self.params = params
        self.param_space = {}
        self.dirs_to_process = []

        self.n_fig_x = 1
        self.n_fig_y = 2
#        self.fig_size = (11.69, 8.27) #A4
        self.fig_size = (14, 10)

    def create_fig(self, fig_size=None):
        print "plotting ...."
        if fig_size == None:
            fig_size = self.fig_size
        self.fig = pylab.figure(figsize=fig_size)

    def set_dirs_to_process(self, list_of_dir_names):

        for i_, folder in enumerate(list_of_dir_names):
            self.dirs_to_process.append(folder)
            self.param_space[i_] = {}


    def collect_files(self):
        all_dirs = []
        for f in os.listdir('.'):
            if os.path.isdir(f):
                all_dirs.append(f)

        self.mandatory_files = ['Spikes/exc_spikes_merged_.ras', \
                'Parameters/simulation_parameters.info', \
                'Parameters/tuning_prop_means.prm']

        sim_id = 0
        for dir_name in all_dirs:
            # check if all necessary files exist
            check_passed = True
            for fn in self.mandatory_files:
                fn_ = dir_name + '/' + fn
#                print 'checking', fn_
                if not os.path.exists(fn_):
                    check_passed = False
            if check_passed:
                self.dirs_to_process.append((dir_name, sim_id, {}))
                sim_id += 1

    def get_xvdiff_integral(self, t_range=None):
        """
        t_range is the limit for the integral
        """
        if t_range != None:
            assert (len(t_range) == 2), 't_range has wrong length! Please give a tuple of len 2'
            assert (t_range[1] > t_range[0]), 'Wrong order of integral limits'

        self.xdiff_integral = np.zeros(len(self.dirs_to_process))
        self.vdiff_integral = np.zeros(len(self.dirs_to_process))
        results_sub_folder = 'Data/'
        fn_base_x = self.params['xdiff_vs_time_fn']
        fn_base_v = self.params['vdiff_vs_time_fn']

        time_bin_size = np.zeros(len(self.dirs_to_process))
        for i_, folder in enumerate(self.dirs_to_process):
            # if self.dirs_to_process has been created by collect_files()
#            fn_x = folder[0] + '/' + results_sub_folder + fn_base_x
#            fn_v = folder[0] + '/' + results_sub_folder + fn_base_v
            fn_x = folder + '/' + results_sub_folder + fn_base_x
            fn_v = folder + '/' + results_sub_folder + fn_base_v
            xdiff = np.loadtxt(fn_x)
            vdiff = np.loadtxt(fn_v)
            n_bins = xdiff[:, 0].size
            assert n_bins == vdiff[:, 0].size, "ERROR in x/v diff integrals!\n%s and %s have different sizes!" % (fn_x, fn_v)
            time_binsize = xdiff[1, 0] - xdiff[0, 0]
            time_bin_size[i_] = time_binsize
            if t_range == None:
                self.xdiff_integral[i_] = xdiff[:, 1].sum()
                self.vdiff_integral[i_] = vdiff[:, 1].sum()
                self.xdiff_integral[i_] = np.sqrt((xdiff[:, 1]**2).sum() / n_bins)
                self.vdiff_integral[i_] = np.sqrt((vdiff[:, 1]**2).sum() / n_bins)
            else:
                idx_0 = (xdiff[:, 0] == t_range[0]).nonzero()[0][0]
                print xdiff[:, 0].max()
                print 'debug', t_range[1] - time_binsize, fn_x
                idx_1 = (xdiff[:, 0] == t_range[1] - time_binsize).nonzero()[0][0]

                self.xdiff_integral[i_] = np.sqrt((xdiff[idx_0:idx_1, 1]**2).sum() / (idx_1 - idx_0))
                self.vdiff_integral[i_] = np.sqrt((vdiff[idx_0:idx_1, 1]**2).sum() / (idx_1 - idx_0))
#                self.vdiff_integral[i_] = vdiff[idx_0:idx_1, 1].sum()

            print 'folder, self.xdiff_integral[i_], self.vdiff_integral[i_]'
            print time_binsize, folder, self.xdiff_integral[i_], self.vdiff_integral[i_]
#            print fn_x, 
        print 'All folders were analysed with the same time bin size:\n', (time_bin_size == time_bin_size.mean()).all()
#        print time_bin_size, time_bin_size.mean()
        output_data = np.array((np.zeros(self.xdiff_integral.size), self.xdiff_integral, self.vdiff_integral))
        self.output_data = output_data.transpose()


    def save_output_data(self, output_fn):
        print 'Saving xdiff and vdiff integral to:', output_fn
        idx = self.output_data.argsort(0)
#        print 'idx', idx
#        print 'data[idx]', self.output_data[idx[:, 0]]
        np.savetxt(output_fn, self.output_data[idx[:, 0]])


    def get_parameter(self, param_name):
        """
        For all simulations (in self.dirs_to_process) get the according parameter value
        """
        for i_, folder in enumerate(self.dirs_to_process):
            param_fn = folder + '/Parameters/simulation_parameters.json'
            f = file(param_fn, 'r')
            param_dict = json.load(f)
            value = param_dict[param_name]
            self.param_space[i_][param_name] = value
        return self.param_space    

    def plot_param_vs_xvdiff_integral(self, param_name, xv='x', t_integral=None, fig_cnt=1):

        if t_integral == None:
            t0 = '0'
            t1 = 't_{sim}'
        else:
            t0 = 't=' + str(t_integral[0])
            t1 = 't=' + str(t_integral[1])

        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        if xv == 'x':
            xvdiff_integral = self.xdiff_integral
            title = '$\int_{%s}^{%s} |\\vec{x}_{stim}(t) - \\vec{x}_{prediction}(t)| dt$ vs. %s' % (t0, t1, param_name)
        else:
            xvdiff_integral = self.vdiff_integral
            title = '$\int_{%s}^{%s} |\\vec{v}_{stim}(t) - \\vec{v}_{prediction}(t)| dt$ vs. %s' % (t0, t1, param_name)

        x_data = np.zeros(len(self.dirs_to_process))
        y_data = xvdiff_integral

        for i_, folder in enumerate(self.dirs_to_process):
            param_value = self.param_space[i_][param_name]
            x_data[i_] = param_value

        print ' Data %s - prediction:\n' % (xv), x_data, '\n', y_data
        ax.plot(x_data, y_data, 'o')
        ax.set_xlim((x_data.min() * .9, x_data.max() * 1.1))
        ax.set_ylim((y_data.min() * .9, y_data.max() * 1.1))
        ax.set_xlabel(param_name, fontsize=18)
        ax.set_ylabel('Integral %s' % xv)
        ax.set_title(title)
        self.output_data[:, 0] = x_data
#        pylab.show()
            


    def get_cgxv(self):

        results_sub_folder = 'Data/'
        fn_base = 'scalar_products_between_tuning_prop_and_cgxv.dat'
        self.cgx = np.zeros((len(self.dirs_to_process), 2))
        self.cgv = np.zeros((len(self.dirs_to_process), 2))
        for i_, folder in enumerate(self.dirs_to_process):
            # if self.dirs_to_process has been created by collect_files()
#            fn = folder[0] + '/' + results_sub_folder + fn_base
            fn = folder + '/' + results_sub_folder + fn_base
            d = np.loadtxt(fn)
            self.cgx[i_, 0] = d[:, 0].mean()
            self.cgx[i_, 1] = d[:, 0].std() / np.sqrt(d[:, 0].size)

            self.cgv[i_, 0] = d[:, 1].mean()
            self.cgv[i_, 1] = d[:, 1].std() / np.sqrt(d[:, 1].size)


    def plot_cgxv_vs_xvdiff(self):

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        for i_, folder in enumerate(self.dirs_to_process):
#            y = self.cgx[i_, 0]
#            yerr = self.cgx[i_, 1]

            y = self.cgv[i_, 0]
            yerr = self.cgv[i_, 1]
            x = self.xdiff_integral[i_]
#            x = self.vdiff_integral[i_]
            
            print 'debug ', x, y, folder
            ax.errorbar(x, y, yerr=yerr, ls='o', c='b')
            ax.plot(x, y, 'o', c='b')

        pylab.show() 



    def build_parameter_space(self):

        # take a sample simulation_parameters.info file to generate all possible keys
        sample_fn = self.dirs_to_process[0][0] + '/Parameters/simulation_parameters.info'
        sample_dict = NTP.ParameterSet(sample_fn)
        all_param_names = sample_dict.keys()



    def check_for_correctness(self):
        """
        This function checks if the folder name has the same value for a 
        given parameter (e.g. 'wee') as in the simulation_parameters.info file
        """
        for dirname, sim_id in self.dirs_to_process:
            idx = dirname.find('wee')
            idx2 = dirname[idx:].find('_')
            val_in_folder = float(dirname[idx+3:idx+idx2])

            fn = dirname + '/Parameters/simulation_parameters.info'
            param_dict = NTP.ParameterSet(fn)
            if param_dict['w_tgt_in_per_cell_ee'] != val_in_folder:
                print 'Mismatch in folder name and parameter dict:', dirname



if __name__ == '__main__':
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.params
    RC = ResultsCollector(params)
    RC.collect_files()
    RC.get_xvdiff_integral() # RMSE
    RC.get_cgxv()

#print "RC.dirs_to_process", RC.dirs_to_process
#RC.check_for_correctness()

