import simulation_parameters
import numpy as np
import utils
import Bcpnn
import os
import sys
import time
import random

class AbstractTrainer(object):

    def __init__(self, params, n_speeds, n_cycles, n_stim, comm=None):
        self.params = params
        self.comm = comm
        self.n_stim = n_stim
        self.n_cycles = n_cycles
        self.n_speeds = n_speeds
        self.selected_conns = None

        # distribute units among processors
        if comm != None:
            self.pc_id, self.n_proc = comm.rank, comm.size
            my_units = utils.distribute_n(params['n_exc'], self.n_proc, self.pc_id)
            self.my_units = range(my_units[0], my_units[1])
        else:
            self.my_units = range(self.params['n_exc'])
            self.pc_id, self.n_proc = 0, 1

        try:
            self.tuning_prop = np.loadtxt(self.params['tuning_prop_means_fn'])
        except:
            print 'Tuning properties file not found: %s\n Will create new ones' % self.params['tuning_prop_means_fn']
            self.tuning_prop = utils.set_tuning_prop(self.params, mode='hexgrid', v_max=self.params['v_max'])
            np.savetxt(self.params['tuning_prop_means_fn'], self.tuning_prop)

        self.initial_value = 1e-5
        self.tau_dict = {'tau_zi' : 50.,    'tau_zj' : 5., 
                        'tau_ei' : 50.,   'tau_ej' : 50., 'tau_eij' : 50.,
                        'tau_pi' : 2400.,  'tau_pj' : 2400., 'tau_pij' : 2400.,
                        # tau_p should be in the order of t_stimulus * n_iterations
                        }
        self.eps = .1 * self.initial_value
        self.normalize = True # normalize input within a 'hypercolumn'

        all_conns = []
        # distribute connections among processors
        for i in xrange(params['n_exc']):
            for j in xrange(params['n_exc']):
                if i != j:
                    all_conns.append((i, j))
        self.my_conns = utils.distribute_list(all_conns, n_proc, pc_id)

 
    def create_stimuli(self, random_order=False):

        distance_from_center = 0.5
        center = (0.5, 0.5)
        thetas = np.linspace(np.pi, 3*np.pi, n_stim, endpoint=False)
#        r = 0.5 # how far the stimulus will move
        v_default = np.sqrt(self.params['motion_params'][2]**2 + self.params['motion_params'][3]**2)

        seed = 0
        np.random.seed(seed)
        random.seed(seed)

        sigma_theta = 2 * np.pi * 0.05
        random_rotation = sigma_theta * (np.random.rand(self.n_cycles * self.n_stim * self.n_speeds) - .5 * np.ones(self.n_cycles * self.n_stim * self.n_speeds))
        v_min, v_max = 0.2, 0.4
        speeds = np.linspace(v_min, v_max, self.n_speeds)

        iteration = 0

        for speed_cycle in xrange(self.n_speeds):
            for cycle in xrange(self.n_cycles):
                stimulus_order = range(self.n_stim)
                if random_order == True:
                    random.shuffle(stimulus_order)

                for stim in stimulus_order:
                    x0 = distance_from_center * np.cos(thetas[stim] + random_rotation[iteration]) + center[0]
                    y0 = distance_from_center * np.sin(thetas[stim] + random_rotation[iteration]) + center[1]
                    u0 = np.cos(np.pi + thetas[stim] + random_rotation[iteration]) * speeds[speed_cycle]#v_default
                    v0 = np.sin(np.pi + thetas[stim] + random_rotation[iteration]) * speeds[speed_cycle]#v_default
                    self.params['motion_params'] = (x0, y0, u0, v0)
#                    print 'Motion params', x0, y0, u0, v0
                    x_stop = x0 + u0 * self.params['t_stimulus'] / params['t_sim']
                    y_stop = y0 + v0 * self.params['t_stimulus'] / params['t_sim']
                    input_str = '#x0\ty0\tu0\tv0\n'
                    input_str += '%.4e\t%.4e\t%.4e\t%.4e' % (x0, y0, u0, v0)

        #            input_str = 'x0=%.4f\ny0 %.4f\nu0 %.4f\nv0 %.4f\nv=%.4f\nx_stop %.4f\ny_stop %.4f' % (x0, y0, u0, v0, np.sqrt(u0**2 + v0**2), x_stop, y_stop)

                    self.training_input_folder = "%sTrainingInput_%d/" % (self.params['folder_name'], iteration)
                    print 'Writing input to %s' % (self.training_input_folder)
                    print input_str
                    if not os.path.exists(self.training_input_folder) and self.pc_id == 0:
                        mkdir = 'mkdir %s' % self.training_input_folder
                        print mkdir
                        os.system(mkdir)
                    if self.comm != None:
                        self.comm.barrier()

                    output_file = open(self.training_input_folder + 'input_params.txt', 'w')
                    output_file.write(input_str)
                    output_file.close()
                    self.create_input_vectors(normalize=self.normalize)
                    if self.comm != None:
                        self.comm.barrier()
                    iteration += 1




    def create_input_vectors(self, normalize=True):
        output_fn_base = self.training_input_folder + self.params['abstract_input_fn_base']
        n_cells = len(self.my_units)
        dt = self.params['dt_rate'] # [ms] time step for the non-homogenous Poisson process 
        time = np.arange(0, params['t_stimulus'], dt)
        L_input = np.empty((n_cells, time.shape[0]))
        for i_time, time_ in enumerate(time):
            if (i_time % 100 == 0):
                print "t:", time_
            L_input[:, i_time] = utils.get_input(self.tuning_prop[self.my_units, :], params, time_/params['t_sim'])

        for i_, unit in enumerate(self.my_units):
            output_fn = output_fn_base + str(unit) + '.dat'
            np.savetxt(output_fn, L_input[i_, :])

        if self.comm != None:
            self.comm.barrier()

        if normalize:
            self.normalize_input(output_fn_base)

        if self.comm != None:
            self.comm.barrier()


    def normalize_input(self, fn_base):

        if pc_id == 0:
            print 'normalize_input'
            dt = self.params['dt_rate'] # [ms] time step for the non-homogenous Poisson process 
            L_input = np.empty((self.params['n_exc'], self.params['t_stimulus']/dt))

            v_max = self.params['v_max']
            if self.params['log_scale']==1:
                v_rho = np.linspace(v_max/self.params['N_V'], v_max, num=self.params['N_V'], endpoint=True)
            else:
                v_rho = np.logspace(np.log(v_max/self.params['N_V'])/np.log(self.params['log_scale']),
                                np.log(v_max)/np.log(self.params['log_scale']), num=self.params['N_V'],
                                endpoint=True, base=self.params['log_scale'])
            v_theta = np.linspace(0, 2*np.pi, self.params['N_theta'], endpoint=False)
            index = 0
            for i_RF in xrange(self.params['N_RF_X']*self.params['N_RF_Y']):
                index_start = index
                for i_v_rho, rho in enumerate(v_rho):
                    for i_theta, theta in enumerate(v_theta):
                        fn = fn_base + str(index) + '.dat'
                        L_input[index, :] = np.loadtxt(fn)
                        index += 1
                index_stop = index
                if (L_input[index_start:index_stop, :].sum() > 1):
                    L_input[index_start:index_stop, :] /= L_input[index_start:index_stop, :].sum()

            for cell in xrange(self.params['n_exc']):
                output_fn = fn_base + str(cell) + '.dat'
                np.savetxt(output_fn, L_input[cell, :])

        if self.comm != None:
            self.comm.barrier()




    def train(self):

        pi_init = self.initial_value * np.ones(params['n_exc'])
        pj_init = self.initial_value * np.ones(params['n_exc'])
        pij_init = self.initial_value ** 2 * np.ones((params['n_exc'], params['n_exc']))
        wij_init = np.zeros((params['n_exc'], params['n_exc']))
        bias_init = np.log(self.initial_value) * np.ones(params['n_exc'])

        comp_times = []
        iteration = 0
        for speed in xrange(self.n_speeds):
            for cycle in xrange(self.n_cycles):
                print '\nCYCLE %d\n' % (cycle)
                for stim in xrange(self.n_stim):
                    t0= time.time()
                    # M A K E    D I R E C T O R Y 
                    training_folder = '%sTrainingResults_%d/' % (self.params['folder_name'], iteration)
                    if not os.path.exists(training_folder) and self.pc_id == 0:
                        mkdir = 'mkdir %s' % training_folder
                        print mkdir
                        os.system(mkdir)
                    if self.comm != None:
                        self.comm.barrier()

                    # C O M P U T E    
                    self.compute_my_pijs(training_folder, pi_init, pj_init, pij_init, wij_init, bias_init, iteration)

                    # U P D A T E
                    all_pi = np.loadtxt(training_folder + 'all_pi_%d.dat' % iteration)
                    all_pj = np.loadtxt(training_folder + 'all_pj_%d.dat' % iteration)
                    all_pij = np.loadtxt(training_folder + 'all_pij_%d.dat' % iteration)
                    all_wij = np.loadtxt(training_folder + 'all_wij_%d.dat' % iteration)
                    for cell in xrange(params['n_exc']):
                        gid, pi = all_pi[cell, :]
                        pi_init[gid] = pi

                        gid, pj = all_pj[cell, :]
                        pj_init[gid] = pj

                    for c in xrange(all_pij[:, 0].size):
                        pre_id, post_id, pij = all_pij[c, :]
                        pij_init[pre_id, post_id] = pij

                    for c in xrange(all_wij[:, 0].size):
                        pre_id, post_id, pij, wij, bias = all_wij[c, :]
                        wij_init[pre_id, post_id] = wij
                        bias_init[post_id] = bias

                    t_comp = time.time() - t0
                    comp_times.append(t_comp)
                    print 'Computation time for training %d: %d sec = %.1f min' % (iteration, t_comp, t_comp / 60.)
                    if self.comm != None:
                        self.comm.barrier()
                    iteration += 1

        total_time = 0.
        for t in comp_times:
            total_time += t
        print 'Total computation time for %d training iterations: %d sec = %.1f min' % (self.n_stim * self.n_cycles, total_time, total_time/ 60.)


    def compute_my_pijs(self, training_folder, pi_init, pj_init, pij_init, wij_init, bias_init, iteration=0):
        conns = self.my_conns

        tau_dict = self.tau_dict
        dt = 1
        print 'pc_id computes pijs for %d connections' % (len(conns))
        my_traces_pre = {}
        my_traces_post = {}
        p_i = {}
        p_j = {}

        p_i_string = '#GID\tp_i\n'
        p_j_string = '#GID\tp_j\n'
        p_ij_string = '#pre_id\tpost_id\tp_ij\n'
        w_ij_string = '#pre_id\tpost_id\tpij[-1]\tw_ij[-1]\tbias\n'
        z_init = self.initial_value
        e_init = self.initial_value

        self.training_input_folder = "%sTrainingInput_%d/" % (self.params['folder_name'], iteration)
        input_fn_base = self.training_input_folder + self.params['abstract_input_fn_base']

        if pi_init == None:
            pi_init = np.ones(self.params['n_exc']) * self.initial_value
        if pj_init == None:
            pj_init = np.ones(self.params['n_exc']) * self.initial_value
        if pij_init == None:
            pij_init = self.initial_value ** 2 * np.ones((params['n_exc'], params['n_exc']))
        if wij_init == None:
            wij_init = np.zeros((params['n_exc'], params['n_exc']))
        if bias_init == None:
            bias_init = np.log(self.initial_value) * np.ones(params['n_exc'])

        eij_init = self.initial_value ** 2 * np.ones((params['n_exc'], params['n_exc']))
#        bias_init= np.log(self.initial_value) * np.ones(params['n_exc'])
#        wij_init = np.zeros((params['n_exc'], params['n_exc']))

        for i in xrange(len(conns)):
            if (i % 500) == 0:
                print "Pc %d conn: \t%d - %d; \t%d / %d\t%.4f percent complete" % (pc_id, conns[i][0], conns[i][1], i, len(conns), i * 100./len(conns))
            pre_id = conns[i][0]
            post_id = conns[i][1]
            if my_traces_pre.has_key(pre_id):
                (zi, ei, pi) = my_traces_pre[pre_id]
            else:
                pre_trace = np.loadtxt(input_fn_base + str(pre_id) + '.dat')
                zi, ei, pi = Bcpnn.compute_traces(pre_trace, tau_dict['tau_zi'], tau_dict['tau_ei'], tau_dict['tau_pi'], eps=self.eps, initial_value=(z_init, e_init, pi_init[pre_id]))
                my_traces_pre[pre_id] = (zi, ei, pi)
                p_i[pre_id] = pi[-1]

            if my_traces_post.has_key(post_id):
                (zj, ej, pj) = my_traces_post[post_id]
            else: 
                post_trace = np.loadtxt(input_fn_base  + str(post_id) + '.dat')
                zj, ej, pj = Bcpnn.compute_traces(post_trace, tau_dict['tau_zj'], tau_dict['tau_ej'], tau_dict['tau_pj'], eps=self.eps, initial_value=(z_init, e_init, pj_init[post_id]))
                my_traces_post[post_id] = (zj, ej, pj)
                p_j[post_id] = pj[-1]

            pij, w_ij, bias = Bcpnn.compute_pij(zi, zj, pi, pj, tau_dict['tau_eij'], tau_dict['tau_pij'], \
                    initial_values=(eij_init[pre_id, post_id], pij_init[pre_id, post_id], wij_init[pre_id, post_id], bias_init[post_id]))
            w_ij_string += '%d\t%d\t%.6e\t%.6e\t%.6e\n' % (pre_id, post_id, pij, w_ij, bias)
            p_ij_string += '%d\t%d\t%.6e\n' % (pre_id, post_id, pij)
        
        for pre_id in p_i.keys():
            p_i_string += '%d\t%.6e\n' % (pre_id, p_i[pre_id])

        for post_id in p_j.keys():
            p_j_string += '%d\t%.6e\n' % (post_id, p_j[post_id])


        # write selected traces to files
        for c in self.selected_conns:
            if c in self.my_conns:
                pre_id, post_id = c[0], c[1]

#                if my_traces_pre.has_key(pre_id):
                fn_tmp = self.params['bcpnntrace_folder'] + 'zi_%d_%d.dat' % (iteration, pre_id)
                print 'Proc %d prints BCPNN pre-traces for cell %d: %s' % (self.pc_id, pre_id, fn_tmp)
                np.savetxt(self.params['bcpnntrace_folder'] + 'zi_%d_%d.dat' % (iteration, pre_id), my_traces_pre[pre_id][0])
                np.savetxt(self.params['bcpnntrace_folder'] + 'ei_%d_%d.dat' % (iteration, pre_id), my_traces_pre[pre_id][1])
                np.savetxt(self.params['bcpnntrace_folder'] + 'pi_%d_%d.dat' % (iteration, pre_id), my_traces_pre[pre_id][2])

#                if my_traces_post.has_key(post_id):
                print 'Proc %d prints BCPNN post-traces for cell %d' % (self.pc_id, post_id)
                np.savetxt(self.params['bcpnntrace_folder'] + 'zj_%d_%d.dat' % (iteration, post_id), my_traces_post[post_id][0])
                np.savetxt(self.params['bcpnntrace_folder'] + 'ej_%d_%d.dat' % (iteration, post_id), my_traces_post[post_id][1])
                np.savetxt(self.params['bcpnntrace_folder'] + 'pj_%d_%d.dat' % (iteration, post_id), my_traces_post[post_id][2])

        if self.comm != None:
            self.comm.barrier()

        # for selected connections compute the eij, pij, weight and bias traces
        if self.pc_id == 0:
            for c in self.selected_conns:
                pre_id, post_id = c[0], c[1]
                zi = np.loadtxt(self.params['bcpnntrace_folder'] + 'zi_%d_%d.dat' % (iteration, pre_id))
                pi = np.loadtxt(self.params['bcpnntrace_folder'] + 'pi_%d_%d.dat' % (iteration, pre_id))
                zj = np.loadtxt(self.params['bcpnntrace_folder'] + 'zj_%d_%d.dat' % (iteration, post_id))
                pj = np.loadtxt(self.params['bcpnntrace_folder'] + 'pj_%d_%d.dat' % (iteration, post_id))
                wij, bias, pij, eij = Bcpnn.compute_pij(zi, zj, pi, pj, self.tau_dict['tau_eij'], self.tau_dict['tau_pij'], get_traces=True, 
                    initial_values=(eij_init[pre_id, post_id], pij_init[pre_id, post_id], wij_init[pre_id, post_id], bias_init[post_id]))
                np.savetxt(self.params['bcpnntrace_folder'] + 'wij_%d_%d_%d.dat' % (iteration, pre_id, post_id), wij)
                np.savetxt(self.params['bcpnntrace_folder'] + 'bias_%d_%d_%d.dat' % (iteration, pre_id, post_id), bias)
                np.savetxt(self.params['bcpnntrace_folder'] + 'eij_%d_%d_%d.dat' % (iteration, pre_id, post_id), eij)
                np.savetxt(self.params['bcpnntrace_folder'] + 'pij_%d_%d_%d.dat' % (iteration, pre_id, post_id), pij)


        pi_output_fn = training_folder + 'pi_%d.dat' % (self.pc_id)
        pi_f = open(pi_output_fn, 'w')
        pi_f.write(p_i_string)
        pi_f.close()

        pj_output_fn = training_folder + 'pj_%d.dat' % (self.pc_id)
        pj_f = open(pj_output_fn, 'w')
        pj_f.write(p_j_string)
        pj_f.close()

        pij_output_fn = training_folder + 'pij_%d.dat' % (self.pc_id)
        pij_f = open(pij_output_fn, 'w')
        pij_f.write(p_ij_string)
        pij_f.close()


        w_ij_fn = training_folder + 'wij_%d.dat' % (self.pc_id)
        print 'Writing w_ij output to:', w_ij_fn
        f = file(w_ij_fn, 'w')
        f.write(w_ij_string)
        f.close()

        if self.comm != None:
            self.comm.barrier()

        if pc_id == 0:
            tmp_fn = training_folder + 'all_wij_%d.dat' % (iteration)
            cat_cmd = 'cat %s* > %s' % (training_folder + 'wij_', tmp_fn)
            print cat_cmd
            os.system(cat_cmd)

            tmp_fn = training_folder + 'all_pi_%d.dat' % (iteration)
            cat_cmd = 'cat %s* > %s' % (training_folder + 'pi_', tmp_fn)
            print cat_cmd
            os.system(cat_cmd)

            tmp_fn = training_folder + 'all_pj_%d.dat' % (iteration)
            cat_cmd = 'cat %s* > %s' % (training_folder + 'pj_', tmp_fn)
            print cat_cmd
            os.system(cat_cmd)

            tmp_fn = training_folder + 'all_pij_%d.dat' % (iteration)
            cat_cmd = 'cat %s* > %s' % (training_folder + 'pij_', tmp_fn)
            print cat_cmd
            os.system(cat_cmd)

        if self.comm != None:
            self.comm.barrier()
        



if __name__ == '__main__':


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


    PS = simulation_parameters.parameter_storage()
    params = PS.params
    PS.create_folders()
    PS.write_parameters_to_file()

    n_speeds = 2
    n_cycles = 5
    n_stim = 8

    AT = AbstractTrainer(params, n_speeds, n_cycles, n_stim, comm)

    cells_to_record = [85, 161, 111, 71, 339]
    selected_connections = []
    for src in cells_to_record:
        for tgt in cells_to_record:
            if src != tgt:
                selected_connections.append((src, tgt))
#    AT.selected_conns = [(85, 337)]
    AT.selected_conns = selected_connections
                        


    AT.create_stimuli(random_order=True)
    AT.train()



