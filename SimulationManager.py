import os
import simulation_parameters
import numpy as np
import utils
import time
import prepare_sim as Prep
import CreateConnections as CC
import NetworkSimModuleNoColumns as simulation
import NeuroTools.parameters as ntp
import Prepare


class SimulationManager(object):
    """
    Manages simulations
    """
    def __init__(self, parameter_storage, comm=None):
        if comm != None:
            self.pc_id, self.n_proc = comm.rank, comm.size
        else:
            self.pc_id, self.n_proc = 0, 1
        self.comm = comm 
        self.ParameterStorage = parameter_storage # the class around the parameters
        self.params = self.ParameterStorage.load_params()# the parameters in a dictionary
        self.sim_cnt = 0

    def update_values(self, new_dict):
        self.ParameterStorage.update_values(new_dict)
        self.params = self.ParameterStorage.load_params() # dictionary storing all the parameters

    def copy_folder(self, input_folder, tgt_folder):
        if self.pc_id == 0:
            print "cp -r %s %s" % (input_folder, tgt_folder)
            os.system("cp -r %s %s" % (input_folder, tgt_folder))
        if self.comm != None:
            self.comm.barrier()


    def create_folders(self):
        folders_exist = self.ParameterStorage.check_folders()
        if not folders_exist:
            if self.pc_id == 0:
                self.ParameterStorage.create_folders()
                self.ParameterStorage.write_parameters_to_file(self.params['params_fn'])# write parameters to a file
            if (self.comm != None):
                self.comm.barrier()

    def prepare_tuning_properties(self):

        Preparer = Prepare.Preparer(self.comm)
        Preparer.prepare_tuning_prop(self.params)
        del Preparer

    def prepare_spiketrains(self, tp):
        Preparer = Prepare.Preparer(self.comm)
        Preparer.prepare_spiketrains(self.params, tp)
        del Preparer



    def prepare_connections(self, input_fn=None):

        if self.pc_id == 0 and self.params['initial_connectivity'] == 'precomputed':
            print "Proc %d computes initial weights ... " % self.pc_id
            tuning_prop = np.loadtxt(self.params['tuning_prop_means_fn'])
            CC.compute_weights_from_tuning_prop(tuning_prop, self.params)

        elif self.pc_id == 0 and self.params['initial_connectivity'] == 'random':
            print "Proc %d shuffles pre-computed weights ... " % self.pc_id
            output_fn = self.params['random_weight_list_fn'] + '0.dat'
            CC.compute_random_weight_list(input_fn, output_fn, self.params)

        if self.comm != None:
            self.comm.barrier()


    def run_sim(self):

        if (self.pc_id == 0):
            print "Simulation run %d: %d cells (%d exc, %d inh)" % (self.sim_cnt+1, self.params['n_cells'], self.params['n_exc'], self.params['n_inh'])
            simulation.run_sim(self.params, self.sim_cnt, self.params['initial_connectivity'])

        else: 
            print "Pc %d waiting for proc 0 to finish simulation" % self.pc_id

        if self.comm != None: 
            self.comm.barrier()

