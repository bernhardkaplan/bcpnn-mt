# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 19:59:18 2013

@author: aliakbari-.m
"""

import sys
import os
import simulation_parameters
import numpy as np
import utils
import time
import CreateConnections as CC
import NetworkSimModuleNoColumns as simulation
import NeuroTools.parameters as ntp
import Prepare
from ParallelObject import PObject

class SimulationManager(PObject):
    """
    Manages simulations
    """
    def __init__(self, parameter_storage, comm=None):
        PObject.__init__(self, parameter_storage, comm)
        self.sim_cnt = 0
        self.cycle_cnt = 0

    def update_values(self, new_dict):
        self.ParameterStorage.update_values(new_dict)
        self.params = self.ParameterStorage.load_params() # dictionary storing all the parameters
        if self.comm != None:
            print 'Pid %d at Barrier in update values' % self.pc_id
            sys.stdout.flush()
            self.comm.Barrier()


    def copy_folder(self, input_folder, tgt_folder):
        if self.pc_id == 0:
            print "cp -r %s %s" % (input_folder, tgt_folder)
            os.system("cp -r %s %s" % (input_folder, tgt_folder))

    def create_folders(self):
#        folders_exist = self.ParameterStorage.check_folders()
#        if not folders_exist:
        if self.pc_id == 0:
            self.ParameterStorage.create_folders()
            self.ParameterStorage.write_parameters_to_file(self.params['params_fn'])# write parameters to a file

        if (self.comm != None):
            self.comm.Barrier()


    def prepare_tuning_properties(self):

        Preparer = Prepare.Preparer(self.ParameterStorage, self.comm)
        Preparer.prepare_tuning_prop()
        Preparer.sort_gids()
        del Preparer
        if self.comm != None:
            t1 = time.time()
            print 'Pid %d at Barrier in prepare_tuning_properties' % self.pc_id
            self.comm.Barrier()
#            sys.stdout.flush()
#            t2 = time.time()
#            dt = t2 - t1
#            print "Process %d waites for %.2f sec in prepare_tuning_properties in cycle %d" % (self.pc_id, dt, self.cycle_cnt)
        


    def prepare_spiketrains(self, tp):
        Preparer = Prepare.Preparer(self.ParameterStorage, self.comm)
        Preparer.prepare_spiketrains(tp)
        del Preparer
        if self.comm != None:
            print 'Pid %d at Barrier in prepare_spiketrains' % self.pc_id
            sys.stdout.flush()
            self.comm.Barrier()



    def prepare_connections(self, input_fn=None):

        if self.params['connectivity'] == 'precomputed':
            print "Proc %d computes initial weights ... " % self.pc_id
            tuning_prop = np.loadtxt(self.params['tuning_prop_means_fn'])
            CC.compute_weights_from_tuning_prop(tuning_prop, self.params, self.comm)

        elif self.pc_id == 0 and self.params['connectivity'] == 'random':
            print "Proc %d shuffles pre-computed weights ... " % self.pc_id
            output_fn = self.params['random_weight_list_fn'] + '0.dat'
            CC.compute_random_weight_list(input_fn, output_fn, self.params)

        if self.comm != None:
            print 'Pid %d at Barrier in prepare_connections' % self.pc_id
            sys.stdout.flush()
            self.comm.Barrier()


    def run_sim(self, connect_exc_exc=True):

        if (self.pc_id == 0):
            print "Simulation run %d: %d cells (%d exc, %d inh)" % (self.sim_cnt+1, self.params['n_cells'], self.params['n_exc'], self.params['n_inh'])
            simulation.run_sim(self.params, self.sim_cnt, self.params['connectivity'], connect_exc_exc)

        else: 
            print "Pc %d waiting for proc 0 to finish simulation" % self.pc_id
            time.sleep(5)
            sys.stdout.flush()

        if self.comm != None: 
            print 'Pid %d at Barrier after run_sim' % self.pc_id
            sys.stdout.flush()
            self.comm.Barrier()
        self.cycle_cnt += 1
