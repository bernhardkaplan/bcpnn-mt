
"""
This script needs to be run on a single core for a set of simulations individually.
It will create the folder structure and print the parameter file,
with which the simulation script NetworkSimModule is to be called.
"""
import os
import simulation_parameters

def clean_up_results_directory(params):
    filenames = [params['exc_nspikes_fn_merged'], \
                params['exc_spiketimes_fn_merged'], \
                params['inh_spiketimes_fn_merged'], \
                params['inh_nspikes_fn_merged'], \
                params['exc_spiketimes_fn_base'], \
                params['inh_spiketimes_fn_base'], \
                params['merged_conn_list_ee'], \
                params['merged_conn_list_ei'], \
                params['merged_conn_list_ie'], \
                params['merged_conn_list_ii']]

    for fn in filenames:
        cmd = 'rm %s*' % (fn)
        print 'Removing %s' % (cmd)
        os.system(cmd)


ps = simulation_parameters.parameter_storage()
params = ps.params
clean_up_results_directory(params)
ps.set_filenames()
ps.create_folders()
ps.write_parameters_to_file()

print 'Ready for simulation:\n\t%s' % (params['params_fn_json'])
