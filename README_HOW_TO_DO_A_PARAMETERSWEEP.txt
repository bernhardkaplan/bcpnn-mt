

1) In NetworkSimModule.py

    define which parameters should be sweeped and
    this script receives them from sys.argv[1:]
    
    e.g.

        sweep_parameter = float(sys.argv[1])
        ps.params['blur_X'] = sweep_parameter

    Set the folder name so that the ps.params['folder_name'] dictionary contains the sweep_parameter.
    Otherwise all results would be written in the same folder --> not good. ...

    NetworkSimModule normally call ps.set_filenames(folder_name) before the simulation runs.


2) In run_sweep.py the NetworkSimModule is just called with the different parameter values.

    import os
    import numpy as np

    for sweep_parameter in [0.05, 0.10, 0.15]:
        os.system('mpirun -np 8 python NetworkSimModule.py %f'  % (sweep_parameter))
        # on 1 core
        #    os.system('python NetworkSimModule.py %f'  % (sweep_parameter))


3) ANALYSIS:

    python run_plot_prediction.py
    This will look for all folders that match a certain and call the script of your choice in this folders.
    You can of course run you own analysis script instead!

    Alternatively you can run sweep_through_results.py which will complain when data files are missing.


    python sweep_through_results.py
    
    This function expects results from the analysis to be in a certain filename,
    e.g. data_fn = '/Data/something_new.dat'

    It currently makes use of the ResultsCollector class to operate on the files that have been created by the run_plot_prediction.py script.

    Important functions from ResultsCollector:
     set_dirs_to_process(dir_names) # find all the folder names to be processed
     get_parameter(param_name)      # gets the parameter value for the given folders

