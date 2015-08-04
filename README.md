
Training:
    mpirun -np 4 python main_training_orientations.py

After training run:
    python WeightAnalyser.py [FOLDERNAME]
    FOLDERNAME is the folder that has been created by main_training_orientations.py.
    Averages the weights across cells within a minicolumn and prints the filename where they are stored, e.g.:
        Saving connection matrix to: TrainingSim_asymmetricTauij__2x1_0-16_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_v0.40-0.8_fmax500/Connections/conn_matrix_mc.dat
    This horribly long filename TrainingSim_asymmetricTauij__2x1_0-16_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_v0.40-0.8_fmax500/Connections/conn_matrix_mc.dat
    must be used in main_test.py 


Testing:
    python main_test.py
    In there you have to set the filename where to find the connection matrices.


The most important files are: 
NetworkModelPyNest.py
    -- contains the actual model as a class implementing functions that can be used in different contexts (training, testing) that do all the work, setting up the network, connecting input, getting weights etc
    e.g. 
        from NetworkModelPyNest import NetworkModel
        NM = NetworkModel(params, iteration=0, comm=comm)
        NM.setup() # create stimuli
        NM.create()
        NM.connect()
    ....

simulation_parameters.py  
    -- Class container for a quite unordered collection of all sorts of parameters which are used from basically all scripts at some point

utils.py    
    -- contains all sorts of helper functions that are used at various occasions and in different contexts

main_test.py
    -- requires that the learned weights are averaged across minicolumns, hence run WeightAnalyser.py before running main_test.py (see above)
    It also runs the analysis and plotting on the activity during testing.

create_training_stimuli.py
    -- contains various functions that can create different training and testing protocols


