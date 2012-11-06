export NCPUS=8
export NCPUS=1
echo 'Starting at ' `date` ' with ' $NCPUS ' cpus'
python prepare_tuning_prop.py 0.1 0.3 # give blur_X and blur_V as arguments
python prepare_selective_inhibition.py
mpirun -np $NCPUS python prepare_spike_trains.py 0.1 0.3 
mpirun -np $NCPUS python prepare_connections.py
echo 'Preparation stopped at' `date`
mpirun -np $NCPUS python NetworkSimModuleNoColumns.py
echo 'Simulation stopped at' `date`
python analyse_simple.py
python analyse_input.py
#python plot_connlist_as_colormap.py
python plot_weight_and_delay_histogram.py
python plot_prediction.py
#python plot_input.py 205
#python plot_input.py 245
echo 'Stopping at' `date`
