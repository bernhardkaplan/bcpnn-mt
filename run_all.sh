echo 'Starting at' `date`
python prepare_tuning_prop.py
python prepare_selective_inhibition.py
mpirun -np 8 python prepare_spike_trains.py
echo 'Preparation stopped at' `date`
#mpirun -np 8 python prepare_connections.py
mpirun -np 8 python NetworkSimModuleNoColumns.py
#python NetworkSimModuleNoColumns.py
python analyse_simple.py
python analyse_input.py
#python plot_connlist_as_colormap.py
python plot_weight_and_delay_histogram.py
python plot_prediction.py
#python plot_input.py 205
#python plot_input.py 245
echo 'Stopping at' `date`
