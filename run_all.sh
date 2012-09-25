echo 'Starting at' `date`
python prepare_tuning_prop.py
mpirun -np 8 python prepare_spike_trains.py
echo 'Preparation stopped at' `date`
#mpirun -np 8 python prepare_connections.py
mpirun -np 8 python NetworkSimModuleNoColumns_GlobalRescaleWeight.py
python analyse_simple.py
python analyse_input.py
python plot_prediction.py
echo 'Stopping at' `date`
