import os
import sys
import numpy as np
import utils
import pylab

if len(sys.argv) > 1:
    param_fn = sys.argv[1]
    if os.path.isdir(param_fn):
        param_fn += '/Parameters/simulation_parameters.json'
    import json
    f = file(param_fn, 'r')
    print 'Loading parameters from', param_fn
    params = json.load(f)
else:
    import simulation_parameters as sp
    network_params = sp.parameter_storage()
    params = network_params.load_params()


def recompute_input(params, tp, gids):
    """
    Returns the envelope of the poisson process that is fed into the cells with gids
    Keyword arguments:
    params -- parameter dictionary
    tp -- tuning property array of all cells
    gids -- array or list of cell gids
    """
    dt = params['dt_rate'] * 10# [ms] time step for the non-homogenous Poisson process
    time = np.arange(0, params['t_sim'], dt)
    n_cells = len(gids)
    L_input = np.zeros((n_cells, time.shape[0]))
    L_avg = np.zeros(time.shape[0])
    L_std = np.zeros(time.shape[0])
    for i_time, time_ in enumerate(time):
        L_input[:, i_time] = utils.get_input(tp[gids, :], params, time_/params['t_stimulus'], motion=params['motion_type'])
        L_avg[i_time] = L_input[:, i_time].mean()
        L_std[i_time] = L_input[:, i_time].std()
    return L_avg, L_std, time

tp = np.loadtxt(params['tuning_prop_means_fn'])
fn = params['exc_volt_anticipation']
fn_g = params['exc_gsyn_anticipation']
#fn = "ResultsBar_AIII/VoltageTraces/exc_volt_anticipation_small.v"
#fn_g = "ResultsBar_AIII/CondTraces/exc_gsyn_anticipation_small.dat"
print 'Loading ', fn
d_volt = np.loadtxt(fn)
print 'Loading ', fn_g
d_gsyn = np.loadtxt(fn_g)

n_pop = 4
selected_gids, pops = utils.select_well_tuned_cells(tp, params, params['n_gids_to_record'], n_pop)
print 'pops', pops

time_axis, volt = utils.extract_trace(d_volt, pops[0][0])

avg_volts = np.zeros((time_axis.size, len(pops) + 1))
avg_gsyns = np.zeros((time_axis.size, len(pops) + 1))
avg_currs = np.zeros((time_axis.size, len(pops) + 1))

avg_volts[:, 0] = time_axis
avg_gsyns[:, 0] = time_axis
avg_currs[:, 0] = time_axis
#selected_gids = utils.all_anticipatory_gids(params)
print 'selected_gids', len(selected_gids)
for j_, pop in enumerate(pops): 
    print 'debug', pop, len(pop)
    time_axis, volt = utils.extract_trace(d_volt, pop[0])
    volt_sum = np.zeros(time_axis.size)
    gsyn_sum = np.zeros(time_axis.size)
    curr_sum = np.zeros(time_axis.size)
    x_group, y_group, u_group, v_group, o_group = np.zeros(len(pop)),np.zeros(len(pop)),np.zeros(len(pop)),np.zeros(len(pop)), np.zeros(len(pop))
    for i_, gid in enumerate(pop):
        x, y, u, v, o = tp[gid, :]
        x_group[i_] = x
        y_group[i_] = y
        u_group[i_] = u
        v_group[i_] = v
        o_group[i_] = o

        time_axis, volt = utils.extract_trace(d_volt,gid)
        volt_sum += volt
        time_axis, gsyn = utils.extract_trace(d_gsyn,gid)
        gsyn_sum += gsyn
        curr_sum += (gsyn * volt)

    print 'debug, population info', j_
    print 'x_avg:', x_group.mean(), x_group.std()
    print 'y_avg:', y_group.mean(), y_group.std()
    print 'u_avg:', u_group.mean(), u_group.std()
    print 'v_avg:', v_group.mean(), v_group.std()
    print 'o_avg:', o_group.mean(), o_group.std()
    avg_volt = volt_sum / len(pop)
    avg_volts[:, j_ + 1] = avg_volt
    
    avg_gsyn = gsyn_sum / len(pop)
    avg_gsyns[:, j_ + 1] = avg_gsyn
   
    avg_curr = curr_sum / len(pop)
    avg_currs[:, j_ + 1] = avg_curr * (-1) # because currents should be positive ...


#data_fn = params['population_voltages_fn']
data_fn = 'temp_output.dat'
print 'Saving output to:', data_fn
np.savetxt(data_fn, avg_volts)

fig = pylab.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
print 'debug', len(pops)
for i in xrange(len(pops)):
    ax1.plot(time_axis, avg_volts[:, i+1], label='pop %d volt' % i, lw=3)
    ax2.plot(time_axis, avg_gsyns[:, i+1], label='pop %d gsyn' % i, lw=3)
    ax3.plot(time_axis, avg_currs[:, i+1], label='pop %d curr' % i, lw=3)


ax1_input = ax1.twinx()
ax2_input = ax2.twinx()
ax3_input = ax3.twinx()
for i in xrange(len(pops)):
    L_avg, L_std, time_coarse = recompute_input(params, tp, pops[i])
#    ax1_input.errorbar(time_coarse, L_avg, yerr=L_std / np.sqrt(len(pops[i])), lw=3, ls='--')
    ax1_input.plot(time_coarse, L_avg, lw=3, ls='--')
    ax2_input.plot(time_coarse, L_avg, lw=3, ls='--')
    ax3_input.plot(time_coarse, L_avg, lw=3, ls='--')


ax1.set_ylabel('V_m [mV]')
ax2.set_ylabel('G_syn [uS]')
ax3.set_ylabel('I [nA]')

pylab.legend()
output_fn = params['figures_folder'] + 'anticipatory_avg_traces.png'
print 'Saving to', output_fn
pylab.savefig(output_fn)

pylab.show()
