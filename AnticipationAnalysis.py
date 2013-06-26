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


cell_type = 'exc'


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

def get_average_spikerate(spiketrains, pops, n_bins=20):

    avg_rate = np.zeros((n_bins, len(pops)))

    for i_, p in enumerate(pops):
        for gid in p:
            if not (spiketrains[gid] == []):
                n, bins = np.histogram(spiketrains[gid], bins=n_bins)
                avg_rate[:, i_] += n
        avg_rate[:, i_] /= float(len(p))

    return avg_rate, bins


tp = np.loadtxt(params['tuning_prop_means_fn'])

# this doesn't work anymore, because we recorded cells given from utils.select_well_tuned_cells
#n_pop = 6
#selected_gids, pops = utils.select_well_tuned_cells_trajectory(tp, params['motion_params'], params, params['n_gids_to_record'], n_pop)
#print 'pops', pops

n_pop = 1
recorded_gids = np.loadtxt(params['gids_to_record_fn'], dtype=np.int)
print 'Recorded_gids', recorded_gids
pops = [recorded_gids]

spike_fn = params['%s_spiketimes_fn_merged' % cell_type] + '.ras'
assert (os.path.exists(spike_fn)), 'File not found %s' % spike_fn
spikes = np.loadtxt(spike_fn)
spiketrains = utils.get_spiketrains(spikes, n_cells=params['n_%s' % cell_type])
avg_rate, avg_rate_bins = get_average_spikerate(spiketrains, pops)

fn = params['%s_volt_anticipation' % cell_type]
fn_g = params['%s_gsyn_anticipation' % cell_type]
#fn = "ResultsBar_AIII/VoltageTraces/exc_volt_anticipation_small.v"
#fn_g = "ResultsBar_AIII/CondTraces/exc_gsyn_anticipation_small.dat"
print 'Loading ', fn
d_volt = np.loadtxt(fn)
print 'Loading ', fn_g
d_gsyn = np.loadtxt(fn_g)


print 'debug gid', pops[0][0]
time_axis, volt = utils.extract_trace(d_volt, pops[0][0])
print 'debug time_axis', time_axis
print 'debug volt', volt


avg_volts = np.zeros((time_axis.size, len(pops) + 1))
avg_gsyns = np.zeros((time_axis.size, len(pops) + 1))
avg_currs = np.zeros((time_axis.size, len(pops) + 1))

avg_volts[:, 0] = time_axis
avg_gsyns[:, 0] = time_axis
avg_currs[:, 0] = time_axis
#selected_gids = utils.all_anticipatory_gids(params)
#print 'selected_gids', len(selected_gids)
for j_, pop in enumerate(pops): 
    print 'debug pop[0]', pop[0]
    time_axis, volt = utils.extract_trace(d_volt, pop[0])
    volt_sum = np.zeros(time_axis.size)
    gsyn_sum = np.zeros(time_axis.size)
    curr_sum = np.zeros(time_axis.size)
    x_group, y_group, u_group, v_group, o_group = np.zeros(len(pop)),np.zeros(len(pop)),np.zeros(len(pop)),np.zeros(len(pop)), np.zeros(len(pop))
    for i_, gid in enumerate(pop):
        print 'Loading volt for gid:', gid
        x, y, u, v, o = tp[gid, :]
        x_group[i_] = x
        y_group[i_] = y
        u_group[i_] = u
        v_group[i_] = v
        o_group[i_] = o

        time_axis, volt = utils.extract_trace(d_volt,gid)
        print 'debug volt shape', volt.shape, volt_sum.shape
        volt_sum += volt
        time_axis, gsyn = utils.extract_trace(d_gsyn,gid)
        gsyn_sum += gsyn
        curr_sum += (gsyn * volt)

#    print 'debug, population info', j_
#    print 'x_avg:', x_group.mean(), x_group.std()
#    print 'y_avg:', y_group.mean(), y_group.std()
#    print 'u_avg:', u_group.mean(), u_group.std()
#    print 'v_avg:', v_group.mean(), v_group.std()
#    print 'o_avg:', o_group.mean(), o_group.std()
    avg_volt = volt_sum / len(pop)
    avg_volts[:, j_ + 1] = avg_volt
    
    avg_gsyn = gsyn_sum / len(pop)
    avg_gsyns[:, j_ + 1] = avg_gsyn
   
    avg_curr = curr_sum / len(pop)
    avg_currs[:, j_ + 1] = avg_curr * (-1) # because currents should be positive ...



#volt_data_fn = params['population_volt_fn']
#print 'Saving output to:', volt_data_fn
#np.savetxt(volt_data_fn, avg_volts)

#cond_data_fn = params['population_cond_fn']
#print 'Saving output to:', cond_data_fn
#np.savetxt(cond_data_fn, avg_conds)

#curr_data_fn = params['population_curr_fn']
#print 'Saving output to:', curr_data_fn
#np.savetxt(curr_data_fn, avg_currs)

colorlist = ['k', 'b', 'g', 'r', 'y', 'c', 'm', '#00f80f', '#deff00', '#ff00e4', '#00ffe6']
fig = pylab.figure()
# set figure parameters, figure size, font sizes etc
fig_width_pt = 800.0  
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0      # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
fig_params = {
          'titel.fontsize': 18,
          'axes.labelsize': 14,
          'figure.figsize': fig_size}
pylab.rcParams.update(fig_params)
# ------------------------

ax0 = fig.add_subplot(411)
ax1 = fig.add_subplot(412)
ax2 = fig.add_subplot(413)
ax3 = fig.add_subplot(414)
for i in xrange(len(pops)):
    ax0.plot(avg_rate_bins[:-1], avg_rate[:, i], 'o-', color=colorlist[i])
    ax1.plot(time_axis, avg_volts[:, i+1], label='pop %d volt' % i, ls='--', lw=2, color=colorlist[i])
    ax2.plot(time_axis, avg_gsyns[:, i+1], label='pop %d gsyn' % i, ls='--', lw=2, color=colorlist[i])
    ax3.plot(time_axis, avg_currs[:, i+1], label='pop %d curr' % i, ls='--', lw=2, color=colorlist[i])


ax0.set_ylabel('Firing rate [Hz]')
ax0.set_title('Mean output_rate, input_signal, v_mem, \n conductances and currents \n for %d groups averaged over ~%d cells per group' % (n_pop, len(pops[0])))
ax1_input = ax1.twinx()
ax2_input = ax2.twinx()
ax3_input = ax3.twinx()
for i in xrange(len(pops)):
    L_avg, L_std, time_coarse = recompute_input(params, tp, pops[i])
    ax1_input.plot(time_coarse, L_avg, lw=3, color=colorlist[i])
    ax2_input.plot(time_coarse, L_avg, lw=3, color=colorlist[i])
    ax3_input.plot(time_coarse, L_avg, lw=3, color=colorlist[i])

#    ax1_input.errorbar(time_coarse, L_avg, yerr=L_std / np.sqrt(len(pops[i])), lw=3, ls='--')

ax1.set_ylabel('V_m [mV]')
ax2.set_ylabel('G_syn [uS]')
ax3.set_ylabel('I [nA]')
ax3.set_xlabel('Time [ms]')

pylab.legend()
output_fn = params['figures_folder'] + 'anticipatory_avg_traces.png'
print 'Saving to', output_fn
pylab.savefig(output_fn)

pylab.show()
