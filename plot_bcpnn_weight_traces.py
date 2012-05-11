import os
import simulation_parameters
#import matplotlib as mpl
import pylab as pl
import re
import numpy as np
from NeuroTools import signals as nts


pl.rcParams['legend.fontsize'] = 'small'


# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

src = 0
tgt = 2
sim_cnt = 0

gid_list = [src, tgt]
gid_list.sort()
fn = params['exc_spiketimes_fn_merged'] + str(sim_cnt) + '.ras'

spklist = nts.load_spikelist(fn)#, range(params['n_exc_per_mc']), t_start=0, t_stop=params['t_sim'])
spiketrains = spklist.spiketrains
spiketimes_pre = spiketrains[src+1.].spike_times
spiketimes_post = spiketrains[tgt+1.].spike_times

# plot spikes
markersize = 10
fig = pl.figure()
ax = fig.add_subplot(611)
for gid in gid_list:

#    fn = params['exc_spiketimes_fn_base'] + str(gid) + '.ras'
#    d = np.loadtxt(fn)
#    ax.plot(d[:, 0], np.ones(d[:,0].size)*gid, '|', markersize=markersize, color='k')
    
    d = spiketrains[gid+1.].spike_times
    ax.plot(d, np.ones(d.size)*gid, '|', markersize=markersize, color='k')

#xmin = 180
#xmax = 400
xmin = 0
xmax = params['t_sim']
ax.set_xlim((0, params['t_sim']))
ax.set_ylim(min(gid_list)-1, max(gid_list)+1)
ax.set_yticks(gid_list)
ax.set_yticklabels(['%d' % i for i in gid_list])
ax.set_title('Output spikes')



# plot z, e, p traces
colors = ['b', 'g']
ax_z = fig.add_subplot(612)
ax_e = fig.add_subplot(613)
ax_p = fig.add_subplot(614)
dt = 1 # depends on how bcpnn_offline.py computes the trace
for gid in gid_list:
    fn = params['ztrace_fn_base'] + str(gid) + '.npy'
    d = np.load(fn)
    x_axis = np.arange(0, d.size, 1)
    ax_z.plot(x_axis, d, ls='-', label='$z_%d$' % gid)
    fn = params['etrace_fn_base'] + str(gid) + '.npy'
    d = np.load(fn)
    ax_e.plot(x_axis, d, ls='-', label='$e_%d$' % gid)
    fn = params['ptrace_fn_base'] + str(gid) + '.npy'
    d = np.load(fn)
    ax_p.plot(x_axis, d, ls='-', lw=2, label='$p_%d$' % gid)

fn = params['etrace_fn_base'] + '%d_%d.npy' % (src, tgt)
d = np.load(fn)
ax_e.plot(x_axis, d, lw=2, label='$e_{%d %d}$' % (src, tgt))
fn = params['ptrace_fn_base'] + '%d_%d.npy' % (src, tgt)
d = np.load(fn)
ax_p.plot(x_axis, d, lw=2, label='$p_{%d %d}$' % (src, tgt))

# weight
ax_w = fig.add_subplot(615)
fn = params['weights_fn_base'] + '%d_%d.npy' % (src, tgt)
d = np.load(fn)
ax_w.plot(x_axis, d, lw=2, label=r'$w_{%d %d}$' % (src, tgt))
ax_w.plot(x_axis, np.ones(d.size)*d.mean(), 'r--', label=r'mean($w_{%d %d}$)' % (src, tgt))

# bias
ax_b = fig.add_subplot(616)
fn = params['bias_fn_base'] + '%d.npy' % (tgt)
d = np.load(fn)
ax_b.plot(x_axis, d, lw=2, label=r'$\beta_%d$' % (tgt))
ax_b.plot(x_axis, np.ones(d.size)*d.mean(), 'r--', label=r'mean($\beta_{%d}$)' % (tgt))

ax_z.legend()
ax_e.legend()
ax_p.legend()
ax_w.legend()
ax_b.legend()

ax_z.set_xlim((xmin, xmax))
ax_e.set_xlim((xmin, xmax))
ax_p.set_xlim((xmin, xmax))
ax_w.set_xlim((xmin, xmax))
ax_b.set_xlim((xmin, xmax))

#connection_matrix = np.load(params['conn_mat_init'])
#non_zeros = connection_matrix.nonzero()
#conns = zip(non_zeros[0], non_zeros[1])
#pl.subplots_adjust(bottom=0.05, top=0.9, hspace=-0.1)

#ax = fig.get_axes()[0]
#ax.set_xticks(xticks)
#ax.set_xticklabels(['%d' % i for i in xticks])
#ax.set_title('Input spike counts')

pl.show()
