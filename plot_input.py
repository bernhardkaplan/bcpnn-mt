import matplotlib
#matplotlib.use('Agg')
import pylab
import numpy as np
import sys
#import rcParams
#rcP= rcParams.rcParams
import os

rcP= { 'axes.labelsize' : 18,
            'label.fontsize': 18,
            'xtick.labelsize' : 18, 
            'ytick.labelsize' : 18, 
            'axes.titlesize'  : 20,
            'legend.fontsize': 9}

pylab.rcParams.update(rcP)


# --------------------------------------------------------------------------
#def get_figsize(fig_width_pt):
#    inches_per_pt = 1.0/72.0                # Convert pt to inch
#    golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
#    fig_width = fig_width_pt*inches_per_pt  # width in inches
#    fig_height = fig_width*golden_mean      # height in inches
#    fig_size =  [fig_width,fig_height]      # exact figsize
#    return fig_size

#params2 = {'backend': 'png',
#          'axes.labelsize': 12,
#          'text.fontsize': 12,
#          'xtick.labelsize': 12,
#          'ytick.labelsize': 12,
#          'legend.pad': 0.2,     # empty space around the legend box
#          'legend.fontsize': 12,
#          'lines.markersize' : 0.1,
#          'font.size': 12,
#          'path.simplify': False,
#          'figure.figsize': get_figsize(800)}

#def set_figsize(fig_width_pt):
#    pylab.rcParams['figure.figsize'] = get_figsize(fig_width_pt)

#pylab.rcParams.update(params2)

# --------------------------------------------------------------------------


if len(sys.argv) == 2:
    gid = int(sys.argv[1])
    import simulation_parameters
    ps = simulation_parameters.parameter_storage()
    params = ps.params


elif len(sys.argv) == 3:
    gid = int(sys.argv[1])
    import json
    param_fn = sys.argv[2]
    if os.path.isdir(param_fn):
        param_fn += '/Parameters/simulation_parameters.json'
    print '\nLoading parameters from %s\n' % (param_fn)
    f = file(param_fn, 'r')
    params = json.load(f)

rate_fn = params['input_rate_fn_base'] + str(gid) + '.npy'
spike_fn = params['input_st_fn_base'] + str(gid) + '.npy'
print 'Loading data from'
print rate_fn
print spike_fn
print 'debug', params['figures_folder']

#else:
#    info = "\n\tuse:\n \
#    \t\tpython plot_input.py   [RATE_ENVELOPE_FILE]  [SPIKE_INPUT_FILE]\n \
#    \tor: \n\
#    \t\tpython plot_input.py [gid of the cell to plot]" 
#    print info


rate = np.load(rate_fn)
#rate /= np.max(rate)
y_min = rate.min()
y_max = rate.max()

spikes = np.load(spike_fn) # spikedata

#spikes *= 10. # because rate(t) = L(t) was created with a stepsize of .1 ms

binsize = 50
n_bins = int(round(params['t_sim'] / binsize))
n, bins = np.histogram(spikes, bins=n_bins, range=(0, params['t_sim']))
print 'n, bins', n, 'total', np.sum(n), 'binsize:', binsize

fig = pylab.figure()
pylab.subplots_adjust(bottom=.10, left=.12, hspace=.02, top=0.94)#55)
ax = fig.add_subplot(211)

nspikes = spikes.size
w_input_exc = 2e-3
cond_in = w_input_exc * 1000. * nspikes
print 'Cond_in: %.3e [nS] nspikes: %d' % (cond_in, nspikes)
ax.set_title('Input spike train and L(t)')
for s in spikes:
    ax.plot((s, s), (0.2 * (y_max-y_min) + y_min, 0.8 * (y_max-y_min) + y_min), c='k')
#rate_half = .5 * (np.max(rate) - np.min(rate))
#ax.plot(spikes, rate_half * np.ones(spikes.size), '|', markersize=1)
#ax.plot(spikes, 0.5 * np.ones(spikes.size), '|', markersize=1)
print 'rate', rate

n_steps = int(round(1. / params['dt_rate']))
rate = rate[::n_steps] # ::10 because dt for rate creation was 0.1 ms
ax.plot(np.arange(rate.size), rate, label='Cond_in = %.3e nS' % cond_in, lw=2, c='b')
#ax.set_xlabel('Time [ms]')
ax.set_xticks([])

ax.set_ylabel('Input rate (t) [kHz]')
def set_yticks(ax, n_ticks=5, endpoint=False):
    ylim = ax.get_ylim()
    ticks = np.linspace(ylim[0], ylim[1], n_ticks, endpoint=endpoint)
    ax.set_yticks(ticks)
    ax.set_yticklabels(['%d' % i for i in ticks])

set_yticks(ax, 5)
#ax.set_yticklabels(['', '.7', '1.4', '2.1', '2.8'])

#ax.legend()
ax = fig.add_subplot(212)
ax.bar(bins[:-1], n, width= bins[1] - bins[0])
#ax.set_title('Binned input spike train, binsize=%.1f ms' % binsize)

ax.set_xlim((0, params['t_sim']))
ax.set_ylabel('Num input spikes')
ax.set_xlabel('Time [ms]')
#ylabels = ax.get_yticklabels()
set_yticks(ax, 4)

output_fn = params['figures_folder'] + 'input_%d.png' % (gid)
print 'Saving to', output_fn
pylab.savefig(output_fn, dpi=200)

#output_fn = 'delme.dat'
#np.savetxt(output_fn, data)
pylab.show()
