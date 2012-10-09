import matplotlib
matplotlib.use('Agg')
import pylab
import numpy as np
import sys

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


params_loaded = False
if len(sys.argv) == 2:
    gid = int(sys.argv[1])
    import simulation_parameters
    ps = simulation_parameters.parameter_storage()
    params = ps.params
    rate_fn = params['input_rate_fn_base'] + str(gid) + '.npy'
    spike_fn = params['input_st_fn_base'] + str(gid) + '.npy'
    params_loaded = True

elif len(sys.argv) == 3:
    rate_fn = sys.argv[1]
    spike_fn = sys.argv[2]
else:
    info = "\n\tuse:\n \
    \t\tpython plot_input.py   [RATE_ENVELOPE_FILE]  [SPIKE_INPUT_FILE]\n \
    \tor: \n\
    \t\tpython plot_input.py [gid of the cell to plot]" 
    print info

rate = np.load(rate_fn)
rate /= np.max(rate)
y_min = rate.min()
y_max = rate.max()

spikes = np.load(spike_fn) # spikedata

#spikes *= 10. # because rate(t) = L(t) was created with a stepsize of .1 ms

n, bins = np.histogram(spikes, bins=20)
binsize = round(bins[1] - bins[0])
print 'n, bins', n, 'total', np.sum(n), 'binsize:', binsize

fig = pylab.figure()
pylab.subplots_adjust(hspace=0.35)
ax = fig.add_subplot(211)

rate_half = .5 * (np.max(rate) - np.min(rate))
nspikes = spikes.size
w_input_exc = 2e-3
cond_in = w_input_exc * 1000. * nspikes
print 'Cond_in: %.3e [nS] nspikes: %d' % (cond_in, nspikes)
ax.set_title('Input spike train and L(t)')
for s in spikes:
    ax.plot((s, s), (0.2 * (y_max-y_min) + y_min, 0.8 * (y_max-y_min) + y_min), c='k')
#ax.plot(spikes, rate_half * np.ones(spikes.size), '|', markersize=1)
#ax.plot(spikes, 0.5 * np.ones(spikes.size), '|', markersize=1)
print 'rate', rate

rate = rate[::10] # ::10 because dt for rate creation was 0.1 ms
ax.plot(np.arange(rate.size), rate, label='Cond_in = %.3e nS' % cond_in, lw=2, c='b')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Normalized motion energy')



#ax.legend()
ax = fig.add_subplot(212)
ax.bar(bins[:-1], n, width= bins[1] - bins[0])
ax.set_title('Binned input spike train, binsize=%.1f ms' % binsize)

ax.set_ylabel('Number of input spikes')
ax.set_xlabel('Times [ms]')

if params_loaded:
    output_fn = params['figures_folder'] + 'input_%d.png' % (gid)
    print 'Saving to', output_fn
    pylab.savefig(output_fn)

#output_fn = 'delme.dat'
#np.savetxt(output_fn, data)
#pylab.show()
