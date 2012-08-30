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

"""

    use:

    python plot_input.py   [RATE_ENVELOPE]  [SPIKE_INPUT_FILE]

"""


rate_fn = sys.argv[1]
rate = np.load(rate_fn)

spike_fn = sys.argv[2]
spikes = np.load(spike_fn) # spikedata

#spikes *= 10. # because rate(t) = L(t) was created with a stepsize of .1 ms

n, bins = np.histogram(spikes, bins=15)
binsize = bins[1] - bins[0]
print 'n, bins', n, 'total', np.sum(n), 'binsize:', binsize

fig = pylab.figure()
ax = fig.add_subplot(211)

rate_half = .5 * (np.max(rate) - np.min(rate))
nspikes = spikes.size
w_input_exc = 2e-3
cond_in = w_input_exc * 1000. * nspikes
print 'Cond_in: %.3e [nS] nspikes: %d' % (cond_in, nspikes)
ax.set_title('Input spike train and L(t)')
ax.plot(spikes, rate_half * np.ones(spikes.size), '|', markersize=1)
print 'rate', rate
rate = rate[::10] # ::10 because dt for rate creation was 0.1 ms
ax.plot(np.arange(rate.size), rate, label='Cond_in = %.3e nS' % cond_in)
ax.legend()
ax = fig.add_subplot(212)
ax.bar(bins[:-1], n)
ax.set_title('Binned input spike train, binsize=%.1f ms' % binsize)


#output_fn = 'delme.dat'
#np.savetxt(output_fn, data)
output_fn = 'delme1.png'
print 'Saving to', output_fn
pylab.savefig(output_fn)
pylab.show()
