import pylab
import numpy as np
import sys
# --------------------------------------------------------------------------
def get_figsize(fig_width_pt):
    inches_per_pt = 1.0/72.0                # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    fig_size =  [fig_width,fig_height]      # exact figsize
    return fig_size

params2 = {'backend': 'png',
          'axes.labelsize': 12,
          'text.fontsize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'legend.pad': 0.2,     # empty space around the legend box
          'legend.fontsize': 12,
          'lines.markersize' : 0.1,
          'font.size': 12,
          'path.simplify': False,
          'figure.figsize': get_figsize(800)}

def set_figsize(fig_width_pt):
    pylab.rcParams['figure.figsize'] = get_figsize(fig_width_pt)

pylab.rcParams.update(params2)

# --------------------------------------------------------------------------

fns = sys.argv[1:]

for fn in fns:
    data = pylab.loadtxt(fn)

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    if (data.ndim == 1):
        x_axis = np.arange(data.size)
        ax.scatter(x_axis, data)
    else:
        ax.plot(data[:,0], data[:,1], 'o', markersize=1, color='k')

    ax.set_title(fn)
#    ax.set_xlim((0, 1000))
    print 'xlim:', ax.get_xlim()
#    ax.set_ylim((0, data[:, 1].max()))

pylab.show()

#output_fn = 'delme.dat'
#np.savetxt(output_fn, data)
#output_fn = 'delme.png'
#print output_fn
#pylab.savefig(output_fn)
pylab.show()
