import pylab
import numpy as np
import sys
import simulation_parameters
PS = simulation_parameters.parameter_storage()
params = PS.params

stim_id = 0
fn = params['activity_folder'] + 'network_response_no_plasticity_stim_%d.dat' % (stim_id)
data = pylab.loadtxt(fn)

t_axis = np.arange(0, params['t_sim'], params['dt_rate'])

rcParams = { 'axes.labelsize' : 18,
            'label.fontsize': 20,
            'xtick.labelsize' : 16, 
            'ytick.labelsize' : 16, 
            'legend.fontsize': 9}

pylab.rcParams.update(rcParams)
#import figure_sizes as fs
fig = pylab.figure()#figsize=fs.get_figsize(800))
ax = fig.add_subplot(111)

for i_ in xrange(data[:, 0].size):
    ax.plot(t_axis, data[i_, :], lw=3)


ax.set_xlabel('Time [ms]')
ax.set_ylabel('Response strength [a.u.]')

pylab.subplots_adjust(right=.90)
pylab.subplots_adjust(top=.90)
pylab.subplots_adjust(bottom=0.10)
pylab.subplots_adjust(left=.10)

#output_fn = 'input.png'
#pylab.savefig(output_fn)

pylab.show()
