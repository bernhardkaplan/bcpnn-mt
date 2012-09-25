import numpy as np
import pylab
from NeuroTools import parameters as ntp

folder_name = 'Times/' # where the files are stored
fn_base = '%s/times_dict_np' % folder_name

n_procs = [24, 48, 96, 192]

def get_values(key):
    d = np.zeros(len(n_procs))
    for i_, n_proc in enumerate(n_procs):
        fn = fn_base + '%d.py' % n_proc
        times = ntp.ParameterSet(fn)
        dct = dict(times) 
        d[i_] = dct[key]
    return d



fig = pylab.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

keys = ['t_all', 't_connect', 't_create', 't_record', 't_calc_conns', 't_sim']
for key in keys:
    data = get_values(key)

    d_normed = data / data[0] 
    ax.plot(n_procs, data, lw=2, label=key)
    ax2.plot(n_procs, d_normed, lw=2, label=key)


ideal_curve = [ 24. / n_procs[i] for i in xrange(len(n_procs))]
ax2.plot(n_procs, ideal_curve, ls='--', label='ideal')
pylab.legend()

#ax.set_xscale('log')
#ax.set_title(key)
ax.set_xlabel('Num cores')
ax.set_ylabel('Time [s]')
ax2.set_xlabel('Num cores')
ax2.set_ylabel('Time compared to %d cores' % (n_procs[0]))

ax.set_xticks(n_procs)
ax2.set_xticks(n_procs)

pylab.show()
