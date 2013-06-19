
import os
import sys
import numpy as np
import utils
import pylab

fn1 = sys.argv[1]
fn2 = sys.argv[2]

legend_txt = '%s - %s' % (fn1, fn2)

d1 = np.loadtxt(fn1)
d2 = np.loadtxt(fn2)
diff = d1 - d2

t_axis = d1[:, 0]
fig = pylab.figure()
ax = fig.add_subplot(111)
ax.plot(t_axis, diff, label=legend_txt)
pylab.legend()
pylab.show()
