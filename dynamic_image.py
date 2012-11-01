#!/usr/bin/env python
"""
An animated image
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import simulation_parameters
#import pylab
import utils

PS = simulation_parameters.parameter_storage()
params = PS.params
n_cells = params['n_exc']
fn = params['tuning_prop_means_fn']
tp = np.loadtxt(fn)
plt.rcParams['lines.markeredgewidth'] = 0

iteration = 23
activity_fn = params['activity_folder'] + 'output_activity_%d.dat' % (iteration)
activity = np.loadtxt(activity_fn)
activity_max = activity.max()
activity_min = activity.min()

scale = 1 / 10.
def paint_dots(num, pos, colors):
    print 'Frame:', num
    fig1.clear()
    dots = []
    for i in xrange(pos[:, 0].size):
        c = colors[num, i, :]
#        l, = plt.plot(pos[i, 0], pos[i, 1], 'o', c=c)
        l = plt.quiver(tp[i, 0], tp[i, 1], scale * tp[i, 2], scale * tp[i, 3], angles='xy', scale_units='xy', scale=1, color=c, headwidth=1)
        dots.append(l)
    return dots


fig1 = plt.figure()
n_frames = activity[:, 0].size
pos = np.zeros((n_cells, 2))
pos[:, 0], pos[:, 1] = tp[:, 0], tp[:, 1]
colors = np.zeros((n_frames, n_cells, 3))

s = 1. # saturation
for frame in xrange(n_frames):
    for cell in xrange(n_cells): 
        if activity[frame, cell] < 0:
            l = 1. - 0.5 * activity[frame, cell] / activity_min
            h = 0.
        else:
            l = 1. - 0.5 * activity[frame, cell] / activity_max
            h = 240.
        assert (0 <= h and h < 360)
        assert (0 <= l and l <= 1)
        assert (0 <= s and s <= 1)
        (r, g, b) = utils.convert_hsl_to_rgb(h, s, l)
        colors[frame, cell, :] = [r, g, b]


dot_ani = animation.FuncAnimation(fig1, paint_dots, n_frames, fargs=(pos, colors), interval=50, blit=True)
output_fn = 'dynamic_images.mp4'
print 'Saving movie', output_fn
dot_ani.save(output_fn)
print 'Finished'
#plt.show()
