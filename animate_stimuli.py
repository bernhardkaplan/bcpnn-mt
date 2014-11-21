import numpy as np
import random
import pylab
from matplotlib import animation
import simulation_parameters
import CreateInput
PS = simulation_parameters.parameter_storage()
params = PS.load_params()                       # params stores cell numbers, etc as a dictionary


n_frames_per_stim = 2 * int(params['t_sim'] / params['t_stimulus'])
n_theta = params['n_theta']
n_training_v = params['n_training_v']
n_cycles = params['n_cycles']
n_stim_per_direction = params['n_stim_per_direction']
n_stim_total = n_training_v * n_theta * n_cycles * n_stim_per_direction
n_frames_total = n_frames_per_stim * n_stim_total

random_order = False
CS = CreateInput.CreateInput()
CS.create_motion_sequence_2D(params, random_order)
all_speeds, all_starting_pos, all_thetas = CS.get_motion_params(random_order=random_order)

# arrays to be filled by the stimulus creation loops below

rcParams = { 'axes.labelsize' : 18,
            'label.fontsize': 20,
            'xtick.labelsize' : 16, 
            'ytick.labelsize' : 16, 
            'axes.titlesize'  : 20,
            'legend.fontsize': 9}
pylab.rcParams.update(rcParams)

# setup
fig = pylab.figure()
ax = fig.add_subplot(111,aspect=1.)
ax.set_xlim((-0.2, 1.2))
ax.set_ylim((-0.2, 1.2))

dot, = ax.plot([], [], 'ko', ms=5)
trace, = ax.plot([], [], '-', lw=2)

rnd_idx = range(n_stim_total)
random.shuffle(rnd_idx)

color_list = ['k', 'b', 'g', 'r', 'y', 'c', 'm', '#00f80f', '#deff00', '#ff00e4', '#00ffe6']

def init_rect():
    ax.plot([0, 1], [0, 0], 'k--', lw=3)
    ax.plot([1, 1], [0, 1], 'k--', lw=3)
    ax.plot([1, 0], [1, 1], 'k--', lw=3)
    ax.plot([0, 0], [1, 0], 'k--', lw=3)
    return dot, trace, 

def animate_dot(i):
    x_pos = x0 + vx * (i / 100.)
    y_pos = y0 + vy * (i / 100.)
    dot.set_data(x_pos, y_pos)
    trace.set_data([x0, x_pos], [y0, y_pos])
    return dot, trace,

def animate_trace(i):
    stim_id = i / n_frames_per_stim
    rnd_stim = rnd_idx[stim_id]
    theta = all_thetas[rnd_stim]
#    v = 5 * all_speeds[rnd_stim]
    v = all_speeds[rnd_stim]
    t0 = stim_id * n_frames_per_stim
    vx, vy = v * np.cos(theta), - v * np.sin(theta)
    x0, y0 = all_starting_pos[rnd_stim, :]
    x_pos = x0 + vx * ((i - t0) / 100.)
    y_pos = y0 + vy * ((i - t0) / 100.)
    dot.set_data(x_pos, y_pos)
    trace.set_data([x0, x_pos], [y0, y_pos])
    color_idx = rnd_stim / n_stim_per_direction
    trace.set_color(color_list[color_idx % len(color_list)])
    return trace, dot, 

# call the animator.  blit=True means only re-draw the parts that have changed.
#anim_dot = animation.FuncAnimation(fig, animate_dot, init_func=init_dot,
#                               frames=300, interval=20, blit=True)

anim_trace = animation.FuncAnimation(fig, animate_trace, init_func=init_rect,
                               frames=n_frames_total, interval=5, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


#init_rect()

#n_stim_total = params['n_theta'] * params['n_training_v'] * params['n_cycles'] * params['n_stim_per_direction']
#stim_start = 0
#stim_stop = 16
#scale = 1
#xpos_list = []
#ypos_list = []
#vx_list = []
#vy_list = []
#c_list = []
#for stim_id in xrange(n_stim_total):
#    theta = all_thetas[stim_id]
#    v = 1.0 * all_speeds[stim_id]
#    vx, vy = v * np.cos(theta), - v * np.sin(theta)
#    x0, y0 = all_starting_pos[stim_id, :]
#    print 'debug stim_id %d' % stim_id, x0, y0, v, vx, vy
#    x_pos = x0 + vx
#    y_pos = y0 + vy
#    xpos_list.append(x0)
#    ypos_list.append(y0)
#    vx_list.append(vx)
#    vy_list.append(vy)
#    color_idx = stim_id / n_stim_per_direction
#    c_list.append(color_list[color_idx % (len(color_list))])

#    c = color_list[color_idx % len(color_list)]
#    ax.quiver(x0, y0, x_pos, y_pos, \
#          angles='xy', scale_units='xy', scale=scale, color=c, headwidth=4, pivot='head')

#ax.quiver(xpos_list, ypos_list, vx_list, vy_list, \
#      angles='xy', scale_units='xy', scale=scale, color=c_list, headwidth=4, pivot='tail')

#for stim_id in xrange(n_stim_total):
#    theta = all_thetas[stim_id]
#    v = 3.0 * all_speeds[stim_id]
#    vx, vy = v * np.cos(theta), - v * np.sin(theta)
#    x0, y0 = all_starting_pos[stim_id, :]
#    print 'debug stim_id %d' % stim_id, x0, y0, v, vx, vy
#    x_pos = x0 + vx
#    y_pos = y0 + vy
#    color_idx = stim_id / n_stim_per_direction
#    c = color_list[color_idx % (len(color_list))]
#    ax.plot([x0, x_pos], [y0, y_pos], c=c, ls=':', lw=3)

#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_title('Training stimuli')
pylab.show()

