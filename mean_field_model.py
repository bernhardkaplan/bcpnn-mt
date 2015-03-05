import matplotlib
import matplotlib.cm
import matplotlib.mlab as mlab
import utils
import pylab
import numpy as np
#import matplotlib.animation as animation
from scipy.special import erf

def skew_normal(x, mu, sigma, alpha):
    # alpha  = skewness parameter
    return (1. / sigma) * mlab.normpdf((x - mu)/sigma, 0., 1.) * (1. + erf(alpha * (x - mu) / (sigma * np.sqrt(2))) )


if __name__ == '__main__':

    dx = 0.02
    dt = 0.05
    x_max = 3.
    t_axis = np.arange(0., .2, dt)
    x_axis = np.arange(0., x_max, dx)
    n_x = x_axis.size
    n_time = t_axis.size
    activity_field = np.zeros(n_x)
    activity_field_prv = np.zeros(n_x)
    sigma_conn = 0.05   # width connectivity kernel
    alpha = 0.5
    x0, v0 = 0.2, 1.
    blur_x = 0.1    # stimulus width
    tau = .3
    w_ee_amp = 0.005
    w_ie_amp = 0.001
    w_input = 0.
    t_start_blank = 0.5
    t_stop_blank = 0.8
    E_max = 1.
    E_min = -1.

    clim = (t_axis[0], t_axis[-1])
    norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m.set_array(t_axis)
    colorlist = m.to_rgba(t_axis)

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    for i_time, t_ in enumerate(t_axis):
        x_stim = x0 + v0 * t_
        print 't_:', t_
        if t_ > t_start_blank and t_ < t_stop_blank:
            stimulus = np.zeros(n_x)
            print 'stim = 0 at t=%.2f' % t_
        else:
            stimulus = mlab.normpdf(x_axis, x_stim, blur_x)
        ax.plot(x_axis, stimulus, c=colorlist[i_time], ls=':')
        for i_x, x_ in enumerate(activity_field):
            # network contribution
            net_input = 0.
            for j_x, x_src in enumerate(activity_field_prv):
#                net_input += w_ee_amp * mlab.normpdf(x_axis, x_, sigma_conn)[j_x] * (activity_field_prv[i_x] - E_max) - w_ie_amp * (activity_field_prv[i_x] - E_min)
                net_input += w_ee_amp * mlab.normpdf(x_axis, x_src, sigma_conn)[j_x] #- w_ie_amp
#                print 'net_input:', net_input
            da = - tau * activity_field_prv[i_x] * dt + w_input * stimulus[i_x] + net_input
            activity_field[i_x] += da
            activity_field_prv[i_x] = activity_field[i_x]
        ax.plot(x_axis, activity_field, c=colorlist[i_time], label='t=%.2f' % t_)

#            update(activity_field, activity



            #activation = skew_normal(
    ax.legend()
    pylab.show()
#ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=10,
#    repeat=False)

