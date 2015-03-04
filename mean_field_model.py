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


    t_axis = np.arange(0., 1., 0.1)
    x_axis = np.arange(0., 1., 0.02)
    sigma = 0.1
    alpha = 0.5
    x0, v0 = 0.2, 1.
    blur_x = 0.1

    clim = (t_axis[0], t_axis[-1])
    norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m.set_array(t_axis)
    colorlist = m.to_rgba(t_axis)

    fig = pylab.figure()
    ax = fig.add_subplot(111)

    for i_time, t_ in enumerate(t_axis):
        x_stim = x0 + v0 * t_
        stimulus = mlab.normpdf(x_axis, x_stim, blur_x)
        ax.plot(x_axis, stimulus, c=colorlist[i_time])
#        for i_x, x_ in enumerate(x_axis):

            #activation = skew_normal(

    pylab.show()
#ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=10,
#    repeat=False)

