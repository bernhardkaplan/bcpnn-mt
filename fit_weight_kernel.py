import sys
import os
import numpy as np
import re
import pylab
import matplotlib.mlab as mlab
from scipy.optimize import leastsq
from scipy.special import erf # error function
from scipy.spatial.distance import euclidean
import scipy.stats
import utils

def residuals_function_quadratic(p, y, x):
    """
    x: normal x coordinate
    p: parameters of the function to fit, e.g. a and b in y = a * x + b
    """
    y1 = quadratic_function_vertex(x, p)
    err = y - y1
    return err 


def residuals_function_gauss(p, y, x):
    """
    x: normal x coordinate
    p: parameters of the function to fit, e.g. a and b in y = a * x + b
    """
    y1 = gauss_shifted(x, p)
    err = y - y1
    return err 


def gauss_shifted(x, p):
    return (p[2] - p[3]) * mlab.normpdf(x, p[0], p[1]) + p[3]


def quadratic_function_vertex(x, p):
    """
    y = a * (x - h)**2 + k
    vertex (x, y) coordinates is (h, k) 
    """
    y = p[0] * (x - p[1])**2 + p[2]
    return y



def get_quality_of_fit(y_measured, y_theory, variance, n_degrees_of_freedom):
    return np.sum( (y_measured - y_theory)**2 / variance**2 ) / n_degrees_of_freedom


def fit_gaussian_kernel(params, n_points=None, fig=None):
    data_fn = params['data_folder'] + 'average_weights_vs_distance.dat'
    d = np.loadtxt(data_fn)
    x = d[:, 0]
    y = d[:, 1]
    variance = d[:, 2]

    # limit fit range, to not overweigh x-range borders
    if n_points != None:
        limit_fit_range = True
    else:
        limit_fit_range = False
    if limit_fit_range:
        nx = int(x.size)
        x_limited = x[nx / 2 - n_points : nx / 2 + n_points + 1]
        y_limited = y[nx / 2 - n_points : nx / 2 + n_points + 1]
        x_fit = x_limited
        y_fit = y_limited
    else:
        x_fit = x
        y_fit = y

    a_init = -1.
    h_init = 0.
    k_init = 2.
    guess_params_quadratic = [a_init, h_init, k_init]
    guess_params_gauss = [0., 0.25, 2., -10.]

    if fig == None:
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x, y)
    else:
        ax = fig.get_axes()[0]

#    opt_params_gauss = leastsq(residuals_function_gauss, guess_params_gauss, args=(y_limited, x_limited), maxfev=1000)[0]
#    opt_params_quadratic = leastsq(residuals_function_quadratic, guess_params_quadratic, args=(y_limited, x_limited), maxfev=1000)[0]

    opt_params_gauss = leastsq(residuals_function_gauss, guess_params_gauss, args=(y_fit, x_fit), maxfev=1000)[0]
    print 'opt_params_gauss:', opt_params_gauss
    opt_gauss = gauss_shifted(x, opt_params_gauss)
    ax.plot(x, opt_gauss)

    if limit_fit_range:
        y_theory = opt_gauss[nx / 2 - n_points : nx / 2 + n_points + 1]
        variance = variance[nx / 2 - n_points : nx / 2 + n_points + 1]
    else:
        y_theory = opt_gauss

    n_degrees_of_freedom = x_fit.size - len(guess_params_gauss) - 1
    reduced_chi_square = get_quality_of_fit(y_fit, y_theory, variance, n_degrees_of_freedom)
    print 'reduced_chi_square:', reduced_chi_square

#    opt_params_quadratic = leastsq(residuals_function_quadratic, guess_params_quadratic, args=(y, x), maxfev=1000)[0]
#    print 'opt_params_quadratic:', opt_params_quadratic
#    opt_quadratic = quadratic_function_vertex(x, opt_params_quadratic)
#    ax.plot(x, opt_quadratic)

    ax.set_ylim((-15., 5.))
    ax.set_xlim((x.min(), x.max()))

    return fig, reduced_chi_square, opt_params_gauss


if __name__ == '__main__':

    folder_name = sys.argv[1]
    params = utils.load_params(folder_name)

    fig, reduced_chi_square, opt_params = fit_gaussian_kernel(params)

#    fig = None
#    for n_points in xrange(10, 40, 1):
#        fig, reduced_chi_square, opt_params = fit_gaussian_kernel(params, n_points, fig)
#        print 'reduced_chi_square:', reduced_chi_square, n_points
    pylab.show()
