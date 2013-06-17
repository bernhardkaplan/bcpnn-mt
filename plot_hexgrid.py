import pylab
import numpy as np

def get_hexgrid_edges(midpoints, midpoints_x_difference):
    """
    plots the edges around the given midpoints as in a 'east-west' hexgrid
    assume midpoints[0, :] gives the x-coordinates
    assume midpoints[1, :] gives the y-coordinates

    Definitions:
    C = 2 * A           # the grid constant (length of an edge)
    A = B / sqrt(3)

    """

    # get the grid constant from the first two midpoints being at the same x-position
    B = (midpoints[1, 1] - midpoints[1, 0]) / 2. # 
    
    A = midpoints_x_difference / 3. # because it's math ... 
    print 'B:', B
    print 'A:', A

    hexgrid_edges = []
    for i in xrange(midpoints[0, :].size):
        x, y = midpoints[0, i], midpoints[1, i]
        # calculate the six points marking the edges
        p1 = (x - 2 * A,    y)
        p2 = (x - A,        y - B)
        p3 = (x + A,        y - B)
        p4 = (x + 2 * A,    y)
        p5 = (x + A,        y + B)
        p6 = (x - A,        y + B)

        edgepoints = [p1, p2, p3, p4, p5, p6]

        # calculate the lines connecting 1 - 2, 2 - 3, 3 - 4, 4 - 5, 5 - 6, 6 - 1
        for i in xrange(len(edgepoints)):
            x1, y1 = edgepoints[i]
            x2, y2 = edgepoints[(i + 1) % len(edgepoints)]
            hexgrid_edges.append(((x1, x2), (y1, y2)))

    return hexgrid_edges


def get_hexgridcell_midpoints(N_RF):
    # np.sqrt(np.sqrt(3)) comes from resolving the problem "how to quantize the square with a hex grid of N_RF**2 dots?"
    N_RF_X = np.int(np.sqrt(N_RF*np.sqrt(3)))
    N_RF_Y = np.int(np.sqrt(N_RF))#/np.sqrt(3)))
    print 'N_RF_X, Y:', N_RF_X, N_RF_Y
    RF = np.zeros((2, N_RF_X*N_RF_Y))
    xlim = (0, 1)
    ylim = (0, np.sqrt(3))
    X, Y = np.mgrid[xlim[0]:xlim[1]:1j*(N_RF_X+1), ylim[0]:ylim[1]:1j*(N_RF_Y+1)]
#    X, Y = np.mgrid[0:1:1j*(N_RF_X+1), 0:1:1j*(N_RF_Y+1)]

    # It's a torus, so we remove the first row and column to avoid redundancy (would in principle not harm)
    X, Y = X[1:, 1:], Y[1:, 1:]
    # Add to every even Y a half RF width to generate hex grid
    Y[::2, :] += (Y[0, 0] - Y[0, 1])/2

    # some modification to fix the grid
    Y /= np.sqrt(3)

    width = 2.
    # finalize the vecvtor of RF coordinates
    RF[0, :] = X.ravel() * width / 2.
    RF[1, :] = Y.ravel() * width / 2.
    return RF


if __name__ == '__main__':

    N_RF = 100
    RF = get_hexgridcell_midpoints(N_RF)

    x_size = 10
    fig = pylab.figure(figsize=(x_size, x_size))

    ax = fig.add_subplot(111, autoscale_on=False, aspect='equal')

    for i in xrange(RF[0, :].size):
        ax.plot(RF[0, i], RF[1, i], 'o', c='k')
        ax.annotate('%d' % i, (RF[0, i], RF[1, i]))

    X = np.unique(RF[0, :])
    #xdiff 
    # plot the hexgrid edges
    xdiff = X[1] - X[0]  # midpoitns x-difference 
    edges = get_hexgrid_edges(RF, xdiff)
    for i_, edge in enumerate(edges):

        ax.plot((edge[0][0], edge[0][1]), (edge[1][0], edge[1][1]), c='k')

    pylab.show()
