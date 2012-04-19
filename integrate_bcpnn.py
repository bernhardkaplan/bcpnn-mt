import numpy

n = 3000
si = numpy.zeros(n)
sj = numpy.zeros(n)
si[200:250] = 1.
sj[300:350] = 1.

zi = numpy.zeros(n)
zj = numpy.zeros(n)
pi = numpy.zeros(n)
pj = numpy.zeros(n)
pij = numpy.zeros(n)

tau_z = 100
tau_p = 500
eps = 1e-2
dt = 1
print "Integrating traces"
for i in xrange(1, n):
    # pre-synaptic trace zi follows si
    dzi = dt * (eps + si[i] - zi[i-1]) / tau_z
    zi[i] = zi[i-1] + dzi

    # post-synaptic trace zj follows sj
    dzj = dt * (eps + sj[i] - zj[i-1]) / tau_z
    zj[i] = zj[i-1] + dzj

    # pre-synaptic probability pi follows zi
    dpi = dt * (eps + zi[i] - pi[i-1]) / tau_p
    pi[i] = pi[i-1] + dpi

    # post-synaptic probability pj follows zj
    dpj = dt * (eps + zj[i] - pj[i-1]) / tau_p
    pj[i] = pj[i-1] + dpj

    # joint probability pij follows zi * zj
    dpij = dt * (eps**2 + zj[i] * zi[i] - pij[i-1]) / tau_p
    pij[i] = pij[i-1] + dpij

print "Plotting"
t = numpy.arange(0, n, dt)
import pylab
pylab.rcParams.update({'legend.fontsize' : 10})
fig = pylab.figure()
ax = fig.add_subplot(311)
ax.plot(t, si, label='si')
ax.plot(t, sj, label='sj')
ax.plot(t, zi, label='zi')
ax.plot(t, zj, label='zi')
ax.plot(t, zi * zj, label='zi * zj')
ax.plot(t, pi, label='pi')
ax.plot(t, pj, label='pj')
ax.plot(t, pi * pj, label='pi * pj')
ax.plot(t, pij, 'o-', label='pij')
#ax.set_title("s and z traces")
ax.legend()

print "Calculating weights and bias"
wij = numpy.zeros(n)
bias = numpy.zeros(n)
for i in xrange(n):
    if (pi[i] < eps or pj[i] < eps):
        print i, "1"
        wij[i] = 0.
    #elif (pij[i] <= eps**2):
    elif (pij[i] <= eps**2 or pi[i] * pj[i] <= eps**2):
    #elif (pij[i] <= pi[i] * pj[i]): # this condition avoids weights going negative
        print i, "2"
        wij[i] = eps**2
    else:
        print i, "3"
        wij[i] = numpy.log(pij[i] / (pi[i] * pj[i]))

    if pj[i] > eps**2:
        bias[i] = numpy.log(pj[i])
    else:
        bias[i] = numpy.log(1/eps**2)

ax = fig.add_subplot(312)
ax.plot(t, wij, label='w_ij')

ax.legend()
ax = fig.add_subplot(313)
ax.plot(t, bias, label='bias')
ax.legend()
pylab.show()
