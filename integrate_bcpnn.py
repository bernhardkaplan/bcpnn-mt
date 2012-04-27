import numpy

n = 3000
si = numpy.zeros(n)
sj = numpy.zeros(n)
t1 = 200
delta_t = 100
spike_width = 50
t2 = t1 + delta_t
f_max = 1.0
si[t1:t1+spike_width] = f_max
sj[t2:t2+spike_width] = f_max

zi = numpy.zeros(n)
zj = numpy.zeros(n)
pi = numpy.zeros(n)
pj = numpy.zeros(n)
pij = numpy.zeros(n)

tau_z = 300
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
ax = fig.add_subplot(111)
ax.plot(t, si, label='si', lw=2)
ax.plot(t, sj, label='sj', lw=2)
ax.plot(t, zi, label='zi', lw=2)
ax.plot(t, zj, label='zi', lw=2)
ax.plot(t, zi * zj, label='zi * zj', lw=2)
ax.plot(t, pi, label='pi', lw=2)
ax.plot(t, pj, label='pj', lw=2)
ax.plot(t, pi * pj, label='pi * pj', lw=2)
ax.plot(t, pij, label='pij', lw=2)
#ax.set_title("s and z traces")
ax.legend()

print "Calculating weights and bias"
wij = numpy.zeros(n)
bias = numpy.zeros(n)
for i in xrange(n):
    if (pi[i] < eps or pj[i] < eps):
        wij[i] = 0.
    elif (pij[i] <= eps**2 or pi[i] * pj[i] <= eps**2):
    #elif (pij[i] <= pi[i] * pj[i]): # this condition avoids weights going negative
        wij[i] = eps**2
    else:
        wij[i] = numpy.log(pij[i] / (pi[i] * pj[i]))

    if pj[i] > eps**2:
        bias[i] = numpy.log(pj[i])
#            bias[i] = pj[i]
    else:
        bias[i] = numpy.log(1/eps**2)

fig = pylab.figure()
ax = fig.add_subplot(111)
ax.plot(t, wij, label='w_ij', lw=2)

fig = pylab.figure()
ax.legend()

ax = fig.add_subplot(111)
ax.plot(t, bias, label='bias', lw=2)
ax.plot(t, numpy.exp(bias), label='exp bias', lw=2)
ax.legend()
pylab.show()
