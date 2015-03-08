import numpy as np
import pylab

#fn = "TestModel_IIII_scaleLatency0.15_wsigmax1.00e-01_wsigmav1.00e-01_wee3.00e-02_wei4.00e-02_wie7.00e-02_wii1.00e-02_delayScale20/Connections/merged_distances_ee.dat"
#fn = "/home/bernhard/workspace/bcpnn-mt/LargeScaleModel_AIII_fmaxstim1.00e+03_scaleLatency0.15_tbb400/Connections/merged_conn_list_ee.dat"
#fn = "/home/bernhard/workspace/bcpnn-mt/LargeScaleModel_AIII_fmaxstim1.00e+03_scaleLatency0.15_tbb400/Parameters/tuning_prop_means.prm"
fn = "SmallScale_AIII_fmaxstim1.00e+03_scaleLatency0.15_tbb400/Parameters/tuning_prop_means.prm"

tp = np.loadtxt(fn)
import utils            
n = tp[:, 0].size
distances = np.zeros((n**2 - n) / 2)
cnt = 0
for i in xrange(n):
    x_i, y_i = tp[i, 0], tp[i, 1]  
    print i
    for j in xrange(i):
        x_j, y_j = tp[j, 0], tp[j, 1]
        distances[cnt] = utils.torus_distance2D(x_i, x_j, y_i, y_j)
        cnt += 1
        
count, bins = np.histogram(distances, bins=100)
pylab.plot(bins[:-1], count)
pylab.xlabel('distances between pairs of cells')
pylab.ylabel('num occurences')
output_fn = 'distances_between_cells.dat'
np.savetxt(output_fn, distances)
print 'saving to:', output_fn 
pylab.show()
