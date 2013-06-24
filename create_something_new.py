

import sys

folder_name = sys.argv[1]

output_fn = folder_name + '/Data/something_new.dat'
print 'I will create something new from this folder:', output_fn

import numpy as np
np.savetxt(output_fn, np.random.randint(0, 100, 10))


# do analysis




