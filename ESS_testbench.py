import time
import numpy as np
import sys
import os
import json
import CreateConnections as CC
import utils
import simulation_parameters_ESS as simulation_parameters

import pyNN
import pyNN.hardware.brainscales as sim
import pyNN.space as space
from pyNN.utility import Timer # for measuring the times to connect etc.
print 'pyNN.version: ', pyNN.__version__
