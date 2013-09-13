import os
import numpy

class MergeSpikefiles(object):

    def __init__(self, params):
        self.params = params


    def merge_files(self, merge_pattern, fn_out):
        rnd_nr1 = numpy.random.randint(0,10**8)
        rnd_nr2 = rnd_nr1 + 1
        fn_out = sorted_pattern_output + ".dat"
        # merge files from different processors
        tmp_file = "tmp_%d" % (rnd_nr2)
        os.system("cat %s_* > %s" % (merge_pattern, tmp_file))
        # sort according to cell id
        os.system("sort -gk 1 %s > %s" % (tmp_file, fn_out))
        os.system("rm %s" % (tmp_file))
