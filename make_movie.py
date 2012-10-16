import os
import sys

if len(sys.argv) < 2:
    input_fn = raw_input('Give input filename: e.g. folder/fig_%.png\n')
else:
    input_fn = sys.argv[1]

if len(sys.argv) < 3:
    output_fn_movie= raw_input('Give output filename\n')
else:
    output_fn_movie= sys.argv[2]

#output_fn_movie = 'prediction.mp4'
fps = 2 # frames per second
command = "avconv -f image2 -r %f -i %s -b 72000 %s" % (fps, input_fn, output_fn_movie)
os.system("rm %s" % output_fn_movie) # remove old one
print 'Creating the movie in file:', output_fn_movie
os.system(command)

