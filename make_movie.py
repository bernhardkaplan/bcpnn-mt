import os
import sys

def avconv(input_fn, output_fn_movie, fps=.5):
#    command = "avconv -f image2 -r %f -i %s -b 72000 %s" % (fps, input_fn, output_fn_movie)
    command = "avconv -f image2 -r %f -i %s %s" % (fps, input_fn, output_fn_movie)
    os.system("rm %s" % output_fn_movie) # remove old one
    print command
    os.system(command)
    print 'Output movie:', output_fn_movie

if __name__ == '__main__':
    if len(sys.argv) < 2:
        input_fn = raw_input('Give input filename: e.g. folder/fig_%.png\n')
    else:
        input_fn = sys.argv[1]

    if len(sys.argv) < 3:
        output_fn_movie= raw_input('Give output filename\n')
    else:
        output_fn_movie= sys.argv[2]

