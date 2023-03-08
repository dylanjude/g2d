
import pickle, sys
from g2d.python import postprocess_g2d, c81utils
# all_outputs = postprocess_g2d.collect_g2d_outputs('results','opt_design_0')

# output c81 format file: done automatically
filename        = sys.argv[1]
c81_data        = c81utils.read_c81_file(filename)
 
# plot the airfoil tables
c81utils.plot_c81(c81_data,filename.rstrip('.c81'),separate_plots=True)
