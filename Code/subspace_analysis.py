#######################################
######## Finding subspaces for ########
##### PFC spatial representations #####
#######################################

import numpy as np
from analysis_functions import *

Data_folder= r'/Users/AdamHarris/Documents/PFC_Subspaces/Data'
Recording_days=np.load(Data_folder+'Recording_days_combined.npy')
Edge_grid=np.load(Data_folder+'Edge_grid.npy')