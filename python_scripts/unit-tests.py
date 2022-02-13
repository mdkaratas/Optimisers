# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

#%%

import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab()

#%%

path= r"/Users/melikedila/Documents/Repos/BDEtools/code"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/Repos/BDEtools/models"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/Repos/BDEtools/unit_tests"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/Repos/BDE-modelling/Cost functions"
eng.addpath(path,nargout= 0)
path= r"/Users/melikedila/Documents/Repos/BDE-modelling/Cost functions/example"
eng.addpath(path,nargout= 0)



#%%   
#                  UNIT TESTS

# neuro1lp_test
neuro1lp_out= eng.neuro1lp_test(nargout=3)
solerr = neuro1lp_out[0]
solcell = neuro1lp_out[1]
solSercell = neuro1lp_out[2]


#%%

# neuro1lp_test_wplot
neuro1lp_out_wplot= eng.neuro1lp_test_wplot(nargout=3)
solerr = neuro1lp_out_wplot[0]
solcell = neuro1lp_out_wplot[1]
solSercell = neuro1lp_out_wplot[2]




#%%

# neuro2lp_test

neuro2lp_out= eng.neuro2lp_test(nargout=3)
solerr = neuro2lp_out[0]
solcell = neuro2lp_out[1]
solSercell = neuro2lp_out[2]

#%%

# neuro2lp_test_wplot

neuro2lp_out_wplot= eng.neuro2lp_test_wplot(nargout=3)
solerr = neuro2lp_out_wplot[0]
solcell = neuro2lp_out_wplot[1]
solSercell = neuro2lp_out_wplot[2]

#%%

# arabid2lp_test

arabid2lp_out= eng.arabid2lp_test(nargout=3)
solerr = arabid2lp_out[0]
solcell = arabid2lp_out[1]
solSercell = arabid2lp_out[2]

#%%

# arabid2lp_test_wplot

arabid2lp_out_wplot= eng.arabid2lp_test_wplot(nargout=3)
solerr = arabid2lp_out_wplot[0]
solcell = arabid2lp_out_wplot[1]
solSercell = arabid2lp_out_wplot[2]

#%%

# arabid3lp_test

arabid3lp_out= eng.arabid3lp_test(nargout=3)
solerr = arabid3lp_out[0]
solcell = arabid3lp_out[1]
solSercell = arabid3lp_out[2]

#%%

# arabid3lp_test_wplot

arabid3lp_out_wplot= eng.arabid3lp_test_wplot(nargout=3)
solerr = arabid2lp_out_wplot[0]
solcell = arabid2lp_out_wplot[1]
solSercell = arabid2lp_out_wplot[2]

#%%

# chaotic_mod_test

chaotic_mod_out= eng.chaotic_mod_test(nargout=3)
solerr = chaotic_mod_out[0]
solcell = chaotic_mod_out[1]
solSercell = chaotic_mod_out[2]

#%%

# chaotic_mod_test_wplot

chaotic_mod_out_wplot= eng.chaotic_mod_test_wplot(nargout=3)
solerr = chaotic_mod_out_wplot[0]
solcell = chaotic_mod_out_wplot[1]
solSercell = chaotic_mod_out_wplot[2]

#%%
circ = eng.circadian_example_full('stochastic',nargout=0)



