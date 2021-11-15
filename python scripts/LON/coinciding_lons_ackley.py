# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 12:33:15 2021

@author: mk633
"""
####

#%%
nm_nodes = []
nm_fvals = []



#%%  yukle - ekle - kaydet         NM

import pickle

with open ('C:/Users/mk633/Desktop/code/nm_n', 'rb') as fp:
    nm_nodes= pickle.load(fp)  
    
with open ('C:/Users/mk633/Desktop/code/nm_f', 'rb') as fp:
    nm_fvals= pickle.load(fp) 


nm_nodes.append(nodes_list)
nm_fvals.append(fvals_list)


with open('C:/Users/mk633/Desktop/code/nm_n', 'wb') as fp:
    pickle.dump(nm_nodes, fp)
    
with open('C:/Users/mk633/Desktop/code/nm_f', 'wb') as fp:
    pickle.dump(nm_fvals, fp)    
    
    
#%%
bh_nodes = []
bh_fvals = []    


#%%  yukle - ekle - kaydet        BH


 with open ('C:/Users/mk633/Desktop/code/bh_n', 'rb') as fp:
    bh_nodes= pickle.load(fp)  
    
with open ('C:/Users/mk633/Desktop/code/bh_f', 'rb') as fp:
    bh_fvals= pickle.load(fp) 


bh_nodes.append(nodes_list)
bh_fvals.append(fvals_list)


with open('C:/Users/mk633/Desktop/code/bh_n', 'wb') as fp:
    pickle.dump(bh_nodes, fp)
    
with open('C:/Users/mk633/Desktop/code/bh_f', 'wb') as fp:
    pickle.dump(bh_fvals, fp)    



# %%

for i in bh_nodes


import matplotlib.pyplot as plt


plt.scatter(bh_fvals, nm_fvals, s=8, c='r', marker="o", label='NM')
plt.xlabel("BH")
plt.ylabel("NM")
#m, b = np.polyfit(bh_fvals, nm_fvals, 1)
plt.plot([0, 20], [0, 20], color = 'red', linewidth = 2)

#plt.plot(bh_fvals, m*bh_fvals + b)
plt.show()





import matplotlib.pyplot as plt


fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(x[:4], y[:4], s=10, c='b', marker="s", label='NM')
ax1.scatter(x[40:],y[40:], s=10, c='r', marker="o", label='BH')
plt.legend(loc='upper left');
plt.show()

















       