#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 13:02:00 2022

@author: mkaratas
"""


read_root = root + "Optimisers/Llyonesse/continuous_fcn_results/"
model_list = {"neuro1lp":2}
optimisers = ["CMAES","NMMSO"]
#optimisers = ["NMMSO"]


for model, n_gates in model_list.items():
    for opt in optimisers:
        with open(read_root + model + "/x_%s_MI_%s_cts.txt"% (opt,model), "rb") as fp:   
            globals()['x_%s_MI_%s' % (opt,model)] = pickle.load(fp)   
        with open(read_root + model + "/f_%s_MI_%s_cts.txt"% (opt,model), "rb") as fp:   
            globals()['f_%s_MI_%s' % (opt,model)] = pickle.load(fp) 
        # if opt =="NMMSO":
        #     with open(read_root + model + "/fit_dict_%s_MI_%s_cts.txt"% (opt,model), "rb") as fp:   
        #         globals()['fit_dict_MI_%s_%s' % (opt,model)] = pickle.load(fp)   
        #     with open(read_root + model + "/design_dict_%s_MI_%s_cts.txt"% (opt,model), "rb") as fp:   
        #         globals()['design_dict_MI_%s_%s' % (opt,model)] = pickle.load(fp)           
        gatesm = list(map(list, itertools.product([0, 1], repeat=n_gates)))      
        for gate in gatesm: #[3:5]
            with open(read_root + model + f"/x_%s_%s_%s_cts.txt"%(opt,gate,model), "rb") as fp:   
                globals()[f'x_%s_%s_%s' % (opt,model,gate)] = pickle.load(fp) 
            with open(read_root + model + f"/f_%s_%s_%s_cts.txt"%(opt,gate,model), "rb") as fp:   
                globals()[f'f_%s_%s_%s' % (opt,model,gate)] = pickle.load(fp)
            if opt == "NMMSO":
                with open(read_root + model + "/fit_dict_%s_%s_%s_cts.txt"% (gate,opt,model), "rb") as fp:   
                    globals()['fit_dict_%s_%s_%s' % (gate,opt,model)] = pickle.load(fp)   
                with open(read_root + model + "/design_dict_%s_%s_%s_cts.txt"% (gate,opt,model), "rb") as fp:   
                    globals()['design_dict_%s_%s_%s' % (gate,opt,model)] = pickle.load(fp) 








modelcma = "CMAES"
modelnmm = "NMMSO"

c1=[]
c2=[]

n = 2
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  
gate = gatesm[1]
savename = f"{gate}"


gate_1g = globals()[f'x_%s_%s_%s' % (modelcma,f"neuro1lp",gate)]
f = globals()[f'f_%s_%s_%s' % (modelcma,f"neuro1lp",gate)]

f = []

for i in globals()[f'f_%s_%s_%s' % (modelcma,f"neuro1lp",gate)]:
    f.append(i)
for i in globals()[f'f_%s_%s_%s' % (modelnmm,f"neuro1lp",gate)]:
    f.append(i)
f = sorted(f)
norm = colors.Normalize(0,8) #(min(f), max(f)) ...   #### colorbari degistirince burayi da degistir !!
colours = cm.RdBu(norm(f))  # div
    

        
t = 'T_1'
e = 'T_2'
p = '\u03C4_1'
c = '\u03C4_2'
s = '\u03C4_3'        
        
        
n = 2
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  
gate = gatesm[1]
savename = f"{gate}"

###  plot  CMAES


ub_y = [0,0,0,0,0,0,0]
lb_y = [24,24,12,1,1,1,1]

cols = ['$τ_1$','$τ_2$','$τ_3$','$T_1$','$T_2$']


ub = {'$τ_1$':24,'$τ_2$' : 24,'$τ_3$' : 12, '$T_1$' :1,'$T_2$' : 1}
lb = {'$τ_1$':0,'$τ_2$' : 0,'$τ_3$' : 0, '$T_1$' :0,'$T_2$' : 0}



T1 =[]
T2 =[]
t1 =[]
t2 =[]
t3 =[]


for i in globals()['x_%s_%s_%s' % (modelcma,"neuro1lp",gate)]:
    T1.append(i[0])
    T2.append(i[1])
    t1.append(i[2])
    t2.append(i[3])
    t3.append(i[4])

    
data = { r'${}$'.format(t): T1, r'${}$'.format(e):T2, r'${}$'.format(p): t1, 
        r'${}$'.format(c): t2 ,r'${}$'.format(s): t3, 'Fitness': globals()[f'f_%s_%s_%s' %(modelcma,"neuro1lp",gate)]} 


df = pd.DataFrame(data)


fig, host = plt.subplots(1, sharey=False, figsize=(14,8))
#fig, host = plt.subplots()
fig.text(0.08, 0.5, 'Parameter values', va='center', rotation='vertical', fontsize = 30)


ys = []
for col in cols:
    print(col)
    ys.append(np.array(df[col]))

y0min = ys[0].min()
dy = ys[0].max() - y0min
#zs = [ys[0]] + [(y - y.min()) / (y.max() - y.min()) * dy + y0min for y in ys[1:]]
for k in ys:
    print(k)
#zs = 
ynames = ['$τ_1$','$τ_2$','$τ_3$','$T_1$','$T_2$']

axes = [host] + [host.twinx() for i in range(len(ys) - 1)]
for i, (ax, y) in enumerate(zip(axes, ys)):  ### i = 15   y = 15 tane 30lu cozum
    #ax.set_ylim(y.min(), y.max())
    ax.set_ylim(lb_y[i], ub_y[i])
    ax.spines['top'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    if ax != host:
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_ticks_position('right')
        ax.spines["right"].set_position(("axes", i / (len(ys) - 1)))

host.set_xlim(0, len(ys) - 1)
host.set_xticks(range(len(ys)))
host.set_xticklabels(ynames)
host.tick_params(axis='x', which='major', pad=7)
host.spines['right'].set_visible(False)



for j in range(len(ys[0])):
    #host.plot(range(len(ys)), [z[j] for z in zs], c=rgb[1][j])
    host.plot(range(len(ys)), [z[j] for z in ys], c=rgb[10][j])
    
    
    
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=0, vmax=8))#(vmin=min(f), vmax=max(f)))   !!color...
position=fig.add_axes([0.94,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
cb = plt.colorbar(sm,cax=position,pad = 0.9)# pad = 0.15 shrink=0.9 colorbar length
tick_locator = ticker.MaxNLocator(nbins=8)
cb.locator = tick_locator

cb.update_ticks()
cb.ax.tick_params(labelsize=17)
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) 
#plt.savefig('Desktop/cont_figs/CMAES_GG_%s_pcp_naxes.eps'%gate, format='eps',bbox_inches='tight')
plt.show()
        