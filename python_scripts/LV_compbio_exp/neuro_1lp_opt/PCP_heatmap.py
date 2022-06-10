#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:54:27 2022

@author: mkaratas
"""




path= r"/Users/mkaratas/Desktop/GitHub/BDEtools/code"
eng.addpath(path,nargout= 0)
path= r"/Users/mkaratas/Desktop/GitHub/BDEtools/unit_tests"
eng.addpath(path,nargout= 0)
path= r"/Users/mkaratas/Desktop/GitHub/BDE-modelling/Cost_functions"
eng.addpath(path,nargout= 0)
path= r"/Users/mkaratas/Desktop/GitHub/BDE-modelling/Cost_functions/neuro1lp_costfcn"
eng.addpath(path,nargout= 0)
path= r"/Users/mkaratas/Desktop/GitHub/BDEtools/models"
eng.addpath(path,nargout= 0)


#%%

# Load data

dataLD = eng.load('dataLD.mat')
dataDD = eng.load('dataDD.mat')
lightForcingLD = eng.load('lightForcingLD.mat')
lightForcingDD = eng.load('lightForcingDD.mat')

#%%

# Convert data to be used by MATLAB

dataLD = dataLD['dataLD']
dataDD = dataDD['dataDD']
lightForcingLD=lightForcingLD['lightForcingLD']
lightForcingDD=lightForcingDD['lightForcingDD']








########################################################################################   PCP for Neuro1loop
modelnmm = "NMMSO"
modelcma = "CMAES"

n = 2
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  ## alttaki de aynisi


gate = gatesm[1]


ub_y = [25,25,12,1.0,1.0]
lb_y = [0.0,0.0,0.0,0.0,0.0]

cols = ['$τ_1$','$τ_2$','$τ_3$','$T_1$','$T_2$']


f = []


for i in globals()[f'f_%s_%s_%s' % (modelnmm,f"neuro1lp",gate)]:
    f.append(i)
f = sorted(f)
norm = colors.Normalize(0,4) #(min(f), max(f)) ...   #### colorbari degistirince burayi da degistir !!
colours = cm.RdBu(norm(f))  # div



t = 'T_1'
e = 'T_2'
p = '\u03C4_1'
c = '\u03C4_2'
s = '\u03C4_3'

T1 =[]
T2 =[]
t1 =[]
t2 =[]
t3 =[]


for i in globals()['x_%s_%s_%s' % (modelnmm,"neuro1lp",gate)]:
    print(i)
    t1.append(i[0])
    t2.append(i[1])
    t3.append(2*i[2])
    T1.append(24*i[3])
    T2.append(24*i[4])
    

data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2,'Fitness': globals()['f_%s_%s_%s' %(modelnmm,"neuro1lp",gate)]} 


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
    host.plot(range(len(ys)), [z[j] for z in ys], c=colours[j])
    
    
    
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=0, vmax=4))#(vmin=min(f), vmax=max(f)))   !!color...
position=fig.add_axes([0.94,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
cb = plt.colorbar(sm,cax=position,pad = 0.9)# pad = 0.15 shrink=0.9 colorbar length
tick_locator = ticker.MaxNLocator(nbins=8)
cb.locator = tick_locator

cb.update_ticks()
cb.ax.tick_params(labelsize=17)
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) 
plt.savefig('Desktop/cont_figs/NMMSO_GG_%s_pcp_neur1loop_naxes.eps'%gate, format='eps',bbox_inches='tight')
plt.show()


#############################################################################################   extracted



# 3 - 18 nmmso



c1=[]
#c2=[]
f_idx = 3
s_idx = 18
n = 2
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  
gate = gatesm[1]
savename = f"{gate}"

gate_1g = globals()['x_%s_%s_%s' %(modelnmm,"neuro1lp",gate)]
gate_1 =[] 
gate_1.append(gate_1g[f_idx]) 
gate_1.append(gate_1g[s_idx])
gate_1g = gate_1
f = globals()['f_%s_%s_%s' %(modelnmm,"neuro1lp",gate)]

fn = []
fn.append(f[f_idx])
fn.append(f[s_idx])
#f = fn


# f = sorted(f)
# norm = colors.Normalize(0, 4) #(min(f), max(f))    #### colorbari degistirince burayi da degistir !!
# colours = cm.RdBu(norm(f))  # div


    
for i in range(len(globals()['f_%s_%s_%s' %(modelnmm,"neuro1lp",gate)])):     
    if globals()['f_%s_%s_%s' %(modelnmm,"neuro1lp",gate)][i] in fn:
        c1.append(colours[f.index(globals()['f_%s_%s_%s' %(modelnmm,"neuro1lp",gate)][i])])   


t = 'T_1'
e = 'T_2'
p = '\u03C4_1'
c = '\u03C4_2'
s = '\u03C4_3'

T1 =[]
T2 =[]
t1 =[]
t2 =[]
t3 =[]


for i in gate_1g:
    print(i)
    t1.append(i[0])
    t2.append(i[1])
    t3.append(2*i[2])
    T1.append(24*i[3])
    T2.append(24*i[4])
    

data = { r'${}$'.format(p): t1, r'${}$'.format(c):t2, r'${}$'.format(s): t3, 
        r'${}$'.format(t): T1 ,r'${}$'.format(e): T2,'Fitness': fn} 


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
    host.plot(range(len(ys)), [z[j] for z in ys], c=c1[j])
    
    
    
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=colors.Normalize(vmin=0, vmax=4))#(vmin=min(f), vmax=max(f)))   !!color...
position=fig.add_axes([0.94,0.125,0.03,0.75]) # first num= distance to axis,2nd = top-botom, 3rd = width , 4th length
cb = plt.colorbar(sm,cax=position,pad = 0.9)# pad = 0.15 shrink=0.9 colorbar length
tick_locator = ticker.MaxNLocator(nbins=8)
cb.locator = tick_locator

cb.update_ticks()
cb.ax.tick_params(labelsize=17)
#plt.title("g = 01")  
matplotlib.rc('xtick', labelsize=20) 
plt.savefig('Desktop/cont_figs/NMMSO_GG_%s_pcp_extr_neur1loop_naxes_3_18.eps'%gate, format='eps',bbox_inches='tight')
plt.show()




## Heatmap for Neuro1loop
########
#######
#########
#############


model = modelcma
model = modelnmm

f_idx = 3
s_idx = 18

n = 2
gatesm = list(map(list, itertools.product([0, 1], repeat=n)))  
gate = gatesm[1]
savename = f"{gate}"

gate_1g = globals()['x_%s_%s_%s' %(model,"neuro1lp",gate)]
gate_1 =[] 
gate_1.append(gate_1g[f_idx]) 
gate_1.append(gate_1g[s_idx])

f = globals()['f_%s_%s_%s' %(model,"neuro1lp",gate)]


class Mat_Py:
    def __init__(self):
        self.cost, self.sol_ld, self.sol_dd, self.dat_ld, self.dat_dd = ([] for _ in range(5))
        
    def update(self, cost, sol_ld, sol_dd, dat_ld, dat_dd):
        self.cost.append(cost)
        self.sol_ld.append({'x':list(sol_ld['x']._data), 'y':np.asarray(sol_ld['y'], dtype=np.longdouble).tolist()})
        self.sol_dd.append({'x':list(sol_dd['x']._data), 'y':np.asarray(sol_dd['y'], dtype=np.longdouble).tolist()})
        self.dat_ld.append({'x':list(dat_ld['x']._data), 'y':np.asarray(dat_ld['y'], dtype=np.longdouble).tolist()})
        self.dat_dd.append({'x':list(sol_dd['x']._data), 'y':np.asarray(dat_dd['y'], dtype=np.longdouble).tolist()})
    
    def get_all(self):
        return [self.cost, self.sol_ld, self.sol_dd, self.dat_ld, self.dat_dd]
         
         



func_output_t = Mat_Py()

t = [[]]
t[0] = gate_1g[f_idx]
t.append(gate_1g[s_idx])


for i in t: 
    print(i)
    gates = list(gate)
    gates = matlab.double(gates)
    inputparams = i
    inputparams = list(inputparams)
    inputparams = matlab.double(inputparams)
    print(inputparams)
    func_output_t.update(*eng.getBoolCost_cts_neuro1lp(inputparams, gates, dataLD, dataDD, lightForcingLD, lightForcingDD,nargout=5))

[cost_t, sol_LD_t, sol_DD_t, datLD_t, datDD_t] = func_output_t.get_all()
    

    

############   24den sonraki hal--- 24 yoksa bir oncekini alsin ama 24 varsa oradan itibaren alsin, cunku data 24den baslayacak

datLD_t[0]['x'] = datLD_t[0]['x'][4:21]
datLD_t[0]['y'][0] = datLD_t[0]['y'][0][4:21]
datLD_t[0]['y'][1] = datLD_t[0]['y'][1][4:21]


datLD_t[1]['x'] = datLD_t[1]['x'][4:21]#[8:42]
datLD_t[1]['y'][0] = datLD_t[1]['y'][0][4:21]#[8:42]
datLD_t[1]['y'][1] = datLD_t[1]['y'][1][4:21]#[8:42]


###################################################  altta fnc yazilmisi var

### alttakini fonk olarak yaz    ------ data 1-data 2 birlestirici fonk yaz !  hem LD icin hem DD icin !
solution = sol_LD_t
data_threshold = datLD_t
data1={}
#data1['x'] = data1['x'] + 3
time, y0, y1 = [[] for i in range(3)]
for i,val in enumerate(solution[0]['x']):
    time.append(val)
    for j,value in enumerate(data_threshold[0]['x']):         
        if val<=value:
            #print(val,value)
            y0.append(data_threshold[0]['y'][0][j-1])
            #print(y0)
            y1.append(data_threshold[0]['y'][1][j-1])
            y = [y0,y1]               
            break
data1['x'] = time
data1['y']= y

### alttakini fonk olarak yaz
data2={}
time, y0, y1= [[] for i in range(3)]
for i,val in enumerate(solution[1]['x']):
    time.append(val)
    for j,value in enumerate(data_threshold[1]['x']):         
        if val<=value:
            #print(val,value)
            y0.append(data_threshold[1]['y'][0][j-1])
            #print(y0)
            y1.append(data_threshold[1]['y'][1][j-1])
            y = [y0,y1]                           
            break
data2['x'] = time
data2['y']= y

light_cbar = [1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]
##############  this is for LD colormap
time_list = np.linspace(24,120,193)    ##  bu tum 24den 120ye kadar olan 0.5 aralikli data
# asagisi tekrar comt out


###  bu tamamen tek satirli T iicn yazildi bunu keep
def time_equalise(time_list,aggregated_data,m=0): # this means the first sol
    y0 = []
    for i,val in enumerate(time_list):
        for j,value in enumerate(aggregated_data['x']):         
            if val<=value:
                y0.append(aggregated_data['y'][j-1])
                y = y0             
                break                 
    while len(time_list)!= len(y0):
        y.append(aggregated_data['y'][-1])
    return y  


light_switch= []
for i in range(4,21) :
    if i == 4:
        light_switch.append(6*i)    
    if i % 2 == 1:
        light_switch.append(6*i)    
light_cbar = {}
light_cbar['x'] = light_switch
light_cbar['y'] = [1,0,1,0,1,0,1,0,1]  
T = time_equalise(time_list,light_cbar,m=0)

#####  Simdi bu asagiis digerleri icin  yazildi

def time_equalise(time_list,aggregated_data,m): # aggregated data time_liste gore ayarlanip degerleri 0-1 diye guncellenecek
    y0,y1= [[] for i in range(2)]
    for i,val in enumerate(time_list):
        for j,value in enumerate(aggregated_data[m]['x']):         
            if val<=value:
                y0.append(float(aggregated_data[m]['y'][0][j-1]))
                y1.append(float(aggregated_data[m]['y'][1][j-1]))
                y = [y0,y1]               
                break                 
    while len(time_list)!= len(y0):
        y[0].append(float(aggregated_data[m]['y'][0][-1]))
        y[1].append(float(aggregated_data[m]['y'][1][-1])  )
    return y      



##################



cb = []
for i in range(len(T)):
    cb.append(np.float(T[i]))
    
# plt.plot(datLD_t[0]['x'],datLD_t[0]['y'][0])
# plt.plot(time_list,first_data[0])


aggregated_data = datLD_t   ##  aydinlik 2 datali x-y dictionarysi
first_data = time_equalise(time_list,aggregated_data,0)   ##   time_equalise function ile bu light data cogaltiliyor sonuc 193 datali olmaoi
second_data = time_equalise(time_list,aggregated_data,1)   

solution = sol_LD_t
first_solution = time_equalise(time_list,solution,0)   
second_solution = time_equalise(time_list,solution,1)   
  
harvest = np.empty([9,len(time_list)],dtype=object)
harvest[0] = cb
harvest[1] = first_data[0]
harvest[2] = first_solution[0]
harvest[3] = second_data[0]
harvest[4] = second_solution[0]
harvest[5] = first_data[1]
harvest[6] = first_solution[1]
harvest[7] = second_data[1]
harvest[8] = second_solution[1]




y_axis_labels = ['Light regime(LD/DD)','$mRNA$ data', '$mRNA$ prediction','$mRNA$ data', '$mRNA$ prediction', 
                 '$Protein_{bulk}$ data', '$Protein_{bulk}$ prediction','$Protein_{bulk}$ data', '$Protein_{bulk}$ prediction',
                 ] # labels for x-axis
cols = time_list

df = pd.DataFrame(harvest, columns=cols,index = y_axis_labels,dtype=float)


plt.figure()
matplotlib.rc('figure', figsize=(20, 9))

sns.set(font_scale=1.9)
s = sns.heatmap(df,yticklabels=True,xticklabels=True,cmap='Reds',cbar = False)  ### Reds, Greys,Blues,PuRd

plt.xticks(rotation=0)
for ind, label in enumerate(s.get_xticklabels()):
    if ind % 16 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)    
plt.tick_params(axis=u'both', which=u'both',length=8,color='black',bottom=False,left=True) 
plt.locator_params(axis='x', nbins=200)  
#plt.savefig('Desktop/cont_figs/%s_heatmap_LD_%s_%s_%s_new_1loop.eps'%(model,gate,f_idx,s_idx), format='eps',bbox_inches='tight')
#plt.tick_params(axis=u'both',length=15,color='black',bottom=True,left=True)                                               
plt.tight_layout() 

























## PCP for Neuro2loop