#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 09:39:52 2021

@author: hiroyasu
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import axes3d, Axes3D

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rc('font',**{'family':'serif','serif':['Times']})
matplotlib.rc('text',usetex=True)
DPI = 300


# Distance vtw label and label name
from matplotlib import rcParams
rcParams['axes.labelpad'] = 15

dover = 1.0
thiss = np.load("data/leo/sim/27samples/thiss_"+str(dover*10)+".npy")
dthiss = np.load("data/leo/sim/27samples/dthiss_"+str(dover*10)+".npy")
Xshiss = np.load("data/leo/sim/27samples/Xshiss_"+str(dover*10)+".npy")
Ushiss = np.load("data/leo/sim/27samples/Ushiss_"+str(dover*10)+".npy")
ueffss = np.load("data/leo/sim/27samples/ueffss_"+str(dover*10)+".npy")
idx_fins = np.load("data/leo/sim/27samples/idx_fins_"+str(dover*10)+".npy")
Nendss = np.load("data/leo/sim/27samples/Nendss_"+str(dover*10)+".npy")
x0ss = np.load("data/leo/sim/27samples/x0ss_"+str(dover*10)+".npy")
xfss = np.load("data/leo/sim/27samples/xfss_"+str(dover*10)+".npy")

thiss1 = np.load("data/leo/sim/23samples/thiss_"+str(dover*10)+".npy")
dthiss1 = np.load("data/leo/sim/23samples/dthiss_"+str(dover*10)+".npy")
Xshiss1 = np.load("data/leo/sim/23samples/Xshiss_"+str(dover*10)+".npy")
Ushiss1 = np.load("data/leo/sim/23samples/Ushiss_"+str(dover*10)+".npy")
ueffss1 = np.load("data/leo/sim/23samples/ueffss_"+str(dover*10)+".npy")
idx_fins1 = np.load("data/leo/sim/23samples/idx_fins_"+str(dover*10)+".npy")
Nendss1 = np.load("data/leo/sim/23samples/Nendss_"+str(dover*10)+".npy")
x0ss1 = np.load("data/leo/sim/23samples/x0ss_"+str(dover*10)+".npy")
xfss1 = np.load("data/leo/sim/23samples/xfss_"+str(dover*10)+".npy")

thiss = np.concatenate((thiss,thiss1),0)
dthiss = np.concatenate((dthiss,dthiss1),0)
Xshiss = np.concatenate((Xshiss,Xshiss1),0)
Ushiss = np.concatenate((Ushiss,Ushiss1),0)
ueffss = np.concatenate((ueffss,ueffss1),0)
idx_fins = np.concatenate((idx_fins,idx_fins1),0)
Nendss = np.concatenate((Nendss,Nendss1),0)
x0ss = np.concatenate((x0ss,x0ss1),0)
xfss = np.concatenate((xfss,xfss1),0)

"""
thiss = np.load("data/leo/sim/50samples/thiss_"+str(dover*10)+".npy")
dthiss = np.load("data/leo/sim/50samples/dthiss_"+str(dover*10)+".npy")
Xshiss = np.load("data/leo/sim/50samples/Xshiss_"+str(dover*10)+".npy")
Ushiss = np.load("data/leo/sim/50samples/Ushiss_"+str(dover*10)+".npy")
ueffss = np.load("data/leo/sim/50samples/ueffss_"+str(dover*10)+".npy")
idx_fins = np.load("data/leo/50samples/sim/idx_fins_"+str(dover*10)+".npy")
Nendss = np.load("data/leo/50samples/sim/Nendss_"+str(dover*10)+".npy")
x0ss = np.load("data/leo/50samples/sim/x0ss_"+str(dover*10)+".npy")
xfss = np.load("data/leo/50samples/sim/xfss_"+str(dover*10)+".npy")
"""

Ndata = thiss.shape[0]
Ncon = ueffss.shape[1]
Nsc = idx_fins.shape[2]
ueffss_mpc = []
ueffss_mpc2 = []
Nends2 = [150*2,150*2,150,150*4]
for i in range(Ndata):
    Ushis = Ushiss[i,:,:,:,:]
    Nends = Nendss[i,:,:]
    idx_fin = idx_fins[i,:,:]
    ueffs = []
    ueffs2 = []
    for c in range(Ncon):
        ueffc = 0
        ueffc2 = 0
        for p in range(Nsc):
            Nend = int(Nends[c,p])
            Nend2 = Nends2[c]
            ueffc2 += np.sum(Ushis[0:Nend2,c,p,:]**2)
            if idx_fin[c,p] == 1:
                #ueffc += np.sum(np.sqrt(np.sum(Ushis[0:Nend,c,p,:]**2,1)))
                ueffc += np.sum(Ushis[0:Nend,c,p,:]**2)
        ueffs.append(ueffc)
        ueffs2.append(ueffc2)
    ueffss_mpc.append(np.array(ueffs))
    ueffss_mpc2.append(np.array(ueffs2))
ueffss_mpc = np.array(ueffss_mpc)
ueffss_mpc2 = np.array(ueffss_mpc2)

success_rate = np.sum(idx_fins,(0,2))/Nsc/Ndata
ueff_ave = np.sum(ueffss,0)/Ndata
ueff_trunc_ave = np.sum(ueffss_mpc,0)/np.sum(idx_fins,(0,2))
ueff_trunc_ave2 = np.sum(ueffss_mpc2,0)/Ndata
dts = np.sum(np.sum(dthiss,1)/Nendss,(0,2))/Nsc/Ndata
ueff_ave_per = ueff_ave/ueff_ave[3]*100

c_caltech_o = [255/256,108/256,12/256]
c_caltech_g = [0/256,88/256,80/256]
c_caltech_b = [0/256,59/256,76/256]
c_caltech_o2 = [249/256,190/256,0/256]
LabelSize = 30
nticks_acc = 30


idx_data = 0
"""
Xshis = Xshiss[idx_data,:,:,:]
Nends = Nendss[idx_data,:,:]
x0s = x0ss[idx_data,:,:]
xfs = xfss[idx_data,:,:]
idx_fin = idx_fins[idx_data,:,:]
"""

Xshis = np.load("data/leo/sim/plot/Xshiss.npy")
Nends = np.load("data/leo/sim/plot/Nendss.npy")
x0s = np.load("data/leo/sim/plot/x0ss.npy")
xfs = np.load("data/leo/sim/plot/xfss.npy")
idx_fin = np.load("data/leo/sim/plot/idx_fins.npy")
fig = plt.figure(figsize=(34,8.9))
fig.subplots_adjust(wspace=0.03)

titles = [r"(a) Learning-based feedforward",r"(b) Robust MPC",r"(c) LAG-ROS",r"(d) Global solution"]
xyz_max = 5.5
xyz_min = -0.5
Xshis = np.where((Xshis <= xyz_max) & (Xshis >= xyz_min),Xshis,np.nan)
for c in range(Ncon):
    if c == 0:
        c_con = 0
    elif c == 1:
        c_con = 2
    elif c == 2:
        c_con = 1
    elif c == 3:
        c_con = 3
    ax = fig.add_subplot(1,Ncon,c+1,projection='3d')
    Xhisc = Xshis[:,c_con,:,:]
    thp = np.linspace(0,2*np.pi,100)
    a1 = ax.scatter(100,100,100,fc="w",ec="k",marker="o",s=100,alpha=0.7,label=r"start")
    a2 = ax.scatter(100,100,100,color="k",s=100,alpha=0.7,label=r"goal")
    if c == 2:
        ax.legend(handles=[a1,a2],ncol=2,loc="upper center",fontsize=28,bbox_to_anchor=(-0.05,0.93))
    for p in range(Nsc):
        colorp = "C"+str(p)
        Nend = int(Nends[c_con,p])
        if c == 0:
            Nend = 350
            ax.plot(Xhisc[0:Nend,p,0],Xhisc[0:Nend,p,1],Xhisc[0:Nend,p,2],color=colorp,lw=3,alpha=0.5,ls="--")
        else:
            ax.plot(Xhisc[0:Nend+1,p,0],Xhisc[0:Nend+1,p,1],Xhisc[0:Nend+1,p,2],color=colorp,lw=3,alpha=0.5,ls="--")
        ax.scatter(x0s[p,:][0],x0s[p,:][1],x0s[p,:][2],fc=(0,0,0,0),ec=colorp,marker="o",s=100)
        ax.scatter(xfs[p,:][0],xfs[p,:][1],xfs[p,:][2],color=colorp,s=100)
        #plt.scatter(Xhisc[Nend,p,0],Xhisc[Nend,p,1],color=colorp,s=100)
    #plt.legend(loc="upper left",fontsize=13)
    ax.set_xlim(-0.5,5.5)
    ax.set_ylim(-0.5,5.5)
    ax.set_zlim(-0.5,5.5)
    ax.xaxis.set_label_coords(2.1,-0.13)
    ax.grid()
    ax.tick_params(labelsize=nticks_acc)
    ax.set_title(titles[c],fontsize=LabelSize)
    ax.set_xlabel(r"$x$ (km)",fontsize=LabelSize)
    ax.set_ylabel(r"$y$ (km)",fontsize=LabelSize)
    ax.set_zlabel(r"$z$ (km)",fontsize=LabelSize)
    #ax.axes.xaxis.set_visible(False)
    #ax.axes.yaxis.set_visible(False)
    #plt.grid()
fname = "figs/leo.pdf"
fig.savefig(fname,bbox_inches='tight',pad_inches=0.2,dpi=DPI)
plt.show()

dovers = [0.2,0.4,0.6,0.8,1.0]

success_rates = []
ueff_aves = []
ueff_trunc_aves = []
ueff_trunc_ave2s = []

thissl = []
dthissl = []
Xshissl = []
Ushissl = []
ueffssl = []
idx_finsl = []
Nendssl = []
x0ssl = []
xfssl = []
for dover in dovers:
    for k in range(10):
        dthissk = np.load("data/leo/sim/dthiss_"+str(k)+str(dover*10)+".npy")
        Xshissk = np.load("data/leo/sim/Xshiss_"+str(k)+str(dover*10)+".npy")
        Ushissk = np.load("data/leo/sim/Ushiss_"+str(k)+str(dover*10)+".npy")
        ueffssk = np.load("data/leo/sim/ueffss_"+str(k)+str(dover*10)+".npy")
        idx_finsk = np.load("data/leo/sim/idx_fins_"+str(k)+str(dover*10)+".npy")
        Nendssk = np.load("data/leo/sim/Nendss_"+str(k)+str(dover*10)+".npy")
        x0ssk = np.load("data/leo/sim/x0ss_"+str(k)+str(dover*10)+".npy")
        xfssk = np.load("data/leo/sim/xfss_"+str(k)+str(dover*10)+".npy")
        if (k == 0) and (dover == 1.0): # when global solution was infeasible
            dthissk = np.delete(dthissk,3,0)
            Xshissk = np.delete(Xshissk,3,0)
            Ushissk = np.delete(Ushissk,3,0)
            ueffssk = np.delete(ueffssk,3,0)
            idx_finsk = np.delete(idx_finsk,3,0)
            Nendssk = np.delete(Nendssk,3,0)
            x0ssk = np.delete(x0ssk,3,0)
            xfssk = np.delete(xfssk,3,0)
            dthiss = dthissk
            Xshiss = Xshissk
            Ushiss = Ushissk
            ueffss = ueffssk
            idx_fins = idx_finsk
            Nendss = Nendssk
            x0ss = x0ssk
            xfss = xfssk
        elif k == 0:
            dthiss = dthissk
            Xshiss = Xshissk
            Ushiss = Ushissk
            ueffss = ueffssk
            idx_fins = idx_finsk
            Nendss = Nendssk
            x0ss = x0ssk
            xfss = xfssk
        else:
            dthiss = np.concatenate((dthiss,dthissk),0)
            Xshiss = np.concatenate((Xshiss,Xshissk),0)
            Ushiss = np.concatenate((Ushiss,Ushissk),0)
            ueffss = np.concatenate((ueffss,ueffssk),0)
            idx_fins = np.concatenate((idx_fins,idx_finsk),0)
            Nendss = np.concatenate((Nendss,Nendssk),0)
            x0ss = np.concatenate((x0ss,x0ssk),0)
            xfss = np.concatenate((xfss,xfssk),0)
    dthissl.append(dthiss)
    Xshissl.append(Xshiss)
    Ushissl.append(Ushiss)
    ueffssl.append(ueffss)
    idx_finsl.append(idx_fins)
    Nendssl.append(Nendss)
    x0ssl.append(x0ss)
    xfssl.append(xfss)

titles = [r"(a)",r"(b)",r"(c)",r"(d)"]    
i_d = 0
for dover in dovers:
    """
    thiss = np.load("data/leo/sim/thiss_"+str(dover*10)+".npy")
    dthiss = np.load("data/leo/sim/dthiss_"+str(dover*10)+".npy")
    Xshiss = np.load("data/leo/sim/Xshiss_"+str(dover*10)+".npy")
    Ushiss = np.load("data/leo/sim/Ushiss_"+str(dover*10)+".npy")
    ueffss = np.load("data/leo/sim/ueffss_"+str(dover*10)+".npy")
    idx_fins = np.load("data/leo/sim/idx_fins_"+str(dover*10)+".npy")
    Nendss = np.load("data/leo/sim/Nendss_"+str(dover*10)+".npy")
    x0ss = np.load("data/leo/sim/x0ss_"+str(dover*10)+".npy")
    xfss = np.load("data/leo/sim/xfss_"+str(dover*10)+".npy")
    """
    dthiss = dthissl[i_d]
    Xshiss = Xshissl[i_d]
    Ushiss = Ushissl[i_d]
    ueffss = ueffssl[i_d]
    idx_fins = idx_finsl[i_d]
    Nendss = Nendssl[i_d]
    x0ss = x0ssl[i_d]
    xfss = xfssl[i_d]
    
    ueffss_mpc = []
    ueffss_mpc2 = []
    Nends2 = [150*2,150*2,150,150*2]
    Ndata = Xshiss.shape[0]
    for i in range(Ndata):
        Ushis = Ushiss[i,:,:,:,:]
        Nends = Nendss[i,:,:]
        idx_fin = idx_fins[i,:,:]
        ueffs = []
        ueffs2 = []
        for c in range(Ncon):
            ueffc = 0
            ueffc2 = 0
            for p in range(Nsc):
                Nend = int(Nends[c,p])
                Nend2 = Nends2[c]
                ueffc2 += np.sum(Ushis[0:Nend2,c,p,:]**2)
                if idx_fin[c,p] == 1:
                    #ueffc += np.sum(np.sqrt(np.sum(Ushis[0:Nend,c,p,:]**2,1)))
                    ueffc += np.sum(Ushis[0:Nend,c,p,:]**2)
            ueffs.append(ueffc)
            ueffs2.append(ueffc2)
        ueffss_mpc.append(np.array(ueffs))
        ueffss_mpc2.append(np.array(ueffs2))
    ueffss_mpc = np.array(ueffss_mpc)
    ueffss_mpc2 = np.array(ueffss_mpc2)
    
    success_rate = np.sum(idx_fins,(0,2))/Nsc/Ndata
    print(dover)
    print(success_rate)
    ueff_ave = np.sum(ueffss,0)/Ndata
    ueff_trunc_ave = np.sum(ueffss_mpc,0)/np.sum(idx_fins,(0,2))
    ueff_trunc_ave2 = np.sum(ueffss_mpc2,0)/Ndata
    dts = np.sum(np.sum(dthiss,1)/Nendss,(0,2))/Nsc/Ndata
    ueff_trunc_ave_per = ueff_trunc_ave2/ueff_trunc_ave2[3]*100
    success_rates.append(success_rate)
    ueff_aves.append(ueff_ave)
    ueff_trunc_aves.append(ueff_trunc_ave)
    ueff_trunc_ave2s.append(ueff_trunc_ave2)
    i_d += 1
success_rates = np.array(success_rates)
ueff_aves = np.array(ueff_aves)
ueff_trunc_aves = np.array(ueff_trunc_aves)
ueff_trunc_ave2s = np.array(ueff_trunc_ave2s)
cp = [0,2,1,3]
LabelSize = 25
leg_size = 20
atra = 0.7
linestyle=[':','-.','--','-']
colors = ['C0','C1','C2','C3']
LW = 3
MS = 7
success_rates[:,3] = np.ones((5))
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(14,4))
fig.subplots_adjust(wspace=0.3)
for c in range(Ncon):
    ax1.plot(np.nan,np.nan,marker="o",color=colors[c],ls=linestyle[c],lw=LW,ms=MS)
ax1.plot(dovers,success_rates[:,3]*100,marker="o",color="C3",ls=linestyle[3],lw=LW,ms=MS)
for c in range(Ncon):
    if c == 2:
        ax1.plot(np.nan,np.nan,marker="o",alpha=atra,ls=linestyle[c],color=colors[c],lw=LW,ms=MS)
    elif c == 3:
        ax1.plot(np.nan,np.nan,marker="o",alpha=atra,ls=linestyle[c],color=colors[c],lw=LW,ms=MS)
    else:
        ax1.plot(dovers,success_rates[:,cp[c]]*100,marker="o",alpha=atra,ls=linestyle[c],color=colors[c],lw=LW,ms=MS)
ax1.plot(dovers,success_rates[:,1]*100,marker="o",color="C2",ls=linestyle[2],lw=LW,ms=MS)
ax1.set_xticks(dovers)
ax1.set_xlabel(r"$\sup_{x,t}\|d(x,t)\|$",fontsize=LabelSize)
ax1.set_ylabel(r"success rate (\%)",fontsize=LabelSize)
ax1.tick_params(labelsize=nticks_acc)
ax1.legend(titles,fontsize=leg_size,loc="lower left")
ax1.grid()
ax2.plot(dovers,ueff_aves[:,3],marker="o",color="C3",ls=linestyle[3],lw=LW,ms=MS)
for c in range(Ncon):
    if c == 2:
        ax2.plot(np.nan,np.nan,marker="o",alpha=atra,ls=linestyle[c],lw=LW,ms=MS)
    elif c == 3:
        ax2.plot(np.nan,np.nan,marker="o",alpha=atra,ls=linestyle[c],lw=LW,ms=MS)
    else:
        ax2.plot(dovers,ueff_aves[:,cp[c]],marker="o",alpha=atra,ls=linestyle[c],lw=LW,ms=MS)
ax2.plot(dovers,ueff_aves[:,1],marker="o",color="C2",ls=linestyle[2],lw=LW,ms=MS)
ax2.set_xticks(dovers)
ax2.set_xlabel(r"$\sup_{x,t}\|d(x,t)\|$",fontsize=LabelSize)
ax2.set_ylabel(r"$\mathcal{L}2$ control effort",fontsize=LabelSize)
ax2.tick_params(labelsize=nticks_acc)
ax2.grid()
fname = "figs/leo_dist.pdf"
plt.savefig(fname,bbox_inches='tight',dpi=DPI)
plt.show()
print(success_rates*100)
print(ueff_aves.T/ueff_aves[:,3]*100)
