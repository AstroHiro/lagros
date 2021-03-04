#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 17:23:40 2020

@author: hiroyasu
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rc('font',**{'family':'serif','serif':['Times']})
matplotlib.rc('text',usetex=True)
DPI = 300


dover = 0.8
thiss = np.load("data/sc_obs/sim/43samples/thiss_"+str(dover*10)+".npy")
dthiss = np.load("data/sc_obs/sim/43samples/dthiss_"+str(dover*10)+".npy")
Xshiss = np.load("data/sc_obs/sim/43samples/Xshiss_"+str(dover*10)+".npy")
Ushiss = np.load("data/sc_obs/sim/43samples/Ushiss_"+str(dover*10)+".npy")
ueffss = np.load("data/sc_obs/sim/43samples/ueffss_"+str(dover*10)+".npy")
idx_fins = np.load("data/sc_obs/sim/43samples/idx_fins_"+str(dover*10)+".npy")
Nendss = np.load("data/sc_obs/sim/43samples/Nendss_"+str(dover*10)+".npy")
Robss = np.load("data/sc_obs/sim/43samples/Robss_"+str(dover*10)+".npy")
xobss = np.load("data/sc_obs/sim/43samples/xobss_"+str(dover*10)+".npy")
x0ss = np.load("data/sc_obs/sim/43samples/x0ss_"+str(dover*10)+".npy")
xfss = np.load("data/sc_obs/sim/43samples/xfss_"+str(dover*10)+".npy")

thiss1 = np.load("data/sc_obs/sim/9samples/thiss_"+str(dover*10)+".npy")
dthiss1 = np.load("data/sc_obs/sim/9samples/dthiss_"+str(dover*10)+".npy")
Xshiss1 = np.load("data/sc_obs/sim/9samples/Xshiss_"+str(dover*10)+".npy")
Ushiss1 = np.load("data/sc_obs/sim/9samples/Ushiss_"+str(dover*10)+".npy")
ueffss1 = np.load("data/sc_obs/sim/9samples/ueffss_"+str(dover*10)+".npy")
idx_fins1 = np.load("data/sc_obs/sim/9samples/idx_fins_"+str(dover*10)+".npy")
Nendss1 = np.load("data/sc_obs/sim/9samples/Nendss_"+str(dover*10)+".npy")
Robss1 = np.load("data/sc_obs/sim/9samples/Robss_"+str(dover*10)+".npy")
xobss1 = np.load("data/sc_obs/sim/9samples/xobss_"+str(dover*10)+".npy")
x0ss1 = np.load("data/sc_obs/sim/9samples/x0ss_"+str(dover*10)+".npy")
xfss1 = np.load("data/sc_obs/sim/9samples/xfss_"+str(dover*10)+".npy")

thiss = np.concatenate((thiss,thiss1),0)
dthiss = np.concatenate((dthiss,dthiss1),0)
Xshiss = np.concatenate((Xshiss,Xshiss1),0)
Ushiss = np.concatenate((Ushiss,Ushiss1),0)
ueffss = np.concatenate((ueffss,ueffss1),0)
idx_fins = np.concatenate((idx_fins,idx_fins1),0)
Nendss = np.concatenate((Nendss,Nendss1),0)
Robss = np.concatenate((Robss,Robss1),0)
xobss = np.concatenate((xobss,xobss1),0)
x0ss = np.concatenate((x0ss,x0ss1),0)
xfss = np.concatenate((xfss,xfss1),0)

"""
thiss = np.load("data/sc_obs/sim/50samples/thiss_"+str(dover*10)+".npy")
dthiss = np.load("data/sc_obs/sim/50samples/dthiss_"+str(dover*10)+".npy")
Xshiss = np.load("data/sc_obs/sim/50samples/Xshiss_"+str(dover*10)+".npy")
Ushiss = np.load("data/sc_obs/sim/50samples/Ushiss_"+str(dover*10)+".npy")
ueffss = np.load("data/sc_obs/sim/50samples/ueffss_"+str(dover*10)+".npy")
idx_fins = np.load("data/sc_obs/50samples/sim/idx_fins_"+str(dover*10)+".npy")
Nendss = np.load("data/sc_obs/50samples/sim/Nendss_"+str(dover*10)+".npy")
Robss = np.load("data/sc_obs/50samples/sim/Robss_"+str(dover*10)+".npy")
xobss = np.load("data/sc_obs/50samples/sim/xobss_"+str(dover*10)+".npy")
x0ss = np.load("data/sc_obs/50samples/sim/x0ss_"+str(dover*10)+".npy")
xfss = np.load("data/sc_obs/50samples/sim/xfss_"+str(dover*10)+".npy")
"""

Ndata = thiss.shape[0]
Ncon = ueffss.shape[1]
Nsc = idx_fins.shape[2]
ueffss_mpc = []
ueffss_mpc2 = []
Nends2 = [150*2,150*2,150,150*2]
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
ueff_trunc_ave_per = ueff_trunc_ave2/ueff_trunc_ave2[3]*100

c_caltech_o = [255/256,108/256,12/256]
c_caltech_g = [0/256,88/256,80/256]
c_caltech_b = [0/256,59/256,76/256]
c_caltech_o2 = [249/256,190/256,0/256]
LabelSize = 22
nticks_acc = 25

idx_data = 14
Xshis = Xshiss[idx_data,:,:,:]
Nends = Nendss[idx_data,:,:]
x0s = x0ss[idx_data,:,:]
xfs = xfss[idx_data,:,:]
xobs = xobss[idx_data,:,:]
Robs = Robss[idx_data]
idx_fin = idx_fins[idx_data,:,:]
Nobs = xobs.shape[0]
fig = plt.figure(figsize=(24,5))
fig.subplots_adjust(wspace=0.1)
titles = [r"(a) Learning-based feedforward",r"(b) Robust MPC",r"(c) LAG-ROS",r"(d) Global solution"]
for c in range(Ncon):
    if c == 0:
        c_con = 0
    elif c == 1:
        c_con = 2
    elif c == 2:
        c_con = 1
    elif c == 3:
        c_con = 3
    ax = fig.add_subplot(1,Ncon,c+1)
    Xhisc = Xshis[:,c_con,:,:]
    thp = np.linspace(0,2*np.pi,100)
    a1 = ax.scatter(np.nan,np.nan,fc="none",ec="k",marker="o",s=100,alpha=0.7,label=r"start")
    a2 = ax.scatter(np.nan,np.nan,color="k",s=100,alpha=0.7,label=r"goal")
    if c == 2:
        ax.legend(ncol=2,loc="upper center",fontsize=19,bbox_to_anchor=(-0.05,1.03))
    for p in range(Nsc):
        colorp = "C"+str(p)
        Nend = int(Nends[c_con,p])
        ax.plot(Xhisc[0:Nend+1,p,0],Xhisc[0:Nend+1,p,1],color=colorp,lw=3,alpha=0.5,ls="--")
        ax.scatter(x0s[p,:][0],x0s[p,:][1],fc="none",ec=colorp,marker="o",s=100)
        ax.scatter(xfs[p,:][0],xfs[p,:][1],color=colorp,s=100)
        #plt.scatter(Xhisc[Nend,p,0],Xhisc[Nend,p,1],color=colorp,s=100)
    for j in range(Nobs):
        c_plot = plt.Circle((xobs[j,:][0],xobs[j,:][1]),Robs,fc="C7",ec="none",alpha=0.5)
        plt.gcf().gca().add_artist(c_plot)
    #plt.legend(loc="upper left",fontsize=13)
    ax.set_xlim(-0.9,5.99)
    ax.set_ylim(-0.9,5.99)
    ax.grid()
    ax.tick_params(labelsize=nticks_acc)
    ax.set_title(titles[c],fontsize=LabelSize)

    if c == 0:
        ax.set_xlabel(r"horizontal coordinate $p_x$ [m]",fontsize=LabelSize)
        ax.set_ylabel(r"vertical coordinate $p_y$ [m]",fontsize=LabelSize)
        ax.xaxis.set_label_coords(2.1,-0.13)
    #ax.axes.xaxis.set_visible(False)
    #ax.axes.yaxis.set_visible(False)
    #plt.grid()
fname = "figs/sc_obs.pdf"
plt.savefig(fname,bbox_inches='tight',dpi=DPI)
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
    for k in range(9):
        thissk = np.load("data/sc_obs/sim/thiss_"+str(k)+str(dover*10)+".npy")
        dthissk = np.load("data/sc_obs/sim/dthiss_"+str(k)+str(dover*10)+".npy")
        Xshissk = np.load("data/sc_obs/sim/Xshiss_"+str(k)+str(dover*10)+".npy")
        Ushissk = np.load("data/sc_obs/sim/Ushiss_"+str(k)+str(dover*10)+".npy")
        ueffssk = np.load("data/sc_obs/sim/ueffss_"+str(k)+str(dover*10)+".npy")
        idx_finsk = np.load("data/sc_obs/sim/idx_fins_"+str(k)+str(dover*10)+".npy")
        Nendssk = np.load("data/sc_obs/sim/Nendss_"+str(k)+str(dover*10)+".npy")
        x0ssk = np.load("data/sc_obs/sim/x0ss_"+str(k)+str(dover*10)+".npy")
        xfssk = np.load("data/sc_obs/sim/xfss_"+str(k)+str(dover*10)+".npy")
        if k == 0:
            thiss = thissk
            dthiss = dthissk
            Xshiss = Xshissk
            Ushiss = Ushissk
            ueffss = ueffssk
            idx_fins = idx_finsk
            Nendss = Nendssk
            x0ss = x0ssk
            xfss = xfssk
        else:
            thiss = np.concatenate((thiss,thissk),0)
            dthiss = np.concatenate((dthiss,dthissk),0)
            Xshiss = np.concatenate((Xshiss,Xshissk),0)
            Ushiss = np.concatenate((Ushiss,Ushissk),0)
            ueffss = np.concatenate((ueffss,ueffssk),0)
            idx_fins = np.concatenate((idx_fins,idx_finsk),0)
            Nendss = np.concatenate((Nendss,Nendssk),0)
            x0ss = np.concatenate((x0ss,x0ssk),0)
            xfss = np.concatenate((xfss,xfssk),0)
    thissl.append(thiss)
    dthissl.append(dthiss)
    Xshissl.append(Xshiss)
    Ushissl.append(Ushiss)
    ueffssl.append(ueffss)
    idx_finsl.append(idx_fins)
    Nendssl.append(Nendss)
    x0ssl.append(x0ss)
    xfssl.append(xfss)
    
i_d = 0
for dover in dovers:
    """
    thiss = np.load("data/sc_obs/sim/thiss_"+str(dover*10)+".npy")
    dthiss = np.load("data/sc_obs/sim/dthiss_"+str(dover*10)+".npy")
    Xshiss = np.load("data/sc_obs/sim/Xshiss_"+str(dover*10)+".npy")
    Ushiss = np.load("data/sc_obs/sim/Ushiss_"+str(dover*10)+".npy")
    ueffss = np.load("data/sc_obs/sim/ueffss_"+str(dover*10)+".npy")
    idx_fins = np.load("data/sc_obs/sim/idx_fins_"+str(dover*10)+".npy")
    Nendss = np.load("data/sc_obs/sim/Nendss_"+str(dover*10)+".npy")
    x0ss = np.load("data/sc_obs/sim/x0ss_"+str(dover*10)+".npy")
    xfss = np.load("data/sc_obs/sim/xfss_"+str(dover*10)+".npy")
    """
    thiss = thissl[i_d]
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
    Ndata = thiss.shape[0]
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
    ueff_trunc_ave_per = ueff_trunc_ave2/ueff_trunc_ave2[3]*100
    success_rates.append(success_rate)
    ueff_aves.append(ueff_ave)
    ueff_trunc_aves.append(ueff_trunc_ave)
    ueff_trunc_ave2s.append(ueff_trunc_ave2)
    i_d += 1
    

titles = [r"(a)",r"(b)",r"(c)",r"(d)"]    
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
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(14,4))
fig.subplots_adjust(wspace=0.36)
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
ax1.legend(titles,fontsize=leg_size,loc="center left",ncol=2,bbox_to_anchor=(0,0.65))
ax1.grid()
ax2.plot(dovers,ueff_trunc_ave2s[:,3],marker="o",color="C3",ls=linestyle[3],lw=LW,ms=MS)
for c in range(Ncon):
    if c == 2:
        ax2.plot(np.nan,np.nan,marker="o",alpha=atra,ls=linestyle[c],lw=LW,ms=MS)
    elif c == 3:
        ax2.plot(np.nan,np.nan,marker="o",alpha=atra,ls=linestyle[c],lw=LW,ms=MS)
    else:
        ax2.plot(dovers,ueff_trunc_ave2s[:,cp[c]],marker="o",alpha=atra,ls=linestyle[c],lw=LW,ms=MS)
ax2.plot(dovers,ueff_trunc_ave2s[:,1],marker="o",color="C2",ls=linestyle[2],lw=LW,ms=MS)
ax2.set_xticks(dovers)
ax2.set_xlabel(r"$\sup_{x,t}\|d(x,t)\|$",fontsize=LabelSize)
ax2.set_ylabel(r"$\mathcal{L}2$ control effort",fontsize=LabelSize)
ax2.tick_params(labelsize=nticks_acc)
ax2.grid()
fname = "figs/sc_obs_dist.pdf"
plt.savefig(fname,bbox_inches='tight',dpi=DPI)
plt.show()
print(success_rates*100)
print(ueff_trunc_ave2s.T/ueff_trunc_ave2s[:,3]*100)






"""
dover = 0.2
Xshis = np.load("data/sc_obs/sim/success/Xshis_"+str(dover*10)+".npy")
Ushis = np.load("data/sc_obs/sim/success/Ushis_"+str(dover*10)+".npy")
ueffs = np.load("data/sc_obs/sim/success/ueffs_"+str(dover*10)+".npy")
idx_fin = np.load("data/sc_obs/sim/success/idx_fin_"+str(dover*10)+".npy")
Nends = np.load("data/sc_obs/sim/success/Nends_"+str(dover*10)+".npy")
Robs = np.load("data/sc_obs/sim/success/Robs_"+str(dover*10)+".npy")
xobs = np.load("data/sc_obs/sim/success/xobs_"+str(dover*10)+".npy")
x0s = np.load("data/sc_obs/sim/success/x0s_"+str(dover*10)+".npy")
xfs = np.load("data/sc_obs/sim/success/xfs_"+str(dover*10)+".npy")

c_caltech_o = [255/256,108/256,12/256]
c_caltech_g = [0/256,88/256,80/256]
c_caltech_b = [0/256,59/256,76/256]
c_caltech_o2 = [249/256,190/256,0/256]

Ncon = Xshis.shape[1]
Nsc = Xshis.shape[2]
Nobs = xobs.shape[0]
Xhisc_r = Xshis[:,1,:,:]

fig = plt.figure(figsize=(11,5))
fig.subplots_adjust(wspace=0.1)
for c in range(Ncon):
    ax = fig.add_subplot(1,2,c+1)
    Xhisc = Xshis[:,c,:,:]
    thp = np.linspace(0,2*np.pi,100)
    ax.scatter(np.nan,np.nan,fc="none",ec=c_caltech_g,marker="o",s=100,alpha=0.7,label=r"start")
    ax.scatter(np.nan,np.nan,color=c_caltech_g,s=100,alpha=0.7,label=r"target")
    for p in range(Nsc):
        colorp = "C"+str(p)
        Nend = int(Nends[c,p])
        ax.plot(Xhisc[0:Nend+1,p,0],Xhisc[0:Nend+1,p,1],color=colorp,lw=3,alpha=0.5,ls="--")
        ax.scatter(x0s[p,:][0],x0s[p,:][1],fc="none",ec=colorp,marker="o",s=100)
        ax.scatter(xfs[p,:][0],xfs[p,:][1],color=colorp,s=100)
        #plt.scatter(Xhisc[Nend,p,0],Xhisc[Nend,p,1],color=colorp,s=100)
    for j in range(Nobs):
        c_plot = plt.Circle((xobs[j,:][0],xobs[j,:][1]),Robs,fc="C7",ec="none",alpha=0.5)
        plt.gcf().gca().add_artist(c_plot)
    #plt.legend(loc="upper left",fontsize=13)
    ax.set_xlim(-0.6,5.6)
    ax.set_ylim(-0.6,5.6)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    #plt.grid()
plt.show()

Xhis1 = Xshis[:,0,:,:]
Xhis2 = Xshis[:,1,:,:]

for j in range(Nobs):
    c_plot = plt.Circle((xobs[j,:][0],xobs[j,:][1]),Robs,fc="C7",ec="none",alpha=0.5)
    plt.gcf().gca().add_artist(c_plot)
for p in range(Nsc):
    colorp = "C"+str(p)
    Nend = int(Nends[c,p])
    plt.plot(Xhis1[0:Nend+1,p,0],Xhis1[0:Nend+1,p,1],color=colorp,lw=3,alpha=0.2,ls="--")
    plt.plot(Xhis2[0:Nend+1,p,0],Xhis2[0:Nend+1,p,1],color=colorp,lw=3,alpha=0.5,ls="-")
    plt.scatter(x0s[p,:][0],x0s[p,:][1],fc="none",ec=colorp,marker="o",s=100)
    plt.scatter(xfs[p,:][0],xfs[p,:][1],color=colorp,marker="x",s=100)
    plt.scatter(Xhis1[Nend,p,0],Xhis1[Nend,p,1],color=colorp,s=100)
    plt.scatter(Xhis2[Nend,p,0],Xhis2[Nend,p,1],color=colorp,s=100,alpha=0.2)
plt.xlim(-0.6,5.6)
plt.ylim(-0.6,5.6)
plt.grid()
plt.show()

xdata = np.load("data/sc_obs/xdata.npy")
obsdata = np.load("data/sc_obs/obsdata.npy")

matplotlib.rcParams.update({'font.size': 0})

fig = plt.figure(figsize=(10,10))
fig.subplots_adjust(wspace=0.1,hspace=0.1)
for isim in range(9,13):
    xs = xdata[isim,:,:,:]
    xobs = obsdata[isim,:,:]
    ax = fig.add_subplot(2,2,isim-8)
    for p in range(Nsc):
        xhis = xs[p,:,:]
        colorp = "C"+str(p)  
        ax.plot(xhis[:,0],xhis[:,1],color=colorp,lw=3,alpha=0.5,ls="--")
        ax.scatter(xhis[0,0],xhis[0,1],fc="none",ec=colorp,marker="o",s=100)
        ax.scatter(xhis[-1,0],xhis[-1,1],color=colorp,s=100)
    for o in range(Nobs):
        c_plot = plt.Circle((xobs[o,:][0],xobs[o,:][1]),Robs,fc="C7",ec="none",alpha=0.5)
        plt.gcf().gca().add_artist(c_plot)
    ax.set_xlim(-0.7,5.7)
    ax.set_ylim(-0.7,5.7)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
plt.show()
"""
    