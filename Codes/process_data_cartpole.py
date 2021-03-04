#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:29:45 2020

@author: hiroyasu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 17:23:40 2020

@author: hiroyasu
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import animation

matplotlib.rcParams.update({'font.size': 15})

dover = 2
thiss = np.load("data/cartpole/sim/thiss_"+str(dover*10)+".npy")
dthiss = np.load("data/cartpole/sim/dthiss_"+str(dover*10)+".npy")
Xshiss = np.load("data/cartpole/sim/Xshiss_"+str(dover*10)+".npy")
Ushiss = np.load("data/cartpole/sim/Ushiss_"+str(dover*10)+".npy")
ueffss = np.load("data/cartpole/sim/ueffss_"+str(dover*10)+".npy")
idx_fins = np.load("data/cartpole/sim/idx_fins_"+str(dover*10)+".npy")
X0s = np.load("data/cartpole/sim/X0s_"+str(dover*10)+".npy")
Xfs = np.load("data/cartpole/sim/Xfs_"+str(dover*10)+".npy")


Ndata = thiss.shape[0]
Ncon = ueffss.shape[1]
Nend = thiss.shape[1]

success_rate = np.sum(idx_fins,0)/Ndata
ueff_ave = np.sum(ueffss,0)/Ndata
dts = np.sum(dthiss,(0,1))/Nend/Ndata
dts[1] = dts[1]/Nend
ueff_ave_per = ueff_ave/ueff_ave[2]*100

"""
idx_data = 0
this = thiss[idx_data,:]
Xshis = Xshiss[idx_data,:,:]
Ushis = Ushiss[idx_data,:,:]
ueffs = ueffss[idx_data,:]
X0 = X0s[idx_data,:]
Xf = Xfs[idx_data,:]

c_caltech_o = [255/256,108/256,12/256]
c_caltech_g = [0/256,88/256,80/256]
c_caltech_b = [0/256,59/256,76/256]
c_caltech_o2 = [249/256,190/256,0/256]

N = Ushis.shape[0]
Ncon = Xshis.shape[1]
n_states = Xshis.shape[2]
ueffs_l = []
for c in range(Ncon):
    Xhisc = Xshis[:,c,:]
    ueffc = 0
    plt.figure()
    for i in range(n_states):
        colorp = "C"+str(i)
        plt.plot(this,Xhisc[:,i],color=colorp)
    #ueffc = np.sum(np.sqrt(np.sum(Uhis[:,c,:]**2,1)))
    ueffc = np.sum(Ushis[:,c,:]**2)
    ueffs_l.append(ueffc)
    plt.grid()
    plt.show()
print(ueffs_l)
        
figs = []
ims = [[0]*(N+1) for i in range(Ncon)]
h_cart = 0.25
w_cart = 0.5
l_pole = 0.5
w_pole = 0.1
r_wheel = 0.05
dt = 0.1
r_fulcrum = 0.02
for c in range(Ncon):
    fig = plt.figure(figsize=(10,1.5))
    Xhis = Xshis[:,c,:]
    for k in range(N+1):
        im = []
        X = Xhis[k,:]
        c_cart = (X[0]-w_cart/2,r_wheel)
        c_pole = (X[0]-w_pole/2*np.cos(X[1]),h_cart+w_pole/2*np.sin(X[1])+r_wheel)
        c_wheel_r = (X[0]-w_cart/2.5,r_wheel)
        c_wheel_f = (X[0]+w_cart/2.5,r_wheel)
        c_fulcrum = (X[0],h_cart+r_wheel)
        a_pole = np.pi/2-np.rad2deg(X[1])
        cart = plt.Rectangle(c_cart,width=w_cart,height=h_cart,fc="C7",ec="C7",alpha=0.5)
        pole = plt.Rectangle(c_pole,width=w_pole,height=l_pole,angle=a_pole,fc="C7",ec="C7",alpha=0.5)
        wheel_r = plt.Circle(c_wheel_r,r_wheel,fc="k",ec="k",alpha=0.7)
        wheel_f = plt.Circle(c_wheel_f,r_wheel,fc="k",ec="k",alpha=0.7)
        fulcrum = plt.Circle(c_fulcrum,r_fulcrum,fc="C7",ec="C7")
        im.append(plt.gcf().gca().add_artist(cart))
        im.append(plt.gcf().gca().add_artist(pole))
        im.append(plt.gcf().gca().add_artist(wheel_r))
        im.append(plt.gcf().gca().add_artist(wheel_f))
        im.append(plt.gcf().gca().add_artist(fulcrum))
        
        c_cart_f = (Xf[0]-w_cart/2,r_wheel)
        c_pole_f = (Xf[0]-w_pole/2*np.cos(X[1]),h_cart+w_pole/2*np.sin(Xf[1])+r_wheel)
        c_wheel_r_f = (Xf[0]-w_cart/2.5,r_wheel)
        c_wheel_f_f = (Xf[0]+w_cart/2.5,r_wheel)
        c_fulcrum_f = (Xf[0],h_cart+r_wheel)
        a_pole_f = np.pi/2-np.rad2deg(Xf[1])
        cart_f = plt.Rectangle(c_cart_f,width=w_cart,height=h_cart,fc="C7",ec="C7",alpha=0.2)
        pole_f = plt.Rectangle(c_pole_f,width=w_pole,height=l_pole,angle=a_pole_f,fc="C7",ec="C7",alpha=0.2)
        wheel_r_f = plt.Circle(c_wheel_r_f,r_wheel,fc="k",ec="k",alpha=0.3)
        wheel_f_f = plt.Circle(c_wheel_f_f,r_wheel,fc="k",ec="k",alpha=0.3)
        fulcrum_f = plt.Circle(c_fulcrum_f,r_fulcrum,fc="C7",ec="C7",alpha=0.5)
        im.append(plt.gcf().gca().add_artist(cart_f))
        im.append(plt.gcf().gca().add_artist(pole_f))
        im.append(plt.gcf().gca().add_artist(wheel_r_f))
        im.append(plt.gcf().gca().add_artist(wheel_f_f))
        im.append(plt.gcf().gca().add_artist(fulcrum_f))
        ims[c][k] = im
    plt.xlim(-7,7)
    plt.ylim(0,1.3)
    plt.title("cart-pole perturbed by d(t) (sup||d(t)|| = 5.0)")
    plt.axes().set_aspect("equal")
    figs.append(fig)
for c in range(Ncon):
    ani = animation.ArtistAnimation(figs[c],ims[c],interval=dt*1000)
    ani.save("movies/cartpole/output"+str(c)+"_"+str(dover*10)+".mp4",writer="ffmpeg")
"""      