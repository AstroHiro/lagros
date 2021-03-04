#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:52:25 2021

@author: hiroyasu
"""


import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import control
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Concatenate

from classncm import NCM

np.random.seed(seed=8)

class NMPC():
    def __init__(self,Nobs,Nsc):
        self.mu = 3.9860e+05 # gm of earth
        self.kJ2 = 2.6330e+10
        self.ome = 7.2921e-5
        self.R_earth = 3781e+03
        self.r_leo = self.R_earth+800
        self.Cd = 1
        self.mass = 1
        self.rho = 1e-8
        self.Area = 1
        self.Cdj = self.Cd
        self.massj = self.mass
        self.rhoj = self.rho
        self.Areaj = self.Area
        self.Nobs = Nobs
        self.Nsc = Nsc
        self.Robs = 0.5
        self.Rsc = 1.0
        self.d_btw_obs = 0.25
        self.Rsafe = self.d_btw_obs/2
        self.Rsen = 2.0
        self.d_btw_0f = 5.0
        self.xlims = np.array([0,5])
        self.ylims = np.array([0,5])
        self.zlims = np.array([0,5])
        self.n_states = 6
        self.n_add_states = 0
        self.m_inputs = 3
        self.q_states = self.n_states//2
        self.dim = 3
        Z = np.zeros((self.n_states+self.q_states,self.q_states))
        eyeq = np.identity(self.q_states)
        self.B = np.vstack((Z,eyeq))
        Zj = np.zeros((self.q_states,self.q_states))
        self.Bj = np.vstack((Zj,eyeq))
        Q = np.identity(self.n_states)*1
        R = np.identity(self.m_inputs)
        Xj0 = np.array([1,1,1,1,1,1])
        r0 = self.r_leo
        v0 = np.sqrt(self.mu/r0)
        h0 = r0*v0
        self.Xc0 = np.array([r0,v0,h0,np.pi/3,np.pi/3,0])
        self.A = self.relativeA(Xj0,self.Xc0,np.array([0,0,0]))
        K,P,_ = control.lqr(self.A,self.Bj,Q,R)
        self.Xlims = np.array([-np.ones(self.n_states),np.ones(self.n_states)])*10
        plims = np.array([self.Xc0,self.Xc0])
        plims[0,5] = -2*np.pi
        plims[1,5] = 2*np.pi
        self.plims = plims
        self.P = P
        self.Q = Q
        self.R = R
        self.dt = 0.1
        self.Nrk = 10
        self.h = self.dt/self.Nrk
        self.N = 150
        self.fname_ncm = "NCM_leo"
        #test1 = self.relativef(Xj0,self.Xc0,np.zeros(3))
        #Mmat,Cmat,Gmat,Dmat = self.lagrangeMCGD(Xj0,self.Xc0,np.zeros(3))
        #test2 = -Cmat@Xj0[3:6]-Gmat-Dmat
        #print(test1[3:6]-test2)
        #Xd = np.array([0.1,0.2,0.3,-0.4,0.5,8])
        #Ud = np.array([-3,7,2])
        #print(self.clfqp_a_lagrange_test(Xj0,Xd,Ud,self.Xc0))
        #print(self.clfqp_a_lagrange(Xj0,Xd,Ud,self.Xc0))
    
    def generate_endpoints(self,final):
        xlims = self.xlims
        ylims = self.ylims
        zlims = self.zlims
        d_btw_0f = self.d_btw_0f
        Nsc = self.Nsc
        Rsc = self.Rsc
        Rsafe = self.Rsafe
        if final == 0:
            x0s = np.zeros((Nsc,self.dim))
            xfs = np.zeros((Nsc,self.dim))
            for p in range(Nsc):
                while True:
                    pxk0 = np.random.uniform(xlims[0],xlims[1])
                    pyk0 = np.random.uniform(ylims[0],ylims[1])
                    pzk0 = np.random.uniform(zlims[0],zlims[1])
                    x0 = np.array([pxk0,pyk0,pzk0])
                    pxkf = np.random.uniform(xlims[0],xlims[1])
                    pykf = np.random.uniform(ylims[0],ylims[1])
                    pzkf = np.random.uniform(zlims[0],zlims[1])
                    xf = np.array([pxkf,pykf,pzkf])
                    idx = 0
                    if np.linalg.norm(xf-x0) <= d_btw_0f:
                        continue
                    for q in range(p):
                        if np.linalg.norm(x0-x0s[q,:]) <= Rsc+Rsafe:
                            break
                        elif np.linalg.norm(xf-xfs[q,:]) <= Rsc+Rsafe:
                            break
                        idx += 1
                    if idx == p:
                        break
                x0s[p,:] = x0
                xfs[p,:] = xf
            self.x0s = x0s
            self.xfs = xfs
        else:
            x0s = np.zeros((Nsc,self.dim))
            xfs = np.zeros((Nsc,self.dim))
            Rf = (xlims[1]/2)*0.9
            """
            pxkf = Rf*np.cos(0)
            pykf = Rf*np.sin(0)
            pzkf = 0
            xfs[0,:] = np.array([pxkf,pykf,pzkf])
            pxkf = Rf*np.cos(np.pi/2)
            pykf = Rf*np.sin(np.pi/2)
            pzkf = 0
            xfs[1,:] = np.array([pxkf,pykf,pzkf])
            pxkf = 0
            pykf = Rf*np.cos(np.pi/2)
            pzkf = Rf*np.sin(np.pi/2)
            xfs[2,:] = np.array([pxkf,pykf,pzkf])
            pxkf = Rf*np.cos(np.pi/4)*np.cos(np.pi/4)
            pykf = Rf*np.cos(np.pi/4)*np.sin(np.pi/4)
            pzkf = Rf*np.sin(np.pi/4)
            xfs[3,:] = np.array([pxkf,pykf,pzkf])
            pxkf = Rf*np.cos(-np.pi/4)*np.cos(3*np.pi/4)
            pykf = Rf*np.cos(-np.pi/4)*np.sin(3*np.pi/4)
            pzkf = Rf*np.sin(-np.pi/4)
            xfs[4,:] = np.array([pxkf,pykf,pzkf])
            pxkf = Rf*np.cos(np.pi/4)*np.cos(3*np.pi/4)
            pykf = Rf*np.cos(np.pi/4)*np.sin(3*np.pi/4)
            pzkf = Rf*np.sin(np.pi/4)
            xfs[5,:] = np.array([pxkf,pykf,pzkf])
            xfs[6,:] = -xfs[0,:]
            xfs[7,:] = -xfs[1,:]
            xfs[8,:] = -xfs[2,:]
            xfs[9,:] = -xfs[3,:]
            xfs[10,:] = -xfs[4,:]
            xfs[11,:] = -xfs[5,:]
            print(xfs)
            """
            """
            Ncir = 4
            Npart = 8
            xfs = []
            for q in range(Ncir):
                for p in range(Npart):
                    pxkf = Rf*np.cos(2*np.pi/Npart*p)*np.cos(np.pi/Ncir*q)+xlims[1]/2
                    pykf = Rf*np.cos(2*np.pi/Npart*p)*np.sin(np.pi/Ncir*q)+ylims[1]/2
                    pzkf = Rf*np.sin(2*np.pi/Npart*p)+zlims[1]/2
                    xf = np.array([pxkf,pykf,pzkf])
                    if np.abs(xf[2]-zlims[1]/2) != Rf:
                        xfs.append(xf)
            xfs.append(np.array([xlims[1]/2,ylims[1]/2,zlims[1]/2+Rf]))    
            xfs.append(np.array([xlims[1]/2,ylims[1]/2,zlims[1]/2-Rf]))
            xfs = np.array(xfs)
            """
            for p in range(Nsc):
                pxk0 = Rf*np.cos(2*np.pi/Nsc*p)+xlims[1]/2
                pyk0 = Rf*np.sin(2*np.pi/Nsc*p)+ylims[1]/2
                pzk0 = 0.1
                x0 = np.array([pxk0,pyk0,pzk0])
                print(x0)
                
                pxkf = Rf*np.cos(2*np.pi/Nsc*p+np.pi)+xlims[1]/2
                pykf = Rf*np.sin(2*np.pi/Nsc*p+np.pi)+ylims[1]/2
                pzkf = zlims[1]-0.1
                xf = np.array([pxkf,pykf,pzkf])
                
                x0s[p,:] = x0
                xfs[p,:] = xf
            
            for p in range(Nsc):
                while True:
                    """
                    pxk0 = np.random.uniform(xlims[0],xlims[1])
                    pyk0 = np.random.uniform(ylims[0],ylims[1])
                    pzk0 = np.random.uniform(zlims[0],zlims[1])
                    x0 = np.array([pxk0,pyk0,pzk0])
                    """
                    x0 = x0s[p,:]
                    xf = xfs[p,:]
                    
                    idx = 0
                    if np.linalg.norm(xf-x0) <= d_btw_0f:
                        continue
                    for q in range(p):
                        if np.linalg.norm(x0-x0s[q,:]) <= Rsc+Rsafe:
                            break
                        elif np.linalg.norm(xf-xfs[q,:]) <= Rsc+Rsafe:
                            break
                        idx += 1
                    if idx == p:
                        break
                x0s[p,:] = x0
            self.x0s = x0s
            self.xfs = xfs
            r = Rf
            th = np.linspace(0,2*np.pi,100)
            phi = np.linspace(0,np.pi,100)
            th,phi = np.meshgrid(th,phi)
            x = np.cos(th)*np.sin(phi)*r+xlims[1]/2
            y = np.sin(th)*np.sin(phi)*r+ylims[1]/2
            z = np.cos(phi)*r+zlims[1]/2
            
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(self.xfs[:,0],self.xfs[:,1],self.xfs[:,2])
            ax.scatter(self.x0s[:,0],self.x0s[:,1],self.x0s[:,2])
            #ax.plot_surface(x,y,z,alpha=0.2)
            plt.show()
        pass
    
    def assignment(self,Xs):
        xfs = self.xfs
        costs = np.zeros((self.Nsc,self.Nsc))
        for p in range(self.Nsc):
            xp = Xs[p,0:self.q_states]
            for q in range(self.Nsc):
                xfq = xfs[q,:]
                costs[p,q] = np.linalg.norm(xp-xfq)
        prices = np.zeros(self.Nsc)
        eps = 0.1
        for k in range(1000):
            jis = [0]*self.Nsc
            for i in range(self.Nsc):
                vi = np.min(costs[i,:]+np.abs(prices))
                idx = np.argsort(costs[i,:]+np.abs(prices))
                ji = np.argmin(costs[i,:]+np.abs(prices))
                jis[i] = ji
                wi = costs[i,:][idx[1]]+prices[idx[1]]
                gmi = wi-vi+eps
                prices[ji] = np.abs(prices[ji])+gmi
            if len(jis) == len(set(jis)):
                break
        Xfs = np.zeros((self.Nsc,self.n_states))
        for p in range(self.Nsc):
            Xfs[p,:] =  np.hstack((xfs[jis[p],:],np.zeros(self.q_states+self.n_add_states)))
        return Xfs

    def generate_environment(self,final=0):
        self.generate_endpoints(final)
        xobs = []
        self.xobs = np.array(xobs)
        pass
    """
    def chieff_old(self,Xc):
        a = Xc[0]
        q1 = Xc[1]
        q2 = Xc[2]
        i = Xc[3]
        Om = Xc[4]
        th = Xc[5]
        e = np.sqrt(q1**2+q2**2)
        om = np.arctan2(q1,q2)
        nu = th-om
        ci = np.cos(i)
        si = np.sin(i)
        cth = np.cos(th)
        sth = np.sin(th)
        com = np.cos(om)
        som = np.sin(om)
        cthom = np.cos(th+om)
        sthom = np.sin(th+om)
        s2th = np.sin(2*th)
        cnu = np.cos(nu)
        snu = np.sin(nu)
        n = np.sqrt(mu/a**3)
        kp = np.sqrt(1-e**2)
        chi = 1+e*cnu
        ep = kJ2*chi**3/(n*a**5*kp**7)
        E = kp*Cdn*vr/(n*a*chi)
        dadt = -2*a*chi*ep/kp**2*(e*snu*(1-3*si**2*sth**2)+chi*si**2*s2th)-2*a*E/kp**2*(e*snu*R1+chi*R2)*Vr
        dq1dt = ep*((chi-(5*chi+2)*si**2*sth**2)*cth+2*e*(-som*cth+sthom*ci**2)*sth)+E*(chi*cth*R1-((chi+1)*sth+e*som)*R2+e*com*ci*sth*R3/si)*Vr
        dq2dt = -ep*sth*(chi*(1-3*si**2*sth**2)+2*(chi+1)*si**2*cth**2+2*e*(com*cth-cthom*ci**2))-E*(chi*sth*R1+((chi+1)*cth+e*com)*R2+e*som*ci*sth*R3/si)*Vr
        didt = -ep*s2i*s2th/2-E*cth*R3*Vr
        dOmdt = -2*ep*ci*sth**2-E*sth/si*R3*Vr
        dthdt = n*chi**2/kp**3+2*ep*ci**2*sth**2+E*ci*sth/si*R3*Vr
        pass
    """
    
    def chieff(self,Xc,Uc):
        mu = self.mu
        kJ2 = self.kJ2
        ome = self.ome
        Cd = self.Cd
        mass = self.mass
        rho = self.rho
        Area = self.Area
        uR = Uc[0]
        uT = Uc[1]
        uN = Uc[2]
        r = Xc[0]
        vx = Xc[1]
        h = Xc[2]
        Om = Xc[3]
        i = Xc[4]
        th = Xc[5]
        si = np.sin(i)
        ci = np.cos(i)
        s2i = np.sin(2*i)
        sth = np.sin(th)
        cth = np.cos(th)
        s2th = np.sin(2*th)
        C = Cd*(Area/mass)*rho/2
        Va = np.array([vx,h/r-ome*r*ci,ome*r*cth*si])
        va = np.linalg.norm(Va)
        drdt = vx
        dvxdt = -mu/r**2+h**2/r**3-kJ2/r**4*(1-3*si**2*sth**2)-C*va*vx+uR
        dhdt = -kJ2*si**2*s2th/r**3-C*va*(h-ome*r**2*ci)+r*uT
        dOmdt = -2*kJ2*ci*sth**2/h/r**3-C*va*ome*r**2*s2th/2/h+(r*sth/h/si)*uN
        didt = -kJ2*s2i*s2th/2/h/r**3-C*va*ome*r**2*si*cth**2/h+(r*cth/h)*uN
        dthdt = h/r**2+2*kJ2*ci**2*sth**2/h/r**3+C*va*ome*r**2*ci*s2th/2/h-(r*sth*ci/h/si)*uN
        dXcdt = np.array([drdt,dvxdt,dhdt,dOmdt,didt,dthdt])
        return dXcdt        
        
    def relativef(self,Xj,Xc,Uc):
        mu = self.mu
        kJ2 = self.kJ2
        ome = self.ome
        Cd = self.Cd
        mass = self.mass
        rho = self.rho
        Area = self.Area
        Cdj = self.Cdj
        massj = self.massj
        rhoj = self.rhoj
        Areaj = self.Areaj
        C = Cd*(Area/mass)*rho/2
        Cj = Cdj*(Areaj/massj)*rhoj/2
        xj = Xj[0]
        yj = Xj[1]
        zj = Xj[2]
        xjdot = Xj[3]
        yjdot = Xj[4]
        zjdot = Xj[5]
        r = Xc[0]
        vx = Xc[1]
        h = Xc[2]
        Om = Xc[3]
        i = Xc[4]
        th = Xc[5]
        uR = Uc[0]
        uT = Uc[1]
        uN = Uc[2]
        dXcdt = self.chieff(Xc,Uc)
        Omdot = dXcdt[3]
        idot = dXcdt[4]
        thdot = dXcdt[5]
        si = np.sin(i)
        ci = np.cos(i)
        s2i = np.sin(2*i)
        sth = np.sin(th)
        cth = np.cos(th)
        s2th = np.sin(2*th)
        lj = Xj[0:3]
        ljdot = Xj[3:6]
        Va = np.array([vx,h/r-ome*r*ci,ome*r*cth*si])
        va = np.linalg.norm(Va)
        omx = idot*cth+Omdot*sth*si+r/h*uN
        omz = thdot+Omdot*ci
        om_vec = np.array([omx,0,omz])
        Vaj = Va+ljdot+np.cross(om_vec,lj)
        vaj = np.linalg.norm(Vaj)
        rj = np.sqrt((r+xj)**2+yj**2+zj**2)
        rjZ = (r+xj)*si*sth+yj*si*cth+zj*ci
        zt = 2*kJ2*si*sth/r**4
        ztj = 2*kJ2*rjZ/rj**5
        alx = -kJ2*s2i*cth/r**5+3*vx*kJ2*s2i*sth/r**4/h-8*kJ2**2*si**3*ci*sth**2*cth/r**6/h**2
        alz = -2*h*vx/r**3-kJ2*si**2*s2th/r**5
        et = np.sqrt(mu/r**3+kJ2/r**5-5*kJ2*si**2*sth**2/r**5)
        etj = np.sqrt(mu/rj**3+kJ2/rj**5-5*kJ2*rjZ**2/rj**7)
        xjddot = 2*yjdot*omz-xj*(etj**2-omz**2)+yj*alz-zj*omx*omz-(ztj-zt)*si*sth-r*(etj**2-et**2)-Cj*vaj*(xjdot-yj*omz)-(Cj*vaj-C*va)*vx-uR
        yjddot = -2*xjdot*omz+2*zjdot*omx-xj*alz-yj*(etj**2-omz**2-omx**2)+zj*alx-(ztj-zt)*si*cth-Cj*vaj*(yjdot+xj*omz-zj*omx)-(Cj*vaj-C*va)*(h/r-ome*r*ci)-uT
        zjddot = -2*yjdot*omx-xj*omx*omz-yj*alx-zj*(etj**2-omx**2)-(ztj-zt)*ci-Cj*vaj*(zjdot+yj*omx)-(Cj*vaj-C*va)*ome*r*cth*si-uN
        Xdot = np.array([xjdot,yjdot,zjdot,xjddot,yjddot,zjddot])
        return Xdot
    
    def relativeA(self,Xj,Xc,Uc):
        dx = 0.001
        A = np.zeros((self.n_states,self.n_states))
        for i in range(self.n_states):
            ei = np.zeros(self.n_states)
            ei[i] = 1
            f1 = self.relativef(Xj+ei*dx,Xc,Uc)
            f0 = self.relativef(Xj-ei*dx,Xc,Uc)
            dfdx = (f1-f0)/2/dx
            A[:,i] = dfdx
        return A
    
    def GetB(self,X):
        B = self.B
        return B

    def ncm_train(self,iTrain):
        # enter your choice of CV-STEM sampling period
        dt = 1
        # specify upper and lower bounds of each state
        Xlims = self.Xlims
        # specify upper and lower bound of contraction rate (and we will find optimal alpha within this region)
        alims = np.array([0.1,30])
        Uc = np.zeros(3)
        # name your NCM
        fname_ncm = self.fname_ncm
        dynamicsf = lambda X,P: self.relativef(X,P,Uc)
        dynamicsg = lambda X,P: self.Bj
        ncm = NCM(dt,dynamicsf,dynamicsg,Xlims,alims,"con",fname_ncm,plims=self.plims)
        # You can use ncm.train(iTrain = 0) instead when using pre-traind NCM models.
        ncm.train(iTrain)
        return ncm
    
    def safeu(self,i,X,Xnow,Uds,Xc,dt):
        Uc = np.zeros(3)
        p = X[0:self.dim]
        v = X[self.q_states:self.q_states+self.dim]
        B = self.Bj
        f = self.relativef(X,Xc,Uc)
        Robs = self.Robs
        Rsafe = self.Rsafe
        Rsc = self.Rsc
        Us = cp.Variable((self.Nsc,self.m_inputs))
        U = Us[i,:]
        constraints = []
        pnext = p+v*dt
        pnnext = p+2*v*dt+(f[self.q_states:self.q_states+self.dim]+B[self.q_states:self.q_states+self.dim,:]@U)*dt**2
        for j in range(self.Nobs):
            xobsj = self.xobs[j,:]
            constraints += [(pnext-xobsj)@(pnnext-xobsj) >= (Robs+Rsafe)**2]
        for psc in range(self.Nsc):
            if psc != i:
                pj = Xnow[psc,:][0:self.dim]
                vj = Xnow[psc,:][self.q_states:self.q_states+self.dim]
                if np.linalg.norm(p-pj) <= self.Rsen:
                    Xj = Xnow[psc,:]
                    Uj = Us[psc,:]
                    fj = self.relativef(Xj,Xc,Uc)
                    Bj = self.Bj
                    pjnext = pj+vj*dt
                    pjnnext = pj+2*vj*dt+(fj[self.q_states:self.q_states+self.dim]+Bj[self.q_states:self.q_states+self.dim,:]@Uj)*dt**2
                    constraints += [(p-pj)@(pnnext-pjnnext) >= (Rsc+Rsafe)**2]
                    #constraints += [(pnext-pjnext)@(pnnext-pjnnext)-Rsc*cp.norm(pnext-pjnext) >= 0]
        prob = cp.Problem(cp.Minimize(cp.norm(Us-Uds,"fro")**2),constraints)
        prob.solve(solver=cp.MOSEK)
        return U.value

    def safeu_global(self,Xc,Xs,Xfs,dt):
        P = self.P
        Q = self.Q
        Us = cp.Variable((self.Nsc,self.m_inputs))
        Robs = self.Robs
        Rsc = self.Rsc
        Rsafe = self.Rsafe
        invR = np.linalg.inv(self.R)
        Uc = np.zeros(3)
        Bj = self.Bj
        constraints = []
        for i in range(self.Nsc):
            Xi = Xs[i,:]
            Xfi = Xfs[i,:]
            Ui = Us[i,:]
            pi = Xi[0:self.dim]
            vi = Xi[self.q_states:self.q_states+self.dim]
            fi = self.relativef(Xi,Xc,Uc)
            ei = Xi-Xfi
            pinext = pi+vi*dt
            pinnext = pi+2*vi*dt+(fi[self.q_states:self.q_states+self.dim]+Bj[self.q_states:self.q_states+self.dim,:]@Ui)*dt**2
            constraints += [2*ei@fi+2*ei@P@Bj@Ui <= -ei@(P@Bj@invR@Bj.T@P+Q)@ei]
        prob = cp.Problem(cp.Minimize(cp.norm(Us,"fro")**2),constraints)
        prob.solve(solver=cp.MOSEK)
        return Us.value
    
    def dynamics(self,t,X,U):
        Xc = X[0:6]
        Uc = np.zeros(3)
        Xj = X[6:12]
        f = self.chieff(Xc,Uc)
        fj = self.relativef(Xj,Xc,Uc)
        B = self.B
        dXdt = np.hstack((f,fj))+B@U
        return dXdt
    
    def rk4(self,t,X,U):
        h = self.h
        k1 = self.dynamics(t,X,U)
        k2 = self.dynamics(t+h/2.,X+k1*h/2.,U)
        k3 = self.dynamics(t+h/2.,X+k2*h/2.,U)
        k4 = self.dynamics(t+h,X+k3*h,U)
        return t+h,X+h*(k1+2.*k2+2.*k3+k4)/6.

    def one_step_sim(self,t,X,U):
        Nrk = self.Nrk
        for num in range(0,Nrk):
             t,X = self.rk4(t,X,U)
        return t,X
    
    def initial_trajectory(self,final=0,x0s=None,xfs=None):
        try:
            X0s = np.hstack((x0s,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
            Xfs = np.hstack((xfs,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
            self.x0s = x0s
            self.xfs = xfs
        except:
            self.generate_environment(final)
            X0s = np.hstack((self.x0s,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
            Xfs = np.hstack((self.xfs,np.zeros((self.Nsc,self.q_states+self.n_add_states)))) 
        N = self.N
        dt = self.dt
        this = []
        Xshis = np.zeros((N+1,self.Nsc,self.n_states))
        Ushis = np.zeros((N,self.Nsc,self.m_inputs))
        Xchis = np.zeros((N+1,self.n_states))
        X0s = np.hstack((self.x0s,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
        Xfs = np.hstack((self.xfs,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
        r0 = self.r_leo
        v0 = np.sqrt(self.mu/r0)
        h0 = r0*v0
        Xc0 = self.Xc0
        Xs = X0s
        Xc = Xc0
        t = 0
        this.append(t)
        Xshis[0,:,:] = Xs
        Xchis[0,:] = Xc
        for i in range(N):
            tnow = t
            Xnow = np.zeros((self.Nsc,self.n_states))
            for p in range(self.Nsc):
                Xnow[p,:] = Xs[p,:]
            Us = np.zeros((self.Nsc,self.m_inputs))
            Us = self.safeu_global(Xc,Xs,Xfs,dt*5)
            for p in range(self.Nsc):
                Xj = Xs[p,:]
                X = np.hstack((Xc,Xj))
                U = Us[p,:]
                t,Xnext = self.one_step_sim(t,X,U)
                t = tnow
                Xs[p,:] = Xnext[6:12]
                Us[p,:] = U
            t += dt
            Xc = Xnext[0:6]
            this.append(t)
            Xshis[i+1,:,:] = Xs
            Ushis[i,:,:] = Us
            Xchis[i+1,:] = Xc
        this = np.array(this)
        Xshis_l = []
        Ushis_l = []
        for p in range(self.Nsc):
            Xshis_l.append(Xshis[:,p,:])
            Ushis_l.append(Ushis[:,p,:])
        plt.figure()
        thp = np.linspace(0,2*np.pi,100)
        for j in range(self.Nobs):
            plt.plot(self.xobs[j,:][0]+self.Robs*np.cos(thp),self.xobs[j,:][1]+self.Robs*np.sin(thp))
        for p in range(self.Nsc):
            #plt.plot(Xshis[:,p,0],Xshis[:,p,1])
            plt.plot(Xshis_l[p][:,0],Xshis_l[p][:,1])
            plt.scatter(self.x0s[p,:][0],self.x0s[p,:][1],color='b')
            plt.scatter(self.xfs[p,:][0],self.xfs[p,:][1],color='r')
        plt.figure()
        plt.plot(Xchis[:,0],Xchis[:,1])
        plt.show()
        self.Xshis_ini = Xshis_l
        self.Ushis_ini = Ushis_l
        self.Xchis_ini = Xchis
        return this,Xshis_l,Ushis_l,Xchis

    def scp(self,final=0):
        this,Xshis,Ushis,Xchis = self.initial_trajectory(final)
        XXdFs,UUdFs,cvx_status = self.SCPnormalized(Xshis,Ushis,Xchis,100,0)
        self.XXds = XXdFs
        self.UUds = UUdFs
        self.Xchis = Xchis
        self.cvx_status = cvx_status
        """
        plt.figure()
        thp = np.linspace(0,2*np.pi,100)
        for j in range(self.Nobs):
            plt.plot(self.xobs[j,:][0]+self.Robs*np.cos(thp),self.xobs[j,:][1]+self.Robs*np.sin(thp))
        for p in range(self.Nsc):
            plt.plot(self.XXds[p][:,0],self.XXds[p][:,1])
            plt.scatter(self.x0s[p,:][0],self.x0s[p,:][1],color='b')
            plt.scatter(self.xfs[p,:][0],self.xfs[p,:][1],color='r')
        plt.show()
        """
        pass

    def online_mpc(self,p_idx,i_idx,X,Xnow,Xc,I):
        XXn = []
        UUn = []
        for p in range(self.Nsc):
            XXn.append(self.Xshis_ini[p][i_idx:self.N+1,:])
            UUn.append(self.Ushis_ini[p][i_idx:self.N,:])
        Xchis = self.Xchis_ini[i_idx:self.N+1,:]
        """
        try:
            XXn = self.XXsn[i]
            UUn = self.UUsn[i]
        except:
            XXn = self.Xshis_ini
            UUn = self.Ushis_ini
        """
        XXmpc,UUmpc,Xs_mpc,Xfs_mpc,Nsc_mpc,idx_mpc = self.observation_mpc(X,Xnow,XXn,UUn)
        XX,UU,cvx_status = self.MPC_SCPj(XXmpc,UUmpc,Xchis,10,0,I,Nsc_mpc,0,Xs_mpc,Xfs_mpc,None,idx_mpc)
        #self.XXn[i] = XX
        #self.UUn[i] = UU
        self.XXsmpc[p_idx] = XX
        self.UUsmpc[p_idx] = UU
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot(XX[:,0],XX[:,1],XX[:,2])
        for j in range(self.Nobs):
            c_plot = plt.Circle((self.xobs[j,:][0],self.xobs[j,:][1]),self.Robs,fc="C7",ec="none",alpha=0.5)
            plt.gcf().gca().add_artist(c_plot)
        ax.scatter(self.x0s[p_idx,0],self.x0s[p_idx,1],self.x0s[p_idx,2])
        ax.scatter(self.xfs[p_idx,0],self.xfs[p_idx,1],self.xfs[p_idx,2])
        ax.set_xlim(-0.5,5.5)
        ax.set_ylim(-0.5,5.5)
        ax.set_zlim(-0.5,5.5)
        ax.grid()
        plt.show()
        U = self.clfqp_a_lagrange(X,XX[0,:],UU[0,:],Xc)
        return U
    
    def SCPnormalized(self,Xhiss,Uhiss,Xchis,epochs,Tidx):
        dt = self.dt
        N = self.N
        X0s = np.hstack((self.x0s,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
        Xfs = np.hstack((self.xfs,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
        cvx_optval = np.inf
        Uc = np.zeros(3)
        for i in range(epochs):
            XXs = []
            UUs = []
            constraints = []
            J = 0
            for p in range(self.Nsc):
                XXs.append(cp.Variable((N+1,X0s[0].size)))
                UUs.append(cp.Variable((N,self.m_inputs)))
            for p in range(self.Nsc):
                XX = XXs[p]
                UU = UUs[p]
                Xhis = Xhiss[p]
                Uhis = Uhiss[p]
                #pen = cp.Variable(nonneg=True)
                constraints += [XX[0,:] == X0s[p]]
                constraints += [XX[N,:] == Xfs[p]]
                Tregion = 0.1
                for k in range(N):
                    Xkp1 = XX[k+1,:]
                    Xk = XX[k,:]
                    Uk = UU[k,:]
                    Xs_k = Xhis[k,:]
                    Xck = Xchis[k,:]
                    C = self.SafetyConstraint(Xk,Xs_k,self.Robs+self.Rsafe)
                    for j in range(self.Nobs):
                        constraints += [-C[j] <= 0]
                    if Tidx == 1:
                        constraints += [cp.norm(Xk-Xs_k) <= Tregion]
                    constraints += [Xkp1 == self.DynamicsLD(dt,Xk,Uk,Xs_k,Xck,Uc)]
                    for q in range(self.Nsc):
                        Xqk = XXs[q][k,:]
                        Xsq_k = Xhiss[q][k,:]
                        Cpq = self.SafetyConstraintSC(Xk,Xs_k,Xqk,Xsq_k,self.Rsc+self.Rsafe)
                        constraints += [-Cpq <= 0]
                J += cp.sum_squares(UU)
            prob = cp.Problem(cp.Minimize(J),constraints)
            try:
                prob.solve(solver=cp.MOSEK)
                print(prob.status)
                print(J.value)
                Xhiss = []
                Uhiss = []
                for p in range(self.Nsc):
                    Xhiss.append(XXs[p].value)
                    Uhiss.append(UUs[p].value)
            except:
                prob.status = "infeasible"
            if prob.status == "infeasible":
                cvx_status = False
                break
            else:
                cvx_status = True
            if np.abs(cvx_optval-prob.value) <= 0.1:
                break
            cvx_optval = prob.value
        for p in range(self.Nsc):
            XXs[p] = XXs[p].value
            UUs[p] = UUs[p].value
        return XXs,UUs,cvx_status
    
    def MPC_SCPj(self,Xhiss,Uhiss,Xchis,epochs,Tidx,I,Nsc_mpc,Nobs_mpc,X0smpc,Xfsmpc,xobs,idx_mpc):
        dt = self.dt
        N = I
        cvx_optval = np.inf
        X0s = X0smpc
        Xfs = Xfsmpc
        Uc = np.zeros(3)
        Xhis = Xhiss[idx_mpc]
        Uhis = Uhiss[idx_mpc]
        print(Nsc_mpc)
        for i in range(epochs):
            XXs = []
            UUs = []
            constraints = []
            penalty = cp.Variable(1)
            J = 0
            XX = cp.Variable((N+1,X0s[0].size))
            UU = cp.Variable((N,self.m_inputs))
            constraints += [XX[0,:] == X0s[idx_mpc]]
            constraints += [XX[N,:] == Xfs[idx_mpc]]
            Tregion = 0.1
            for k in range(N):
                Xkp1 = XX[k+1,:]
                Xk = XX[k,:]
                Uk = UU[k,:]
                Xs_k = Xhis[k,:]
                Xck = Xchis[k,:]
                for j in range(Nobs_mpc):
                    xobsj = xobs[j,:]
                    C = self.SafetyConstraintj(Xk,Xs_k,xobsj,self.Robs+self.Rsafe)
                    constraints += [-C <= 0]
                if Tidx == 1:
                    constraints += [cp.norm(Xk-Xs_k) <= Tregion]
                constraints += [Xkp1 == self.DynamicsLD(dt,Xk,Uk,Xs_k,Xck,Uc)]
                for q in range(Nsc_mpc):
                    Xsq_k = Xhiss[q][k,:]
                    if q != idx_mpc:
                        Cpq = self.SafetyConstraintSC(Xk,Xs_k,Xsq_k,Xsq_k,self.Rsc+self.Rsafe)
                        constraints += [-Cpq <= penalty]
                J += cp.sum_squares(UU)
            prob = cp.Problem(cp.Minimize(J+100*penalty**2),constraints)
            try:
                prob.solve(solver=cp.MOSEK)
                print("penalty",penalty.value)
                print(prob.status)
                print(J.value)
                Xhis = XX.value
                Uhis = UU.value
            except:
                prob.status = "infeasible"
            if prob.status == "infeasible":
                cvx_status = False
                break
            else:
                cvx_status = True
            if np.abs(cvx_optval-prob.value) <= 0.1:
                break
            cvx_optval = prob.value
        XX = XX.value
        UU = UU.value
        return XX,UU,cvx_status
    
    def SafetyConstraint(self,Xk,Xs_k,R):
        Xobs = self.xobs
        C = []
        for i in range(Xobs.shape[0]):
            C.append((Xs_k[0:self.dim]-Xobs[i,:])@(Xk[0:self.dim]-Xobs[i,:])-R*cp.norm(Xs_k[0:self.dim]-Xobs[i,:]))
        return C

    def SafetyConstraintj(self,Xk,Xs_k,xobsj,R):
        C = (Xs_k[0:self.dim]-xobsj)@(Xk[0:self.dim]-xobsj)-R*cp.norm(Xs_k[0:self.dim]-xobsj)
        return C

    def SafetyConstraintSC(self,Xk,Xs_k,Xk2,Xs_k2,R):
        C = (Xs_k[0:self.dim]-Xs_k2[0:self.dim])@(Xk[0:self.dim]-Xk2[0:self.dim])-R*cp.norm(Xs_k[0:self.dim]-Xs_k2[0:self.dim])
        return C
    
    def DynamicsLD(self,dt,Xk,Uk,Xs_k,Xck,Uc):
        fn = self.relativef(Xs_k,Xck,Uc)
        An = self.relativeA(Xs_k,Xck,Uc)
        dXdt = fn+An@(Xk-Xs_k)+self.Bj@Uk
        Xkp1 = Xk+dXdt*dt
        return Xkp1    
    
    def simulation(self):
        XXds = self.XXds
        UUds = self.UUds
        Xc0 = self.Xc0
        XXhiss = []
        for p in range(self.Nsc):
            XXd0 = XXds[p][0,:]
            t = 0
            N = np.size(UUds[p][:,0])
            Xhis = np.zeros((N+1,self.n_states))
            Xhis[0,:] = XXd0
            X = XXd0
            Xc = Xc0
            for k in range(N):
                U = UUds[p][k,:]
                Xall = np.hstack((Xc,X))
                t,Xall = self.one_step_sim(t,Xall,U)
                Xc = Xall[0:6]
                X = Xall[6:12]
                Xhis[k+1,:] = X
            XXhiss.append(Xhis)
        fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        ax = Axes3D(fig)
        for p in range(self.Nsc):
            ax.plot(XXhiss[p][:,0],XXhiss[p][:,1],XXhiss[p][:,2])
            ax.scatter(self.x0s[p,:][0],self.x0s[p,:][1],self.x0s[p,:][2],color='b')
            ax.scatter(self.xfs[p,:][0],self.xfs[p,:][1],self.xfs[p,:][2],color='r')
        plt.show()
        return XXhiss,UUds
    
    def preprocess_data(self,file_path):
        xdata = np.load(file_path+"xdata.npy")
        udata = np.load(file_path+"udata.npy")
        self.Rsc = np.load(file_path+"Rscdata.npy")
        self.Rsafe = np.load(file_path+"Rscdata.npy")
        Nsc_data = xdata.shape[1]
        print(Nsc_data)
        Ndata = np.shape(xdata)[0]
        Nsamp = self.N
        n = np.shape(xdata)[3]
        m = np.shape(udata)[3]
        self.n = n
        self.m = m
        xdata2 = np.zeros((Ndata,Nsc_data,Nsamp,n))
        xfdata = np.zeros((Ndata,Nsc_data,Nsamp,n))
        xjdata = np.zeros((Ndata,Nsc_data,Nsamp,n*(Nsc_data-1)))
        for k in range(Ndata):
            for i in range(Nsc_data):
                xf = xdata[k,i,-1,:]
                for s in range(Nsamp):
                    x = xdata[k,i,s,:]
                    xdata2[k,i,s,:] = x
                    xfdata[k,i,s,:] = xf-x
                    ij = 0
                    xjs = np.zeros((Nsc_data-1,self.n_states))
                    for j in range(Nsc_data):
                        if j != i:
                            xj = xdata[k,j,s,:]
                            xjs[ij,:] = xj-x
                            ij += 1
                    idxj = np.argsort(np.sum(xjs**2,1))
                    for j in range(Nsc_data-1):
                        xjdata[k,i,s,self.n_states*j:self.n_states*(j+1)] = xjs[idxj[j],:]
        xdata2 = xdata2.reshape(-1,self.n_states)
        udata = udata.reshape(-1,self.m_inputs)
        xfdata = xfdata.reshape(-1,self.n_states)
        xjdata = xjdata.reshape(-1,self.n_states*(Nsc_data-1))
        N = Ndata*Nsc_data*Nsamp
        idx_d = np.random.choice(N,N,replace=False)
        xdata2 = xdata2[idx_d,:]
        udata = udata[idx_d,:]
        xfdata = xfdata[idx_d,:]
        xjdata = xjdata[idx_d,:]
        return xdata2,udata,xfdata,xjdata
    
    def preprocess_data_robust(self,file_path,idx_load):
        if idx_load == 0:
            xdata = np.load(file_path+"xdata.npy")
            udata = np.load(file_path+"udata.npy")
            Xchis = np.load(file_path+"Xchis.npy")
            self.Rsc = np.load(file_path+"Rscdata.npy")
            self.Rsafe = np.load(file_path+"Rscdata.npy")
            Nsc_data = xdata.shape[1]
            Ndata = np.shape(xdata)[0]
            Nsamp = self.N
            Nrobust = 100
            n = np.shape(xdata)[3]
            m = np.shape(udata)[3]
            self.n = n
            self.m = m
            xdata_r = np.zeros((Ndata,Nsc_data,Nsamp,Nrobust,n))
            udata_r = np.zeros((Ndata,Nsc_data,Nsamp,Nrobust,m))
            xfdata = np.zeros((Ndata,Nsc_data,Nsamp,Nrobust,n))
            xjdata = np.zeros((Ndata,Nsc_data,Nsamp,Nrobust,n*(Nsc_data-1)))
            for k in range(Ndata):
                for i in range(Nsc_data):
                    print(i)
                    for s in range(Nsamp):
                        Xc = Xchis[s,:]
                        xd = xdata[k,i,s,:]
                        ud = udata[k,i,s,:]
                        xf = xdata[k,i,-1,:]
                        for r in range(Nrobust):
                            x = xd+unifrand2(d_over=0.5/2,nk=self.n_states)
                            u = self.clfqp_a_lagrange(x,xd,ud,Xc)
                            xdata_r[k,i,s,r,:] = x
                            udata_r[k,i,s,r,:] = u
                            xfdata[k,i,s,r,:] = xf-x
                            ij = 0
                            xjs = np.zeros((Nsc_data-1,self.n_states))
                            for j in range(Nsc_data):
                                if j != i:
                                    xj = xdata[k,j,s,:]
                                    xjs[ij,:] = xj-x
                                    ij += 1
                            idxj = np.argsort(np.sum(xjs**2,1))
                            for j in range(Nsc_data-1):
                                xjdata[k,i,s,r,self.n_states*j:self.n_states*(j+1)] = xjs[idxj[j],:]
            xdata_r = xdata_r.reshape(-1,n)
            udata_r = udata_r.reshape(-1,m)
            xfdata = xfdata.reshape(-1,n)
            xjdata = xjdata.reshape(-1,n*(Nsc_data-1))
            N = Ndata*Nsc_data*Nsamp*Nrobust
            idx_d = np.random.choice(N,N,replace=False)
            xdata_r = xdata_r[idx_d,:]
            udata_r = udata_r[idx_d,:]
            xfdata = xfdata[idx_d,:]
            xjdata = xjdata[idx_d,:]
            np.save("data/leo/xdata_pp.npy",xdata_r)
            np.save("data/leo/udata_pp.npy",udata_r)
            np.save("data/leo/xfdata_pp.npy",xfdata)
            np.save("data/leo/xjdata_pp.npy",xjdata)
            self.xdata_r = xdata_r
            self.udata_r = udata_r
            self.xfdata = xfdata
            self.xjdata = xjdata
        else:
            self.xdata_r = np.load("data/leo/xdata_pp.npy")
            self.udata_r = np.load("data/leo/udata_pp.npy")
            self.xfdata = np.load("data/leo/xfdata_pp.npy")
            self.xjdata = np.load("data/leo/xjdata_pp.npy")
            n = np.shape(self.xdata_r)[1]
            m = np.shape(self.udata_r)[1]
            self.n = n
            self.m = m
        pass
    
    def preprocess_data_robust_ncm(self,file_path,idx_load,iTrain):
        if idx_load == 0:
            xdata = np.load(file_path+"xdata.npy")
            udata = np.load(file_path+"udata.npy")
            Xchis = np.load(file_path+"Xchis.npy")
            self.Rsc = np.load(file_path+"Rscdata.npy")
            self.Rsafe = np.load(file_path+"Rscdata.npy")
            Nsc_data = xdata.shape[1]
            Ndata = np.shape(xdata)[0]
            Nsamp = self.N
            Nrobust = 100
            n = np.shape(xdata)[3]
            m = np.shape(udata)[3]
            self.n = n
            self.m = m
            xdata_r = np.zeros((Ndata,Nsc_data,Nsamp,Nrobust,n))
            udata_r = np.zeros((Ndata,Nsc_data,Nsamp,Nrobust,m))
            xfdata = np.zeros((Ndata,Nsc_data,Nsamp,Nrobust,n))
            xjdata = np.zeros((Ndata,Nsc_data,Nsamp,Nrobust,n*(Nsc_data-1)))
            ncm = self.ncm_train(iTrain)
            for k in range(Ndata):
                for i in range(Nsc_data):
                    print(i)
                    for s in range(Nsamp):
                        Xc = Xchis[s,:]
                        xd = xdata[k,i,s,:]
                        ud = udata[k,i,s,:]
                        xf = xdata[k,i,-1,:]
                        for r in range(Nrobust):
                            x = xd+unifrand2(d_over=ncm.Jcv_opt,nk=self.n_states)
                            u = self.clfqp_a_ncm(x,xd,ud,Xc,ncm)
                            xdata_r[k,i,s,r,:] = x
                            udata_r[k,i,s,r,:] = u
                            xfdata[k,i,s,r,:] = xf-x
                            ij = 0
                            xjs = np.zeros((Nsc_data-1,self.n_states))
                            for j in range(Nsc_data):
                                if j != i:
                                    xj = xdata[k,j,s,:]
                                    xjs[ij,:] = xj-x
                                    ij += 1
                            idxj = np.argsort(np.sum(xjs**2,1))
                            for j in range(Nsc_data-1):
                                xjdata[k,i,s,r,self.n_states*j:self.n_states*(j+1)] = xjs[idxj[j],:]
            xdata_r = xdata_r.reshape(-1,n)
            udata_r = udata_r.reshape(-1,m)
            xfdata = xfdata.reshape(-1,n)
            xjdata = xjdata.reshape(-1,n*(Nsc_data-1))
            N = Ndata*Nsc_data*Nsamp*Nrobust
            idx_d = np.random.choice(N,N,replace=False)
            xdata_r = xdata_r[idx_d,:]
            udata_r = udata_r[idx_d,:]
            xfdata = xfdata[idx_d,:]
            xjdata = xjdata[idx_d,:]
            np.save("data/leo/xdata_pp.npy",xdata_r)
            np.save("data/leo/udata_pp.npy",udata_r)
            np.save("data/leo/xfdata_pp.npy",xfdata)
            np.save("data/leo/xjdata_pp.npy",xjdata)
            self.xdata_r = xdata_r
            self.udata_r = udata_r
            self.xfdata = xfdata
            self.xjdata = xjdata
        else:
            self.xdata_r = np.load("data/leo/xdata_pp.npy")
            self.udata_r = np.load("data/leo/udata_pp.npy")
            self.xfdata = np.load("data/leo/xfdata_pp.npy")
            self.xjdata = np.load("data/leo/xjdata_pp.npy")
            n = np.shape(self.xdata_r)[1]
            m = np.shape(self.udata_r)[1]
            self.n = n
            self.m = m
        pass
    
    def get_Xdata(self,xfdata,xjdata,Ndata):
        Xdata = []
        xjdata2 = np.zeros((Ndata,self.n_states*(self.Nsc-1)))
        Nsc_data = int(xjdata.shape[1]/self.n_states)+1
        for k in range(Ndata):
            xjs = xjdata[k,:]
            for p in range(Nsc_data-1):
                xj = xjs[self.n_states*p:self.n_states*(p+1)]
                if np.linalg.norm(xj[0:self.dim]) <= self.Rsen:
                    xjdata2[k,self.n_states*p:self.n_states*(p+1)] = xj
        for p in range(self.Nsc-1):
            Xdata.append(xjdata2[0:Ndata,self.n_states*p:self.n_states*(p+1)])
        Xdata.append(xfdata[0:Ndata,:])
        self.Xdata = Xdata
        pass
    
    def train(self,file_path,Nbatch=32,Nlayers=3,Nunits=100,Nepochs=10000,
              ValidationSplit=0.1,Patience=20,Verbose=1):
        xdata2,udata,xfdata,xjdata = self.preprocess_data(file_path)
        X = np.hstack((xfdata,xjdata))
        y = udata
        n = self.n
        m =self.m
        model = Sequential(name="NMPC")
        model.add(Dense(Nunits,activation="relu",input_shape=(n*self.Nsc,)))
        for l in range(Nlayers-1):
            model.add(Dense(Nunits,activation="relu"))
        model.add(Dense(m))
        model.summary()
        model.compile(loss="mean_squared_error",optimizer="adam")
        es = EarlyStopping(monitor="val_loss",patience=Patience)
        model.fit(X,y,batch_size=Nbatch,epochs=Nepochs,verbose=Verbose,\
                  callbacks=[es],validation_split=ValidationSplit)
        self.model = model
        model.save("data/leo/neuralnets/model.h5")
        pass
    
    def train_robust(self,file_path,idx_load=1,Nbatch=32,Nlayers=3,Nunits=100,Nepochs=10000,
              ValidationSplit=0.1,Patience=20,Verbose=1):
        self.preprocess_data_robust(file_path,idx_load)
        xdata_r = self.xdata_r
        udata_r = self.udata_r
        xfdata = self.xfdata
        xjdata = self.xjdata
        X = np.hstack((xfdata,xjdata))
        y = udata_r
        Ndata = 50000
        X = X[0:Ndata,:]
        y = y[0:Ndata,:]
        n = self.n
        m =self.m
        model = Sequential(name="NMPC")
        model.add(Dense(Nunits,activation="relu",input_shape=(n*self.Nsc,)))
        for l in range(Nlayers-1):
            model.add(Dense(Nunits,activation="relu"))
        model.add(Dense(m))
        model.summary()
        model.compile(loss="mean_squared_error",optimizer="adam")
        es = EarlyStopping(monitor="val_loss",patience=Patience)
        model.fit(X,y,batch_size=Nbatch,epochs=Nepochs,verbose=Verbose,\
                  callbacks=[es],validation_split=ValidationSplit)
        self.model = model
        model.save("data/leo/neuralnets/model_robust.h5")        
        pass

    def train_glas(self,file_path,Nbatch=32,Nphi=32,Nrho=32,Nlayers=3,Nunits=100,Nepochs=10000,
              ValidationSplit=0.1,Patience=20,Verbose=1):
        xdata2,udata,xfdata,xjdata = self.preprocess_data(file_path)
        Ndata = xdata2.shape[0]
        self.get_Xdata(xfdata,xjdata,Ndata)
        X = self.Xdata
        y = udata
        y = y[0:Ndata,:]
        n = self.n
        m =self.m
        inp = []
        phinet = Sequential(name="phi")
        phinet.add(Dense(Nunits,activation='relu',input_shape=(self.n_states,)))
        phinet.add(Dense(Nphi,activation=None))
        rhonet = Sequential(name="rho")
        rhonet.add(Dense(Nunits,activation='relu',input_shape=(Nphi,)))
        rhonet.add(Dense(Nrho,activation=None))
        phi = []
        for j in range(self.Nsc-1):
            inp.append(Input(shape=(self.n_states,)))
            phi.append(phinet(inp[j]))
        inputx0f = Input(shape=(n,))
        inp.append(inputx0f)
        outobs = rhonet(Add()(phi))
        psinet = Sequential(name="psi")
        psinet.add(Dense(Nunits,activation='relu',input_shape=(Nrho+n,)))
        for l in range(Nlayers):
            psinet.add(Dense(Nunits,activation="relu"))
            #psinet.add(Dropout(rate=0.2))
        psinet.add(Dense(m))
        out = psinet(Concatenate(axis=1)([outobs,inputx0f]))
        model = Model(inputs=[inp],outputs=out)
        model.summary()
        model.compile(loss="mean_squared_error",optimizer="adam")
        es = EarlyStopping(monitor="val_loss",patience=Patience)
        model.fit(X,y,batch_size=Nbatch,epochs=Nepochs,verbose=1,\
                  callbacks=[es],validation_split=ValidationSplit)
        self.model = model
        model.save("data/leo/neuralnets/model_glas.h5")
        pass
    
    def train_robust_glas(self,file_path,idx_load=1,Nbatch=32,Nphi=32,Nrho=32,Nlayers=3,Nunits=100,Nepochs=10000,
              ValidationSplit=0.1,Patience=20,Verbose=1,iTrain=0):
        self.preprocess_data_robust(file_path,idx_load)
        xdata_r = self.xdata_r
        udata_r = self.udata_r
        xfdata = self.xfdata
        xjdata = self.xjdata
        Ndata = 50000
        self.get_Xdata(xfdata,xjdata,Ndata)
        X = self.Xdata
        y = udata_r
        y = y[0:Ndata,:]
        n = self.n
        m =self.m
        inp = []
        phinet = Sequential(name="phi")
        phinet.add(Dense(Nunits,activation='relu',input_shape=(self.n_states,)))
        phinet.add(Dense(Nphi,activation=None))
        rhonet = Sequential(name="rho")
        rhonet.add(Dense(Nunits,activation='relu',input_shape=(Nphi,)))
        rhonet.add(Dense(Nrho,activation=None))
        phi = []
        for j in range(self.Nsc-1):
            inp.append(Input(shape=(self.n_states,)))
            phi.append(phinet(inp[j]))
        inputx0f = Input(shape=(n,))
        inp.append(inputx0f)
        outobs = rhonet(Add()(phi))
        psinet = Sequential(name="psi")
        psinet.add(Dense(Nunits,activation='relu',input_shape=(Nrho+n,)))
        for l in range(Nlayers):
            psinet.add(Dense(Nunits,activation="relu"))
            #psinet.add(Dropout(rate=0.2))
        psinet.add(Dense(m))
        out = psinet(Concatenate(axis=1)([outobs,inputx0f]))
        model = Model(inputs=[inp],outputs=out)
        model.summary()
        model.compile(loss="mean_squared_error",optimizer="adam")
        es = EarlyStopping(monitor="val_loss",patience=Patience)
        model.fit(X,y,batch_size=Nbatch,epochs=Nepochs,verbose=1,\
                  callbacks=[es],validation_split=ValidationSplit)
        self.model = model
        model.save("data/leo/neuralnets/model_robust_glas.h5")
        pass

    def lagrangeMCGD(self,Xj,Xc,Uc):
        mu = self.mu
        kJ2 = self.kJ2
        ome = self.ome
        Cd = self.Cd
        mass = self.mass
        rho = self.rho
        Area = self.Area
        Cdj = self.Cdj
        massj = self.massj
        rhoj = self.rhoj
        Areaj = self.Areaj
        C = Cd*(Area/mass)*rho/2
        Cj = Cdj*(Areaj/massj)*rhoj/2
        xj = Xj[0]
        yj = Xj[1]
        zj = Xj[2]
        xjdot = Xj[3]
        yjdot = Xj[4]
        zjdot = Xj[5]
        r = Xc[0]
        vx = Xc[1]
        h = Xc[2]
        Om = Xc[3]
        i = Xc[4]
        th = Xc[5]
        uR = Uc[0]
        uT = Uc[1]
        uN = Uc[2]
        dXcdt = self.chieff(Xc,Uc)
        Omdot = dXcdt[3]
        idot = dXcdt[4]
        thdot = dXcdt[5]
        si = np.sin(i)
        ci = np.cos(i)
        s2i = np.sin(2*i)
        sth = np.sin(th)
        cth = np.cos(th)
        s2th = np.sin(2*th)
        lj = Xj[0:3]
        ljdot = Xj[3:6]
        Va = np.array([vx,h/r-ome*r*ci,ome*r*cth*si])
        va = np.linalg.norm(Va)
        omx = idot*cth+Omdot*sth*si+r/h*uN
        omz = thdot+Omdot*ci
        om_vec = np.array([omx,0,omz])
        Vaj = Va+ljdot+np.cross(om_vec,lj)
        vaj = np.linalg.norm(Vaj)
        rj = np.sqrt((r+xj)**2+yj**2+zj**2)
        rjZ = (r+xj)*si*sth+yj*si*cth+zj*ci
        zt = 2*kJ2*si*sth/r**4
        ztj = 2*kJ2*rjZ/rj**5
        alx = -kJ2*s2i*cth/r**5+3*vx*kJ2*s2i*sth/r**4/h-8*kJ2**2*si**3*ci*sth**2*cth/r**6/h**2
        alz = -2*h*vx/r**3-kJ2*si**2*s2th/r**5
        et = np.sqrt(mu/r**3+kJ2/r**5-5*kJ2*si**2*sth**2/r**5)
        etj = np.sqrt(mu/rj**3+kJ2/rj**5-5*kJ2*rjZ**2/rj**7)
        xjddot = 2*yjdot*omz-xj*(etj**2-omz**2)+yj*alz-zj*omx*omz-(ztj-zt)*si*sth-r*(etj**2-et**2)-Cj*vaj*(xjdot-yj*omz)-(Cj*vaj-C*va)*vx-uR
        yjddot = -2*xjdot*omz+2*zjdot*omx-xj*alz-yj*(etj**2-omz**2-omx**2)+zj*alx-(ztj-zt)*si*cth-Cj*vaj*(yjdot+xj*omz-zj*omx)-(Cj*vaj-C*va)*(h/r-ome*r*ci)-uT
        zjddot = -2*yjdot*omx-xj*omx*omz-yj*alx-zj*(etj**2-omx**2)-(ztj-zt)*ci-Cj*vaj*(zjdot+yj*omx)-(Cj*vaj-C*va)*ome*r*cth*si-uN
        Mmat = np.identity(self.q_states)
        Cmat = -2*np.array([[0,omz,0],[-omz,0,omx],[0,-omx,0]])
        Gmat = np.array([[etj**2-omz**2,-alz,omx*omz],[alz,etj**2-omz**2-omx**2,-alx],[omx*omz,alx,etj**2-omx**2]])@Xj[0:3]+(ztj-zt)*np.array([si*sth,si*cth,ci])+np.array([r*(etj**2-et**2),0,0])
        Dmat = -np.array([-Cj*vaj*(xjdot-yj*omz)-(Cj*vaj-C*va)*vx,-Cj*vaj*(yjdot+xj*omz-zj*omx)-(Cj*vaj-C*va)*(h/r-ome*r*ci),-Cj*vaj*(zjdot+yj*omx)-(Cj*vaj-C*va)*ome*r*cth*si])
        return Mmat,Cmat,Gmat,Dmat

    def clfqp_a(self,X,Xd,Ud,Xc):
        e = X-Xd
        A = self.A
        B =self.Bj
        Bd =self.Bj
        Uc = np.zeros(3)
        f = self.relativef(X,Xc,Uc)+B@Ud
        fd = self.relativef(Xd,Xc,Uc)+Bd@Ud
        Q = np.identity(self.n_states)*10
        R = np.identity(self.m_inputs)
        K,P,_ = control.lqr(A,B,Q,R)
        U = cp.Variable(self.m_inputs)
        invR = np.linalg.inv(R)
        phi0 = 2*e@P@(f-fd)+e@(P@B@invR@B.T@P+Q)@e
        phi1 = (2*e@P@B).T
        if phi0 > 0:
            U = -(phi0/(phi1.T@phi1))*phi1
            U = Ud+U.ravel()
        else:
            U = Ud
        return U

    def clfqp_a_lagrange(self,X,Xd,Ud,Xc):
        K = np.identity(self.q_states)*0.5
        Lam = np.identity(self.q_states)
        Uc = np.zeros(3)
        M,C,G,D = self.lagrangeMCGD(X,Xc,Uc)
        Md,Cd,Gd,Dd = self.lagrangeMCGD(Xd,Xc,Uc)
        q = X[0:3]
        qd = Xd[0:3]
        q_dot = X[3:6]
        qd_dot = Xd[3:6]
        #q_ddot = np.linalg.solve(M,-C@q_dot-G-D)
        qd_ddot = np.linalg.solve(Md,-Cd@qd_dot-Gd-Dd)
        qr_dot = qd_dot-Lam@(q-qd)
        qr_ddot = qd_ddot-Lam@(q_dot-qd_dot)
        s = q_dot-qr_dot
        phi0 = 2*s@Ud-2*s@(M@qr_ddot+C@qr_dot+G+D)+2*s@K@s
        phi1 = 2*s
        if phi0 > 0:
            U = -(phi0/(phi1.T@phi1))*phi1
            U = Ud+U.ravel()
        else:
            U = Ud
        return U

    def clfqp_a_lagrange_mpc(self,X,Xd,Ud,Xc):
        K = np.identity(self.q_states)*1
        Lam = np.identity(self.q_states)
        Uc = np.zeros(3)
        M,C,G,D = self.lagrangeMCGD(X,Xc,Uc)
        Md,Cd,Gd,Dd = self.lagrangeMCGD(Xd,Xc,Uc)
        q = X[0:3]
        qd = Xd[0:3]
        q_dot = X[3:6]
        qd_dot = Xd[3:6]
        #q_ddot = np.linalg.solve(M,-C@q_dot-G-D)
        qd_ddot = np.linalg.solve(Md,-Cd@qd_dot-Gd-Dd)
        qr_dot = qd_dot-Lam@(q-qd)
        qr_ddot = qd_ddot-Lam@(q_dot-qd_dot)
        s = q_dot-qr_dot
        phi0 = 2*s@Ud-2*s@(M@qr_ddot+C@qr_dot+G+D)+2*s@K@s
        phi1 = 2*s
        if phi0 > 0:
            U = -(phi0/(phi1.T@phi1))*phi1
            U = Ud+U.ravel()
        else:
            U = Ud
        return U

    def clfqp_a_lagrange_test(self,X,Xd,Ud,Xc):
        K = np.identity(self.q_states)
        Lam = np.identity(self.q_states)
        Uc = np.zeros(3)
        M,C,G,D = self.lagrangeMCGD(X,Xc,Uc)
        Md,Cd,Gd,Dd = self.lagrangeMCGD(Xd,Xc,Uc)
        q = X[0:3]
        qd = Xd[0:3]
        q_dot = X[3:6]
        qd_dot = Xd[3:6]
        #q_ddot = np.linalg.solve(M,-C@q_dot-G-D)
        qd_ddot = np.linalg.solve(Md,-Cd@qd_dot-Gd-Dd)
        qr_dot = qd_dot-Lam@(q-qd)
        qr_ddot = qd_ddot-Lam@(q_dot-qd_dot)
        s = q_dot-qr_dot
        U = cp.Variable(3)
        constraints = [2*s@(-G-D-M@qr_ddot-C@qr_dot+U) <= -2*s@K@s]
        prob = cp.Problem(cp.Minimize(cp.norm(U-Ud)**2),constraints)
        prob.solve(solver=cp.MOSEK)
        return U.value

    def clfqp_a_ncm(self,X,Xd,Ud,Xc,ncm):
        e = X-Xd
        A = self.A
        B =self.Bj
        Bd =self.Bj
        Uc = np.zeros(3)
        f = self.relativef(X,Xc,Uc)+B@Ud
        fd = self.relativef(Xd,Xc,Uc)+Bd@Ud
        M = ncm.ncm(X,Xc)
        alp = ncm.alp_opt
        U = cp.Variable(self.m_inputs)
        phi0 = 2*e@M@(f-fd)+2*alp*e@M@e
        phi1 = (2*e@M@B).T
        if phi0 > 0:
            U = -(phi0/(phi1.T@phi1))*phi1
            U = Ud+U.ravel()
        else:
            U = Ud
        return U
    
    def observation(self,i,X,Xf,Xnow):
        Xfi = Xf-X
        ij = 0
        Xjs = np.zeros((self.Nsc-1,self.n_states))
        for j in range(self.Nsc):
            if j != i:
                Xj = Xnow[j,:]
                Xjs[ij,:] = Xj-X
                ij += 1
        idxj = np.argsort(np.sum(Xjs**2,1))
        Xji = np.zeros(self.n_states*(self.Nsc-1))
        for j in range(self.Nsc-1):
            Xji[self.n_states*j:self.n_states*(j+1)] = Xjs[idxj[j],:]
        Xout = np.hstack((Xfi,Xji))
        return np.array([Xout])
    
    def observation_glas(self,i,X,Xf,Xnow):
        Xfi = Xf-X
        ij = 0
        Xjs = np.zeros((self.Nsc-1,self.n_states))
        for j in range(self.Nsc):
            if j != i:
                Xj = Xnow[j,:]
                Xjs[ij,:] = Xj-X
                ij += 1
        idxj = np.argsort(np.sum(Xjs**2,1))
        Xji = np.zeros(self.n_states*(self.Nsc-1))
        for j in range(self.Nsc-1):
            if np.linalg.norm(Xjs[idxj[j],:][0:self.dim]) <= self.Rsen:
                Xji[self.n_states*j:self.n_states*(j+1)] = Xjs[idxj[j],:]
        Xout = []
        for j in range(self.Nsc-1):
            Xout.append(np.array([Xji[self.n_states*j:self.n_states*(j+1)]]))
        Xout.append(np.array([Xfi]))
        return Xout
    
    def observation_mpc(self,X,Xnow,XXn,UUn):
        p = X[0:self.dim]
        Xs_mpc = []
        Xfs_mpc = []
        XXmpc = []
        UUmpc = []
        X0s = np.hstack((self.x0s,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
        Xfs = np.hstack((self.xfs,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
        ij = 0
        for j in range(self.Nsc):
            pj = Xnow[j,0:self.dim]
            if np.linalg.norm(p-pj) <= self.Rsen:
                if np.linalg.norm(p-pj) == 0:
                    idx_mpc = ij
                    print(ij)
                ij += 1
                Xs_mpc.append(Xnow[j,:])
                Xfs_mpc.append(Xfs[j,:])
                XXmpc.append(XXn[j])
                UUmpc.append(UUn[j])
        Nsc_mpc = len(Xs_mpc)
        Xs_mpc = np.array(Xs_mpc)
        Xfs_mpc = np.array(Xfs_mpc)
        return XXmpc,UUmpc,Xs_mpc,Xfs_mpc,Nsc_mpc,idx_mpc
    
    def nmpc_simulation(self,dover,XXds=None,UUds=None,dtc0_scp=None,x0s=None,xfs=None,final=0):
        self.XXsmpc = [0]*self.Nsc
        self.UUsmpc = [0]*self.Nsc
        model = load_model("data/leo/neuralnets/model_glas.h5")
        model_r = load_model("data/leo/neuralnets/model_robust_glas.h5")
        #self.generate_environment(final=1)
        if dtc0_scp == None:
            tc0 = time.time()
            self.scp(final)
            dtc0_scp = time.time()-tc0
            XXds = self.XXds
            UUds = self.UUds
        else:
            dtc0_scp = 100
            self.x0s = x0s
            self.xfs = xfs
        self.initial_trajectory(final,self.x0s,self.xfs)
        N = int(self.N*5)
        dt = self.dt
        this = []
        Ncon = 4
        Xshis = np.zeros((N+1,Ncon,self.Nsc,self.n_states))
        Ushis = np.zeros((N,Ncon,self.Nsc,self.m_inputs))
        Xchis = np.zeros((N+1,self.n_states))
        dthis = np.zeros((N,Ncon,self.Nsc))
        X0s = np.hstack((self.x0s,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
        Xfs = np.hstack((self.xfs,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
        
        Xc0 = self.Xc0
        Xc = Xc0
        t = 0
        this.append(t)
        XXs = np.zeros((Ncon,self.Nsc,self.n_states))
        eps = 0.2
        for c in range(Ncon):
            XXs[c,:,:] = X0s
            Xshis[0,c,:,:] = X0s
        Xchis[0,:] = Xc
        idx_fin = np.zeros((Ncon,self.Nsc))
        Nends = np.ones((Ncon,self.Nsc))*N
        for i in range(N):
            tnow = t
            XXnow = np.zeros((Ncon,self.Nsc,self.n_states))
            for c in range(Ncon):
                for p in range(self.Nsc):
                    XXnow[c,p,:] = XXs[c,p,:]
            UUs = np.zeros((Ncon,self.Nsc,self.m_inputs))
            ds = np.zeros((Nsc,self.n_states))
            for p in range(Nsc):
                d = unifrand2(dover,self.q_states)
                d = np.hstack((np.zeros(self.q_states),d))
                ds[p,:] = d
            for c in range(Ncon):
                Xnow = XXnow[c,:,:]
                Xs = XXs[c,:,:]
                #Xfs = self.assignment(Xs)
                self.xfs = Xfs[:,0:self.q_states]
                Uds = np.zeros((self.Nsc,self.m_inputs))
                for p in range(self.Nsc):
                    if idx_fin[c,p] == 0:
                        X = Xs[p,:]
                        Xf = Xfs[p,:]
                        #XX = self.observation(p,X,Xf,Xnow)
                        XX = self.observation_glas(p,X,Xf,Xnow)
                        if c == 0:
                            tc0 = time.time()
                            U = model.predict(XX)
                            dtc0 = time.time()-tc0
                        elif c == 1:
                            tc0 = time.time()
                            U = model_r.predict(XX)
                            dtc0 = time.time()-tc0
                        elif c == 2:
                            if (i%50 == 0) & (i < self.N):
                                I = self.N-i
                                if I <= 0:
                                    I = 10
                                tc0 = time.time()
                                U = self.online_mpc(p,i,X,Xnow,Xc,I)
                                dtc0 = time.time()-tc0
                                print(dtc0)
                                Ith = i
                            else:
                                if (i < self.N):
                                    impc = i-Ith
                                    tc0 = time.time()
                                    U = self.clfqp_a_lagrange_mpc(X,self.XXsmpc[p][impc,:],self.UUsmpc[p][impc,:],Xc)
                                    dtc0 = time.time()-tc0
                                else:
                                    tc0 = time.time()
                                    U = self.clfqp_a_lagrange_mpc(X,self.XXsmpc[p][-1,:],self.UUsmpc[p][-1,:],Xc)       
                                    dtc0 = time.time()-tc0
                        elif c == 3:
                            tc0 = time.time()
                            if (i < self.N):
                                U = self.clfqp_a_lagrange_mpc(X,XXds[p][i,:],UUds[p][i,:],Xc)
                            else:
                                U = self.clfqp_a_lagrange_mpc(X,XXds[p][-1,:],UUds[p][-1,:],Xc)
                            dtc0 = time.time()-tc0+dtc0_scp
                        U = U.ravel()
                        Uds[p,:] = U
                        dthis[i,c,p] = dtc0
                for p in range(self.Nsc):
                    if idx_fin[c,p] == 0:
                        X = Xs[p,:]
                        ts0 = time.time()
                        U = self.safeu(p,X,Xnow,Uds,Xc,dt*5)
                        dts0 = time.time()-ts0
                        dthis[i,c,p] += dts0
                        #U = self.safeuMPC(p,X,Xnow,Xfs,Uds,model_r,dt*2)
                        #U = Uds[p,:]
                        #print(U)
                        Xall = np.hstack((Xc,X))
                        t,Xall = self.one_step_sim(t,Xall,U)
                        d = ds[p,:]
                        X = Xall[6:12]
                        X += d*dt
                        t = tnow
                        XXs[c,p,:] = X
                        UUs[c,p,:] = U
                        if np.linalg.norm(X[0:3]-Xf[0:3]) <= eps:
                            idx_fin[c,p] = 1
                            Nends[c,p] = i
                            print(idx_fin)
                            print(Nends)
            t += dt
            this.append(t)
            Xshis[i+1,:,:,:] = XXs
            Ushis[i,:,:,:] = UUs
            Xc = Xall[0:6]
            Xchis[i+1,:] = Xc
            if np.sum(idx_fin) == Ncon*self.Nsc:
                print("FIN")
                break
        this = np.array(this)
        ueffs = []
        for c in range(Ncon):
            Xhisc = Xshis[:,c,:,:]
            ueffc = 0
            fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')
            ax = Axes3D(fig)
            for p in range(self.Nsc):
                Nend = int(Nends[c,p])
                #ueffc += np.sum(np.sqrt(np.sum(Ushis[0:Nend,c,p,:]**2,1)))
                ueffc += np.sum(Ushis[0:Nend,c,p,:]**2)
                #plt.plot(Xshis[:,p,0],Xshis[:,p,1])
                ax.plot(Xhisc[0:Nend+1,p,0],Xhisc[0:Nend+1,p,1],Xhisc[0:Nend+1,p,2])
                #ax.scatter(self.x0s[p,:][0],self.x0s[p,:][1],self.x0s[p,:][2],color='b')
                #ax.scatter(self.xfs[p,:][0],self.xfs[p,:][1],self.xfs[p,:][2],color='r')
            ueffs.append(ueffc)
            #ax.set_xlim(-0.5,5.5)
            #ax.set_ylim(-0.5,5.5)
            #ax.set_zlim(-0.5,5.5)
            ax.grid()
            r = np.linalg.norm(self.xfs[0,:]-np.array([self.xlims[1]/2,self.ylims[1]/2,self.zlims[1]/2]))
            th = np.linspace(0,2*np.pi,100)
            phi = np.linspace(0,np.pi,100)
            th,phi = np.meshgrid(th,phi)
            x = np.cos(th)*np.sin(phi)*r+self.xlims[1]/2
            y = np.sin(th)*np.sin(phi)*r+self.ylims[1]/2
            z = np.cos(phi)*r+self.zlims[1]/2
            
            ax.scatter(self.xfs[:,0],self.xfs[:,1],self.xfs[:,2])
            ax.scatter(self.x0s[:,0],self.x0s[:,1],self.x0s[:,2])
            ax.plot_surface(x,y,z,alpha=0.2)
            plt.show()
            plt.show()
        print(ueffs)
        print(idx_fin)
        """
        figs = []
        ims = [[0]*(N+1) for i in range(Ncon)]
        for c in range(Ncon):
            fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')
            ax = Axes3D(fig)
            for k in range(N+1):
                im = []
                for p in range(self.Nsc):                   
                    Nend = int(Nends[c,p])
                    colorp = "C"+str(p)
                    if k <= Nend+1:
                        im.append(ax.scatter(Xshis[k,c,p,0],Xshis[k,c,p,1],Xshis[k,c,p,2],color=colorp))
                        im += ax.plot(Xshis[0:k,c,p,0],Xshis[0:k,c,p,1],Xshis[0:k,c,p,2],color=colorp) # plot returns list
                    else:
                        im.append(ax.scatter(Xshis[Nend+1,c,p,0],Xshis[Nend+1,c,p,1],Xshis[Nend+1,c,p,2],color=colorp))
                        im += ax.plot(Xshis[0:Nend+1,c,p,0],Xshis[0:Nend+1,c,p,1],Xshis[0:Nend+1,c,p,2],color=colorp) # plot returns list
                    im.append(ax.scatter(self.x0s[p,0],self.x0s[p,1],self.x0s[p,2],facecolors='w',edgecolors=colorp)) # somehow facecolor = 'none' did not work with ax
                    im.append(ax.scatter(self.xfs[p,0],self.xfs[p,1],self.xfs[p,2],color=colorp,marker="x"))
                ims[c][k] = im
            ax.set_title("sup||d(t)|| = "+str(dover))
            ax.grid()
            ax.set_xlim(-0.5,5.5)
            ax.set_ylim(-0.5,5.5)
            ax.set_zlim(-0.5,5.5)
            figs.append(fig)
        for c in range(Ncon):
            ani = animation.ArtistAnimation(figs[c],ims[c],interval=dt*1000)
            ani.save("movies/leo/output"+str(c)+"_"+str(dover*10)+".mp4",writer="ffmpeg")    
        """
        return this,dthis,Xshis,Ushis,ueffs,idx_fin,Nends
    
def unifrand2(d_over=1,nk=4):
    d_over_out = d_over+1
    while d_over_out > d_over:
            d = np.random.uniform(-d_over,d_over,size=nk)
            d_over_out = np.linalg.norm(d)
    return d
    
if __name__ == "__main__":
    Nobs = 0
    Nsc = 10
    Nsim = 10
    """
    xdata = []
    udata = []
    fails = []
    for i in range(Nsim):
        start_time = time.time()
        #ncmp.preprocess_data(False)
        #ncmp.train()
        while True:
            nmpc = NMPC(Nobs,Nsc)
            nmpc.scp()
            print(nmpc.cvx_status)
            if nmpc.cvx_status:
                break
            else:
                fails.append(i)
        xhis,uhis = nmpc.simulation()
        xdata.append(np.array(xhis))
        udata.append(np.array(uhis))
        np.save("data/leo/xdata.npy",np.array(xdata))
        np.save("data/leo/udata.npy",np.array(udata))
        np.save("data/leo/Rscdata.npy",nmpc.Rsc)
        np.save("data/leo/Rsafedata.npy",nmpc.Rsafe)
        np.save("data/leo/Xchis.npy",nmpc.Xchis)
        t_one_epoch = time.time()-start_time
        print("time:",t_one_epoch,"s")
        print("i =",i)
    xdata = np.array(xdata)
    udata = np.array(udata)
    np.save("data/leo/xdata.npy",np.array(xdata))
    np.save("data/leo/udata.npy",np.array(udata))
    np.save("data/leo/Rscdata.npy",nmpc.Rsc)
    np.save("data/leo/Rsafedata.npy",nmpc.Rsafe)
    np.save("data/leo/Xchis.npy",nmpc.Xchis)
    """
    
    Nobs = 0
    Nsc = 10
    nmpc = NMPC(Nobs,Nsc)
    
    file_path = "data/leo/"
    #nmpc.train_robust_glas(file_path,idx_load=0,iTrain=1)
    #nmpc.train_robust_glas(file_path,idx_load=0)
    #nmpc.train_glas(file_path)
    """
    dovers = [0,0.2,0.4,0.6,0.8,1.0]
    dovers = [1.0]
    for dover in dovers:
        this,dthis,Xshis,Ushis,ueffs,idx_fin,Nends = nmpc.nmpc_simulation(dover)
    """
    
    Ndata = 50
    dover = 1.0
    thiss = []
    dthiss = []
    Xshiss = []
    Ushiss = []
    ueffss = []
    idx_fins = []
    Nendss = []
    x0ss = []
    xfss = []
    
    nmpc = NMPC(Nobs,Nsc)
    this,dthis,Xshis,Ushis,ueffs,idx_fin,Nends = nmpc.nmpc_simulation(dover=0.8,final=1)
    np.save("data/leo/sim/plot/thiss.npy",this)
    np.save("data/leo/sim/plot/dthiss.npy",dthis)
    np.save("data/leo/sim/plot/Xshiss.npy",Xshis)
    np.save("data/leo/sim/plot/Ushiss.npy",Ushis)
    np.save("data/leo/sim/plot/ueffss.npy",ueffs)
    np.save("data/leo/sim/plot/idx_fins.npy",idx_fin)
    np.save("data/leo/sim/plot/Nendss.npy",Nends)
    np.save("data/leo/sim/plot/x0ss.npy",nmpc.x0s)
    np.save("data/leo/sim/plot/xfss.npy",nmpc.xfs)