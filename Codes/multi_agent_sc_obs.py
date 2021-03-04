#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:56:53 2020

@author: hiroyasu
"""

import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt
from matplotlib import animation
import control
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Lambda
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Concatenate
import matplotlib.patches as patches

from classncm import NCM

np.random.seed(seed=8) # for 43 sampels
#np.random.seed(seed=2) # for 7 samples

class NMPC():
    def __init__(self,Nobs,Nsc):
        self.m = 1
        self.I = 1
        self.l = 1
        self.b = 1
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
        self.n_states = 6
        self.n_add_states = 1
        self.m_inputs = 8
        self.q_states = self.n_states//2
        self.dim = 2
        self.Xlims = np.array([-np.ones(self.n_states),np.ones(self.n_states)])*10
        Z = np.zeros((self.q_states,self.q_states))
        eyeq = np.identity(self.q_states)
        self.A = np.vstack((np.hstack((Z,eyeq)),np.hstack((Z,Z))))
        Q = np.identity(self.n_states)*1
        R = np.identity(self.m_inputs)
        X0 = np.array([0,0,np.pi/6,0,0,np.pi/3])
        K,P,_ = control.lqr(self.A,self.GetB(X0),Q,R)
        self.P = P
        self.K = K
        self.Q = Q
        self.R = R
        self.dt = 0.1
        self.Nrk = 10
        self.h = self.dt/self.Nrk
        self.N = 150
        self.umin = 0
        self.fname_ncm = "NCM_sc_obs"
    
    def generate_endpoints(self):
        xlims = self.xlims
        ylims = self.ylims
        d_btw_0f = self.d_btw_0f
        Nsc = self.Nsc
        Rsc = self.Rsc
        Rsafe = self.Rsafe
        x0s = np.zeros((Nsc,self.dim))
        xfs = np.zeros((Nsc,self.dim))
        for p in range(Nsc):
            while True:
                pxk0 = np.random.uniform(xlims[0],xlims[1])
                pyk0 = np.random.uniform(ylims[0],ylims[1])
                x0 = np.array([pxk0,pyk0])
                pxkf = np.random.uniform(xlims[0],xlims[1])
                pykf = np.random.uniform(ylims[0],ylims[1])
                xf = np.array([pxkf,pykf])
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
        pass

    def generate_environment(self):
        Nobs = self.Nobs
        xlims = self.xlims
        ylims = self.ylims
        d_btw_obs = self.d_btw_obs
        self.generate_endpoints()
        Robs = self.Robs
        xobs = []
        k = 0
        while len(xobs) < Nobs:
            pxoi = np.random.uniform(xlims[0],xlims[1])
            pyoi = np.random.uniform(ylims[0],ylims[1])
            xoi = np.array([pxoi,pyoi])
            dsmin = []
            for p in range(self.Nsc):
                x0 = self.x0s[p,:]
                xf = self.xfs[p,:]
                ds = [np.linalg.norm(xoi-x0)+d_btw_obs,\
                      np.linalg.norm(xoi-xf)+d_btw_obs]
                for xo in xobs:
                    ds.append(np.linalg.norm(xoi-xo))
                dsmin.append(min(ds))
            d = min(dsmin)
            if (d > 2*Robs+d_btw_obs):
                xobs.append(xoi)
            k += 1
            if k == 100:
                self.generate_endpoints()
                xobs = []
                k = 0
        self.xobs = np.array(xobs)
        pass
    
    def GetB(self,X):
        m = self.m
        I = self.I
        l = self.l
        b = self.b
        th = X[2]
        T = np.array([[np.cos(th)/m,np.sin(th)/m,0],[-np.sin(th)/m,np.cos(th)/m,0],[0,0,1/2./I]])
        H = np.array([[-1,-1,0,0,1,1,0,0],[0,0,-1,-1,0,0,1,1],[-l,l,-b,b,-l,l,-b,b]])
        B = np.vstack((np.zeros((self.q_states,self.m_inputs)),T@H))
        return B

    def GetdBdX(self,X):
        m = self.m
        l = self.l
        b = self.b
        th = X[2]
        T = np.array([[-np.sin(th)/m,np.cos(th)/m,0],[-np.cos(th)/m,-np.sin(th)/m,0],[0,0,0]])
        H = np.array([[-1,-1,0,0,1,1,0,0],[0,0,-1,-1,0,0,1,1],[-l,l,-b,b,-l,l,-b,b]])
        dBdX = np.vstack((np.zeros((self.q_states,self.m_inputs)),T@H))
        return dBdX

    def ncm_train(self,iTrain):
        # enter your choice of CV-STEM sampling period
        dt = 1
        # specify upper and lower bounds of each state
        Xlims = self.Xlims
        # specify upper and lower bound of contraction rate (and we will find optimal alpha within this region)
        alims = np.array([0.1,30])
        # name your NCM
        fname_ncm = self.fname_ncm
        dynamicsf = lambda X: self.A@X
        dynamicsg = self.GetB
        # In this case, we select non-default upper bounds for external disturbances (d1_over=0.0015 and d2_over=1.5).
        # Also, the step size for line search of alpha is also changed to non-default value (da = 0.01).
        ncm = NCM(dt,dynamicsf,dynamicsg,Xlims,alims,"con",fname_ncm)
        # You can use ncm.train(iTrain = 0) instead when using pre-traind NCM models.
        ncm.train(iTrain)
        return ncm
    
    def safeu(self,i,X,Xnow,Uds,dt):
        p = X[0:self.dim]
        v = X[self.q_states:self.q_states+self.dim]
        A = self.A
        B = self.GetB(X)
        umin = self.umin
        f = A@X
        Robs = self.Robs
        Rsafe = self.Rsafe
        Rsc = self.Rsc
        Us = cp.Variable((self.Nsc,self.m_inputs))
        U = Us[i,:]
        constraints = []
        constraints += [-U <= umin]
        pnext = p+v*dt
        pnnext = p+2*v*dt+(f[self.q_states:self.q_states+self.dim]+B[self.q_states:self.q_states+self.dim,:]@U)*dt**2
        for j in range(self.Nobs):
            xobsj = self.xobs[j,:]
            constraints += [(p-xobsj)@(pnnext-xobsj) >= (Robs+Rsafe)**2]
        for psc in range(self.Nsc):
            if psc != i:
                pj = Xnow[psc,:][0:self.dim]
                vj = Xnow[psc,:][self.q_states:self.q_states+self.dim]
                if np.linalg.norm(p-pj) <= self.Rsen:
                    Xj = Xnow[psc,:]
                    Uj = Us[psc,:]
                    constraints += [-Uj <= umin]
                    fj = A@Xj
                    Bj = self.GetB(Xj)
                    pjnext = pj+vj*dt
                    pjnnext = pj+2*vj*dt+(fj[self.q_states:self.q_states+self.dim]+Bj[self.q_states:self.q_states+self.dim,:]@Uj)*dt**2
                    constraints += [(p-pj)@(pnnext-pjnnext) >= (Rsc+Rsafe)**2]
                    #constraints += [(pnext-pjnext)@(pnnext-pjnnext)-Rsc*cp.norm(pnext-pjnext) >= 0]
        prob = cp.Problem(cp.Minimize(cp.norm(Us-Uds,"fro")**2),constraints)
        prob.solve(solver=cp.MOSEK)
        return U.value

    def safeu_global(self,Xs,Xfs,dt):
        A = self.A
        P = self.P
        Q = self.Q
        Us = cp.Variable((self.Nsc,self.m_inputs))
        Robs = self.Robs
        Rsc = self.Rsc
        Rsafe = self.Rsafe
        invR = np.linalg.inv(self.R)
        constraints = []
        for i in range(self.Nsc):
            Xi = Xs[i,:]
            Xfi = Xfs[i,:]
            Ui = Us[i,:]
            pi = Xi[0:self.dim]
            vi = Xi[self.q_states:self.q_states+self.dim]
            fi = A@Xi
            ei = Xi-Xfi
            pinext = pi+vi*dt
            Bi = self.GetB(Xi)
            pinnext = pi+2*vi*dt+(fi[self.q_states:self.q_states+self.dim]+Bi[self.q_states:self.q_states+self.dim,:]@Ui)*dt**2
            constraints += [ei@(P@A+A.T@P)@ei+2*ei@P@Bi@Ui <= -ei@(P@Bi@invR@Bi.T@P+Q)@ei]
            for o in range(self.Nobs):
                xobsj = self.xobs[o,:]
                #constraints += [(pinext-xobsj)@(pinnext-xobsj) >= (Robs+Rsafe)**2]
            for j in range(self.Nsc):
                if j <= i:
                    continue
                Xj = Xs[j,:]
                Uj = Us[j,:]
                pj = Xj[0:self.dim]
                vj = Xj[self.q_states:self.q_states+self.dim]
                fj = A@Xj
                Bj = self.GetB(Xj)
                pjnext = pj+vj*dt
                pjnnext = pj+2*vj*dt+(fj[self.q_states:self.q_states+self.dim]+Bj[self.q_states:self.q_states+self.dim,:]@Uj)*dt**2
                #constraints += [(pinext-pjnext)@(pinnext-pjnnext)-(Rsc+Rsafe)*cp.norm(pinext-pjnext) >= 0]
        prob = cp.Problem(cp.Minimize(cp.norm(Us,"fro")**2),constraints)
        prob.solve(solver=cp.MOSEK)
        return Us.value
    
    def dynamics(self,t,X,U):
        A = self.A
        B = self.GetB(X)
        dXdt = A@X+B@U
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
    
    def initial_trajectory(self,x0s=None,xfs=None,xobs=None):
        N = self.N
        dt = self.dt
        this = []
        Xshis = np.zeros((N+1,self.Nsc,self.n_states))
        Ushis = np.zeros((N,self.Nsc,self.m_inputs))
        try:
            X0s = np.hstack((x0s,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
            Xfs = np.hstack((xfs,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
            self.x0s = x0s
            self.xfs = xfs
            self.xobs = xobs
        except:
            self.generate_environment()
            X0s = np.hstack((self.x0s,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
            Xfs = np.hstack((self.xfs,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
        Xs = X0s
        t = 0
        this.append(t)
        Xshis[0,:,:] = Xs
        for i in range(N):
            tnow = t
            Xnow = np.zeros((self.Nsc,self.n_states))
            for p in range(self.Nsc):
                Xnow[p,:] = Xs[p,:]
            Us = np.zeros((self.Nsc,self.m_inputs))
            Us = self.safeu_global(Xs,Xfs,dt*5)
            for p in range(self.Nsc):
                X = Xs[p,:]
                U = Us[p,:]
                t,X = self.one_step_sim(t,X,U)
                t = tnow
                Xs[p,:] = X
                Us[p,:] = U
            t += dt
            this.append(t)
            Xshis[i+1,:,:] = Xs
            Ushis[i,:,:] = Us
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
        self.Xshis_ini = Xshis_l
        self.Ushis_ini = Ushis_l
        return this,Xshis_l,Ushis_l

    def scp(self):
        this,Xshis,Ushis = self.initial_trajectory()
        self.Xshis_ini = Xshis
        self.Ushis_ini = Ushis
        XXdFs,UUdFs,cvx_status = self.SCPnormalized(Xshis,Ushis,100,0)
        self.XXds = XXdFs
        self.UUds = UUdFs
        self.cvx_status = cvx_status
        
        plt.figure()
        thp = np.linspace(0,2*np.pi,100)
        for j in range(self.Nobs):
            plt.plot(self.xobs[j,:][0]+self.Robs*np.cos(thp),self.xobs[j,:][1]+self.Robs*np.sin(thp))
        for p in range(self.Nsc):
            plt.plot(self.XXds[p][:,0],self.XXds[p][:,1])
            plt.scatter(self.x0s[p,:][0],self.x0s[p,:][1],color='b')
            plt.scatter(self.xfs[p,:][0],self.xfs[p,:][1],color='r')
        plt.show()
        
        pass

    def online_mpc(self,p_idx,i_idx,X,Xnow,I):
        XXn = []
        UUn = []
        for p in range(self.Nsc):
            XXn.append(self.Xshis_ini[p][i_idx:self.N+1,:])
            UUn.append(self.Ushis_ini[p][i_idx:self.N,:])
        """
        try:
            XXn = self.XXsn[i]
            UUn = self.UUsn[i]
        except:
            XXn = self.Xshis_ini
            UUn = self.Ushis_ini
        """
        XXmpc,UUmpc,Xs_mpc,Xfs_mpc,xobs_mpc,Nsc_mpc,Nobs_mpc,idx_mpc = self.observation_mpc(X,Xnow,XXn,UUn)
        XX,UU,cvx_status = self.MPC_SCPj(XXmpc,UUmpc,10,0,I,Nsc_mpc,Nobs_mpc,Xs_mpc,Xfs_mpc,xobs_mpc,idx_mpc)
        #self.XXn[i] = XX
        #self.UUn[i] = UU
        self.XXsmpc[p_idx] = XX
        self.UUsmpc[p_idx] = UU
        """
        plt.figure()
        plt.plot(XX[:,0],XX[:,1])
        for j in range(self.Nobs):
            c_plot = plt.Circle((self.xobs[j,:][0],self.xobs[j,:][1]),self.Robs,fc="C7",ec="none",alpha=0.5)
            plt.gcf().gca().add_artist(c_plot)
        plt.scatter(self.x0s[p_idx,0],self.x0s[p_idx,1])
        plt.scatter(self.xfs[p_idx,0],self.xfs[p_idx,1])
        plt.xlim(-0.6,5.6)
        plt.ylim(-0.6,5.6)
        plt.grid()
        plt.show()
        """
        U = self.clfqp_a_mpc(X,XX[0,:],UU[0,:])
        return U
    
    def SCPnormalized(self,Xhiss,Uhiss,epochs,Tidx):
        dt = self.dt
        N = self.N
        X0s = np.hstack((self.x0s,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
        Xfs = np.hstack((self.xfs,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
        cvx_optval = np.inf
        p_prev = 1e16
        for i in range(epochs):
            XXs = []
            UUs = []
            constraints = []
            if p_prev <= 0.15:
                penalty = 0
            else:
                penalty = cp.Variable(1)
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
                    Us_k = Uhis[k,:]
                    constraints += [-Uk <= self.umin]
                    C = self.SafetyConstraint(Xk,Xs_k,self.Robs+self.Rsafe)
                    for j in range(self.Nobs):
                        constraints += [-C[j] <= penalty]
                    if Tidx == 1:
                        constraints += [cp.norm(Xk-Xs_k) <= Tregion]
                    constraints += [Xkp1 == self.DynamicsLD(dt,Xk,Uk,Xs_k,Us_k)]
                    for q in range(self.Nsc):
                        Xqk = XXs[q][k,:]
                        Xsq_k = Xhiss[q][k,:]
                        if q != p:
                            Cpq = self.SafetyConstraintSC(Xk,Xs_k,Xqk,Xsq_k,self.Rsc+self.Rsafe)
                            constraints += [-Cpq <= 0]
                J += cp.sum_squares(UU)
            prob = cp.Problem(cp.Minimize(J+100*penalty**2),constraints)
            try:
                prob.solve(solver=cp.MOSEK)
                #print(prob.status)
                #print("opt val",J.value)
                try:
                    #print("penalty",penalty.value)
                    p_prev = penalty.value[0]
                except:
                    #print("penalty",penalty)
                    p_prev = penalty
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
    
    def MPC_SCPj(self,Xhiss,Uhiss,epochs,Tidx,I,Nsc_mpc,Nobs_mpc,X0s,Xfs,xobs,mpc_idx):
        dt = self.dt
        N = I
        Nsc = Nsc_mpc
        Nobs = Nobs_mpc
        cvx_optval = np.inf
        p_prev = 1e16
        Xhis = Xhiss[mpc_idx]
        Uhis = Uhiss[mpc_idx]
        for i in range(epochs):
            XXs = []
            UUs = []
            constraints = []
            """
            if p_prev >= 0.2:
                penalty = cp.Variable(1)
            else:
                penalty = 0
            """
            penalty = cp.Variable(1)
            J = 0
            XX = cp.Variable((N+1,X0s[0].size))
            UU = cp.Variable((N,self.m_inputs))
            constraints += [XX[0,:] == X0s[mpc_idx]]
            constraints += [XX[N,:] == Xfs[mpc_idx]]
            Tregion = 0.1
            for k in range(N):
                Xkp1 = XX[k+1,:]
                Xk = XX[k,:]
                Uk = UU[k,:]
                Xs_k = Xhis[k,:]
                Us_k = Uhis[k,:]
                constraints += [-Uk <= self.umin]
                C = self.SafetyConstraint(Xk,Xs_k,self.Robs+self.Rsafe)
                for j in range(Nobs):
                    xobsj = xobs[j,:]
                    C = self.SafetyConstraintj(Xk,Xs_k,xobsj,self.Robs+self.Rsafe)
                    constraints += [-C <= penalty]
                if Tidx == 1:
                    constraints += [cp.norm(Xk-Xs_k) <= Tregion]
                constraints += [Xkp1 == self.DynamicsLD(dt,Xk,Uk,Xs_k,Us_k)]
                for q in range(Nsc):
                    Xsq_k = Xhiss[q][k,:]
                    if q != mpc_idx:
                        Cpq = self.SafetyConstraintSC(Xk,Xs_k,Xsq_k,Xsq_k,self.Rsc+self.Rsafe)
                        constraints += [-Cpq <= penalty]
            J += cp.sum_squares(UU)
            prob = cp.Problem(cp.Minimize(J+100*penalty**2),constraints)
            try:
                prob.solve(solver=cp.MOSEK)
                #print(prob.status)
                #print("opt val",J.value)
                try:
                    #print("penalty",penalty.value)
                    p_prev = penalty.value[0]
                except:
                    #print("penalty",penalty)
                    p_prev = penalty
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
    
    def MPC_SCP(self,Xhiss,Uhiss,epochs,Tidx,I,Nsc_mpc,Nobs_mpc,X0s,Xfs,xobs):
        dt = self.dt
        N = I
        Nsc = Nsc_mpc
        Nobs = Nobs_mpc
        cvx_optval = np.inf
        p_prev = 1e16
        for i in range(epochs):
            XXs = []
            UUs = []
            constraints = []
            """
            if p_prev >= 0.2:
                penalty = cp.Variable(1)
            else:
                penalty = 0
            """
            penalty = cp.Variable(1)
            J = 0
            for p in range(Nsc):
                XXs.append(cp.Variable((N+1,X0s[0].size)))
                UUs.append(cp.Variable((N,self.m_inputs)))
            for p in range(Nsc):
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
                    Us_k = Uhis[k,:]
                    constraints += [-Uk <= self.umin]
                    for j in range(Nobs):
                        xobsj = xobs[j,:]
                        C = self.SafetyConstraintj(Xk,Xs_k,xobsj,self.Robs+self.Rsafe)
                        constraints += [-C <= penalty]
                    if Tidx == 1:
                        constraints += [cp.norm(Xk-Xs_k) <= Tregion]
                    constraints += [Xkp1 == self.DynamicsLD(dt,Xk,Uk,Xs_k,Us_k)]
                    for q in range(Nsc):
                        Xqk = XXs[q][k,:]
                        Xsq_k = Xhiss[q][k,:]
                        Cpq = self.SafetyConstraintSC(Xk,Xs_k,Xqk,Xsq_k,self.Rsc+self.Rsafe)
                        constraints += [-Cpq <= penalty]
                J += cp.sum_squares(UU)
            prob = cp.Problem(cp.Minimize(J+100*penalty**2),constraints)
            try:
                prob.solve(solver=cp.MOSEK)
                print(prob.status)
                print("opt val",J.value)
                try:
                    print("penalty",penalty.value)
                    p_prev = penalty.value[0]
                except:
                    print("penalty",penalty)
                    p_prev = penalty
                Xhiss = []
                Uhiss = []
                for p in range(Nsc):
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
        for p in range(Nsc):
            XXs[p] = XXs[p].value
            UUs[p] = UUs[p].value
        return XXs,UUs,cvx_status
    
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
    
    def DynamicsLD(self,dt,Xk,Uk,Xs_k,Us_k):
        th_k = Xk[2]
        ths_k = Xs_k[2]
        A = self.A
        Bs = self.GetB(Xs_k)
        dBdths = self.GetdBdX(Xs_k)
        dXdt = A@Xk+Bs@Uk+(dBdths*(th_k-ths_k))@Us_k
        Xkp1 = Xk+dXdt*dt
        return Xkp1    
    
    def simulation(self):
        XXds = self.XXds
        UUds = self.UUds
        XXhiss = []
        for p in range(self.Nsc):
            XXd0 = XXds[p][0,:]
            t = 0
            N = np.size(UUds[p][:,0])
            Xhis = np.zeros((N+1,self.n_states))
            Xhis[0,:] = XXd0
            X = XXd0
            for k in range(N):
                U = UUds[p][k,:]
                t,X = self.one_step_sim(t,X,U)
                Xhis[k+1,:] = X
            XXhiss.append(Xhis)
        plt.figure()
        thp = np.linspace(0,2*np.pi,100)
        for j in range(self.Nobs):
            plt.plot(self.xobs[j,:][0]+self.Robs*np.cos(thp),self.xobs[j,:][1]+self.Robs*np.sin(thp))
        for p in range(self.Nsc):
            plt.plot(XXhiss[p][:,0],XXhiss[p][:,1])
            plt.scatter(self.x0s[p,:][0],self.x0s[p,:][1],color='b')
            plt.scatter(self.xfs[p,:][0],self.xfs[p,:][1],color='r')
        plt.show()
        return XXhiss,UUds
    
    def preprocess_data(self,file_path):
        xdata = np.load(file_path+"xdata.npy")
        udata = np.load(file_path+"udata.npy")
        obsdata = np.load(file_path+"obsdata.npy")
        self.Rsc = np.load(file_path+"Rscdata.npy")
        self.Rsafe = np.load(file_path+"Rscdata.npy")
        Ndata = np.shape(xdata)[0]
        Nsamp = self.N
        n = np.shape(xdata)[3]
        m = np.shape(udata)[3]
        self.n = n
        self.m = m
        xdata2 = np.zeros((Ndata,self.Nsc,Nsamp,n))
        xfdata = np.zeros((Ndata,self.Nsc,Nsamp,n))
        xjdata = np.zeros((Ndata,self.Nsc,Nsamp,n*(self.Nsc-1)))
        xodata = np.zeros((Ndata,self.Nsc,Nsamp,self.dim*self.Nobs))
        for k in range(Ndata):
            xok= obsdata[k,:,:]
            for i in range(self.Nsc):
                xf = xdata[k,i,-1,:]
                for s in range(Nsamp):
                    x = xdata[k,i,s,:]
                    xdata2[k,i,s,:] = x
                    xfdata[k,i,s,:] = xf-x
                    ij = 0
                    xjs = np.zeros((self.Nsc-1,self.n_states))
                    for j in range(self.Nsc):
                        if j != i:
                            xj = xdata[k,j,s,:]
                            xjs[ij,:] = xj-x
                            ij += 1
                    idxj = np.argsort(np.sum(xjs**2,1))
                    for j in range(self.Nsc-1):
                        xjdata[k,i,s,self.n_states*j:self.n_states*(j+1)] = xjs[idxj[j],:]
                    for o in range(self.Nobs):
                        xobs = xok[o,:]
                        doa = xobs-x[0:self.dim]
                        xodata[k,i,s,2*o:2*(o+1)] = doa
        xdata2 = xdata2.reshape(-1,self.n_states)
        udata = udata.reshape(-1,self.m_inputs)
        xfdata = xfdata.reshape(-1,self.n_states)
        xjdata = xjdata.reshape(-1,self.n_states*(self.Nsc-1))
        xodata = xodata.reshape(-1,self.dim*self.Nobs)
        N = Ndata*self.Nsc*Nsamp
        idx_d = np.random.choice(N,N,replace=False)
        xdata2 = xdata2[idx_d,:]
        udata = udata[idx_d,:]
        xfdata = xfdata[idx_d,:]
        xjdata = xjdata[idx_d,:]
        xodata = xodata[idx_d,:]
        return xdata2,udata,xfdata,xjdata,xodata
    
    def preprocess_data_robust(self,file_path,idx_load):
        if idx_load == 0:
            xdata = np.load(file_path+"xdata.npy")
            udata = np.load(file_path+"udata.npy")
            obsdata = np.load(file_path+"obsdata.npy")
            self.Rsc = np.load(file_path+"Rscdata.npy")
            self.Rsafe = np.load(file_path+"Rscdata.npy")
            Ndata = np.shape(xdata)[0]
            Nsamp = self.N
            Nrobust = 100
            n = np.shape(xdata)[3]
            m = np.shape(udata)[3]
            self.n = n
            self.m = m
            xdata_r = np.zeros((Ndata,self.Nsc,Nsamp,Nrobust,n))
            udata_r = np.zeros((Ndata,self.Nsc,Nsamp,Nrobust,m))
            xfdata = np.zeros((Ndata,self.Nsc,Nsamp,Nrobust,n))
            xjdata = np.zeros((Ndata,self.Nsc,Nsamp,Nrobust,n*(self.Nsc-1)))
            xodata = np.zeros((Ndata,self.Nsc,Nsamp,Nrobust,self.dim*self.Nobs))
            for k in range(Ndata):
                xok= obsdata[k,:,:]
                for i in range(self.Nsc):
                    print(i)
                    for s in range(Nsamp):
                        xd = xdata[k,i,s,:]
                        ud = udata[k,i,s,:]
                        xf = xdata[k,i,-1,:]
                        for r in range(Nrobust):
                            x = xd+unifrand2(d_over=0.5/2,nk=self.n_states)
                            u = self.clfqp_a(x,xd,ud)
                            xdata_r[k,i,s,r,:] = x
                            udata_r[k,i,s,r,:] = u
                            xfdata[k,i,s,r,:] = xf-x
                            ij = 0
                            xjs = np.zeros((self.Nsc-1,self.n_states))
                            for j in range(self.Nsc):
                                if j != i:
                                    xj = xdata[k,j,s,:]
                                    xjs[ij,:] = xj-x
                                    ij += 1
                            idxj = np.argsort(np.sum(xjs**2,1))
                            for j in range(self.Nsc-1):
                                xjdata[k,i,s,r,self.n_states*j:self.n_states*(j+1)] = xjs[idxj[j],:]
                            for o in range(self.Nobs):
                                xobs = xok[o,:]
                                doa = xobs-x[0:self.dim]
                                xodata[k,i,s,r,2*o:2*(o+1)] = doa
            xdata_r = xdata_r.reshape(-1,n)
            udata_r = udata_r.reshape(-1,m)
            xfdata = xfdata.reshape(-1,n)
            xjdata = xjdata.reshape(-1,n*(self.Nsc-1))
            xodata = xodata.reshape(-1,self.dim*self.Nobs)
            N = Ndata*self.Nsc*Nsamp*Nrobust
            idx_d = np.random.choice(N,N,replace=False)
            xdata_r = xdata_r[idx_d,:]
            udata_r = udata_r[idx_d,:]
            xfdata = xfdata[idx_d,:]
            xjdata = xjdata[idx_d,:]
            xodata = xodata[idx_d,:]
            np.save("data/sc_obs/xdata_pp.npy",xdata_r)
            np.save("data/sc_obs/udata_pp.npy",udata_r)
            np.save("data/sc_obs/xfdata_pp.npy",xfdata)
            np.save("data/sc_obs/xjdata_pp.npy",xjdata)
            np.save("data/sc_obs/xodata_pp.npy",xodata)
            self.xdata_r = xdata_r
            self.udata_r = udata_r
            self.xfdata = xfdata
            self.xjdata = xjdata
            self.xodata = xodata
        else:
            self.xdata_r = np.load("data/sc_obs/xdata_pp.npy")
            self.udata_r = np.load("data/sc_obs/udata_pp.npy")
            self.xfdata = np.load("data/sc_obs/xfdata_pp.npy")
            self.xjdata = np.load("data/sc_obs/xjdata_pp.npy")
            self.xodata = np.load("data/sc_obs/xodata_pp.npy")
            n = np.shape(self.xdata_r)[1]
            m = np.shape(self.udata_r)[1]
            self.n = n
            self.m = m
        pass
    
    def preprocess_data_robust_ncm(self,file_path,idx_load,iTrain):
        if idx_load == 0:
            xdata = np.load(file_path+"xdata.npy")
            udata = np.load(file_path+"udata.npy")
            obsdata = np.load(file_path+"obsdata.npy")
            self.Rsc = np.load(file_path+"Rscdata.npy")
            self.Rsafe = np.load(file_path+"Rscdata.npy")
            Ndata = np.shape(xdata)[0]
            Nsamp = self.N
            Nrobust = 100
            n = np.shape(xdata)[3]
            m = np.shape(udata)[3]
            self.n = n
            self.m = m
            xdata_r = np.zeros((Ndata,self.Nsc,Nsamp,Nrobust,n))
            udata_r = np.zeros((Ndata,self.Nsc,Nsamp,Nrobust,m))
            xfdata = np.zeros((Ndata,self.Nsc,Nsamp,Nrobust,n))
            xjdata = np.zeros((Ndata,self.Nsc,Nsamp,Nrobust,n*(self.Nsc-1)))
            xodata = np.zeros((Ndata,self.Nsc,Nsamp,Nrobust,self.dim*self.Nobs))
            ncm = self.ncm_train(iTrain)
            for k in range(Ndata):
                xok= obsdata[k,:,:]
                for i in range(self.Nsc):
                    print(i)
                    for s in range(Nsamp):
                        xd = xdata[k,i,s,:]
                        ud = udata[k,i,s,:]
                        xf = xdata[k,i,-1,:]
                        for r in range(Nrobust):
                            x = xd+unifrand2(d_over=ncm.Jcv_opt,nk=self.n_states)
                            u = self.clfqp_a_ncm(x,xd,ud,ncm)
                            xdata_r[k,i,s,r,:] = x
                            udata_r[k,i,s,r,:] = u
                            xfdata[k,i,s,r,:] = xf-x
                            ij = 0
                            xjs = np.zeros((self.Nsc-1,self.n_states))
                            for j in range(self.Nsc):
                                if j != i:
                                    xj = xdata[k,j,s,:]
                                    xjs[ij,:] = xj-x
                                    ij += 1
                            idxj = np.argsort(np.sum(xjs**2,1))
                            for j in range(self.Nsc-1):
                                xjdata[k,i,s,r,self.n_states*j:self.n_states*(j+1)] = xjs[idxj[j],:]
                            for o in range(self.Nobs):
                                xobs = xok[o,:]
                                doa = xobs-x[0:self.dim]
                                xodata[k,i,s,r,2*o:2*(o+1)] = doa
            xdata_r = xdata_r.reshape(-1,n)
            udata_r = udata_r.reshape(-1,m)
            xfdata = xfdata.reshape(-1,n)
            xjdata = xjdata.reshape(-1,n*(self.Nsc-1))
            xodata = xodata.reshape(-1,self.dim*self.Nobs)
            N = Ndata*self.Nsc*Nsamp*Nrobust
            idx_d = np.random.choice(N,N,replace=False)
            xdata_r = xdata_r[idx_d,:]
            udata_r = udata_r[idx_d,:]
            xfdata = xfdata[idx_d,:]
            xjdata = xjdata[idx_d,:]
            xodata = xodata[idx_d,:]
            np.save("data/sc_obs/xdata_pp.npy",xdata_r)
            np.save("data/sc_obs/udata_pp.npy",udata_r)
            np.save("data/sc_obs/xfdata_pp.npy",xfdata)
            np.save("data/sc_obs/xjdata_pp.npy",xjdata)
            np.save("data/sc_obs/xodata_pp.npy",xodata)
            self.xdata_r = xdata_r
            self.udata_r = udata_r
            self.xfdata = xfdata
            self.xjdata = xjdata
            self.xodata = xodata
        else:
            self.xdata_r = np.load("data/sc_obs/xdata_pp.npy")
            self.udata_r = np.load("data/sc_obs/udata_pp.npy")
            self.xfdata = np.load("data/sc_obs/xfdata_pp.npy")
            self.xjdata = np.load("data/sc_obs/xjdata_pp.npy")
            self.xodata = np.load("data/sc_obs/xodata_pp.npy")
            n = np.shape(self.xdata_r)[1]
            m = np.shape(self.udata_r)[1]
            self.n = n
            self.m = m
        pass
    
    def get_Xdata(self,xfdata,xjdata,xodata,Ndata):
        Xdata = []
        xjdata2 = np.zeros_like(xjdata)
        xjdata2 = xjdata2[0:Ndata,0:self.dim*(self.Nsc-1)]
        xodata2 = np.zeros_like(xodata)
        xodata2 = xodata2[0:Ndata,:]
        for k in range(Ndata):
            xjs = xjdata[k,:]
            xos = xodata[k,:]
            for p in range(self.Nsc-1):
                xj = xjs[self.n_states*p:self.n_states*(p+1)]
                if np.linalg.norm(xj[0:self.dim]) <= self.Rsen:
                    xjdata2[k,self.dim*p:self.dim*(p+1)] = xj[0:self.dim]
            for o in range(self.Nobs):
                xo = xos[self.dim*o:self.dim*(o+1)]
                if np.linalg.norm(xj)-self.Robs <= self.Rsen:
                    xodata2[k,self.dim*o:self.dim*(o+1)] = xo
        for p in range(self.Nsc-1):
            Xdata.append(xjdata2[0:Ndata,self.dim*p:self.dim*(p+1)])
        for o in range(self.Nobs):
            Xdata.append(xodata2[0:Ndata,self.dim*o:self.dim*(o+1)])
        Xdata.append(xfdata[0:Ndata,:])
        self.Xdata = Xdata
        pass

    def train_glas(self,file_path,Nbatch=32,Nphi=32,Nrho=32,Nlayers=3,Nunits=100,Nepochs=10000,
              ValidationSplit=0.1,Patience=20,Verbose=1):
        xdata2,udata,xfdata,xjdata,xodata = self.preprocess_data(file_path)
        Ndata = self.Nsc*15*self.N
        self.get_Xdata(xfdata,xjdata,xodata,Ndata)
        X = self.Xdata
        y = udata
        y = y[0:Ndata,:]
        print(np.max(y))
        n = self.n
        m =self.m
        inp = []
        phinet = Sequential(name="phi")
        phinet.add(Dense(Nunits,activation='relu',input_shape=(self.dim,)))
        phinet.add(Dense(Nphi,activation=None))
        rhonet = Sequential(name="rho")
        rhonet.add(Dense(Nunits,activation='relu',input_shape=(Nphi,)))
        rhonet.add(Dense(Nrho,activation=None))
        phi = []
        for j in range(self.Nsc-1):
            inp.append(Input(shape=(self.dim,)))
            phi.append(phinet(inp[j]))
            
        phinet2 = Sequential(name="phi2")
        phinet2.add(Dense(Nunits,activation='relu',input_shape=(self.dim,)))
        phinet2.add(Dense(Nphi,activation=None))
        rhonet2 = Sequential(name="rho2")
        rhonet2.add(Dense(Nunits,activation='relu',input_shape=(Nphi,)))
        rhonet2.add(Dense(Nrho,activation=None))
        phi2 = []
        for o in range(self.Nobs):
            inp.append(Input(shape=(self.dim,)))
            phi2.append(phinet2(inp[self.Nsc-1+o]))
        
        inputx0f = Input(shape=(n,))
        inp.append(inputx0f)
        
        outagts = rhonet(Add()(phi))
        outobs = rhonet2(Add()(phi2))
        
        psinet = Sequential(name="psi")
        psinet.add(Dense(Nunits,activation='relu',input_shape=(2*Nrho+n,)))
        for l in range(Nlayers):
            psinet.add(Dense(Nunits,activation="relu"))
            #psinet.add(Dropout(rate=0.2))
        psinet.add(Dense(m,activation="relu"))
        out = psinet(Concatenate(axis=1)([outagts,outobs,inputx0f]))
        
        model = Model(inputs=[inp],outputs=out)
        model.summary()
        model.compile(loss="mean_squared_error",optimizer="adam")
        es = EarlyStopping(monitor="val_loss",patience=Patience)
        model.fit(X,y,batch_size=Nbatch,epochs=Nepochs,verbose=1,\
                  callbacks=[es],validation_split=ValidationSplit)
        self.model = model
        model.save("data/sc_obs/neuralnets/model_glas.h5")
        pass
    
    def train_robust_glas(self,file_path,idx_load=1,Nbatch=32,Nphi=32,Nrho=32,Nlayers=3,Nunits=100,Nepochs=10000,
              ValidationSplit=0.1,Patience=20,Verbose=1,iTrain=0):
        self.preprocess_data_robust_ncm(file_path,idx_load,iTrain)
        xdata_r = self.xdata_r
        udata_r = self.udata_r
        xfdata = self.xfdata
        xjdata = self.xjdata
        xodata = self.xodata
        Ndata = self.Nsc*15*self.N
        self.get_Xdata(xfdata,xjdata,xodata,Ndata)
        X = self.Xdata
        y = udata_r
        y = y[0:Ndata,:]
        n = self.n
        m = self.m
        inp = []
        phinet = Sequential(name="phi")
        phinet.add(Dense(Nunits,activation='relu',input_shape=(self.dim,)))
        phinet.add(Dense(Nphi,activation=None))
        rhonet = Sequential(name="rho")
        rhonet.add(Dense(Nunits,activation='relu',input_shape=(Nphi,)))
        rhonet.add(Dense(Nrho,activation=None))
        phi = []
        for j in range(self.Nsc-1):
            inp.append(Input(shape=(self.dim,)))
            phi.append(phinet(inp[j]))
            
        phinet2 = Sequential(name="phi2")
        phinet2.add(Dense(Nunits,activation='relu',input_shape=(self.dim,)))
        phinet2.add(Dense(Nphi,activation=None))
        rhonet2 = Sequential(name="rho2")
        rhonet2.add(Dense(Nunits,activation='relu',input_shape=(Nphi,)))
        rhonet2.add(Dense(Nrho,activation=None))
        phi2 = []
        for o in range(self.Nobs):
            inp.append(Input(shape=(self.dim,)))
            phi2.append(phinet2(inp[self.Nsc-1+o]))
        
        inputx0f = Input(shape=(n,))
        inp.append(inputx0f)
        
        outagts = rhonet(Add()(phi))
        outobs = rhonet2(Add()(phi2))
        
        psinet = Sequential(name="psi")
        psinet.add(Dense(Nunits,activation='relu',input_shape=(2*Nrho+n,)))
        for l in range(Nlayers):
            psinet.add(Dense(Nunits,activation="relu"))
            #psinet.add(Dropout(rate=0.2))
        psinet.add(Dense(m,activation="relu"))
        out = psinet(Concatenate(axis=1)([outagts,outobs,inputx0f]))
        
        model = Model(inputs=[inp],outputs=out)
        model.summary()
        model.compile(loss="mean_squared_error",optimizer="adam")
        es = EarlyStopping(monitor="val_loss",patience=Patience)
        model.fit(X,y,batch_size=Nbatch,epochs=Nepochs,verbose=1,\
                  callbacks=[es],validation_split=ValidationSplit)
        self.model = model
        model.save("data/sc_obs/neuralnets/model_robust_glas.h5")
        pass

    def clfqp_a(self,X,Xd,Ud):
        """
        e = X-Xd
        A = self.A
        B =self.GetB(X)
        Bd =self.GetB(Xd)
        f = A@X+B@Ud
        fd = A@Xd+Bd@Ud
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
        """
        umin = self.umin
        e = X-Xd
        A = self.A
        B = self.GetB(X)
        Bd = self.GetB(Xd)
        f = A@X
        fd = A@Xd
        Q = np.identity(self.n_states)*10
        R = np.identity(self.m_inputs)
        K,P,_ = control.lqr(A,B,Q,R)
        U = cp.Variable(self.m_inputs)
        invR = np.linalg.inv(R)
        dVdt = 2*e@P@(f+B@U-fd-Bd@Ud)
        constraints = [dVdt <= -e@(P@B@invR@B.T@P+Q)@e]
        constraints += [-U <= umin]
        prob = cp.Problem(cp.Minimize(cp.norm(U-Ud)**2),constraints)
        prob.solve(solver=cp.MOSEK)
        return U.value
 
    def clfqp_a_ncm(self,X,Xd,Ud,ncm):
        umin = self.umin
        e = X-Xd
        A = self.A
        B = self.GetB(X)
        Bd = self.GetB(Xd)
        f = A@X
        fd = A@Xd
        I = np.identity(self.n_states)
        alp = ncm.alp_opt
        M = ncm.ncm(X,np.empty(0))
        U = cp.Variable(self.m_inputs)
        dVdt = 2*e@M@(f+B@U-fd-Bd@Ud)
        dMdt = (ncm.nu_opt*I-M)/ncm.dt
        constraints = [dVdt <= -2*alp*e@M@e]
        constraints += [-U <= umin]
        prob = cp.Problem(cp.Minimize(cp.norm(U-Ud)**2),constraints)
        prob.solve(solver=cp.MOSEK)
        return U.value

    def clfqp_a_mpc(self,X,Xd,Ud):
        """
        e = X-Xd
        A = self.A
        B =self.GetB(X)
        Bd =self.GetB(Xd)
        f = A@X+B@Ud
        fd = A@Xd+Bd@Ud
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
        """
        umin = self.umin
        e = X-Xd
        A = self.A
        B = self.GetB(X)
        Bd = self.GetB(Xd)
        f = A@X
        fd = A@Xd
        Q = np.identity(self.n_states)*5
        R = np.identity(self.m_inputs)
        K,P,_ = control.lqr(A,B,Q,R)
        U = cp.Variable(self.m_inputs)
        invR = np.linalg.inv(R)
        dVdt = 2*e@P@(f+B@U-fd-Bd@Ud)
        constraints = [dVdt <= -e@(P@B@invR@B.T@P+Q)@e]
        constraints += [-U <= umin]
        prob = cp.Problem(cp.Minimize(cp.norm(U-Ud)**2),constraints)
        prob.solve(solver=cp.MOSEK)
        return U.value
    
    def observation_glas(self,i,X,Xf,Xnow):
        p = X[0:self.dim]
        xobs = self.xobs
        Robs = self.Robs
        Xfi = Xf-X
        ij = 0
        Xjs = np.zeros((self.Nsc-1,self.n_states))
        for j in range(self.Nsc):
            if j != i:
                Xj = Xnow[j,:]
                Xjs[ij,:] = Xj-X
                ij += 1
        idxj = np.argsort(np.sum(Xjs**2,1))
        Xji = np.zeros(self.dim*(self.Nsc-1))
        for j in range(self.Nsc-1):
            if np.linalg.norm(Xjs[idxj[j],:][0:self.dim]) <= self.Rsen:
                Xji[self.dim*j:self.dim*(j+1)] = Xjs[idxj[j],:][0:self.dim]
        xo = [np.zeros((1,self.dim))]*self.Nobs
        for o in range(self.Nobs):
            xobsi = xobs[o,:]
            xoi = xobsi-p
            if np.linalg.norm(xoi)-(Robs) <= self.Rsen:
                xo[o] = np.array([xoi])
        Xout = []
        for j in range(self.Nsc-1):
            Xout.append(np.array([Xji[self.dim*j:self.dim*(j+1)]]))
        for o in range(self.Nobs):
            Xout.append(xo[o])
        Xout.append(np.array([Xfi]))
        return Xout
    
    def observation_mpc(self,X,Xnow,XXn,UUn):
        p = X[0:self.dim]
        xobs = self.xobs
        Robs = self.Robs
        Xs_mpc = []
        Xfs_mpc = []
        XXmpc = []
        UUmpc = []
        xobs_mpc = []
        X0s = np.hstack((self.x0s,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
        Xfs = np.hstack((self.xfs,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
        ij = 0
        for j in range(self.Nsc):
            pj = Xnow[j,0:self.dim]
            if np.linalg.norm(p-pj) <= self.Rsen:
                if np.linalg.norm(p-pj) == 0:
                    idx_mpc = ij
                    #print(ij)
                ij += 1
                Xs_mpc.append(Xnow[j,:])
                Xfs_mpc.append(Xfs[j,:])
                XXmpc.append(XXn[j])
                UUmpc.append(UUn[j])
        Nsc_mpc = len(Xs_mpc)
        Xs_mpc = np.array(Xs_mpc)
        Xfs_mpc = np.array(Xfs_mpc)
        for o in range(self.Nobs):
            xobsi = xobs[o,:]
            if np.linalg.norm(p-xobsi)-(Robs) <= self.Rsen:
                xobs_mpc.append(xobs[o,:])
        Nobs_mpc = len(xobs_mpc)
        xobs_mpc = np.array(xobs_mpc)
        return XXmpc,UUmpc,Xs_mpc,Xfs_mpc,xobs_mpc,Nsc_mpc,Nobs_mpc,idx_mpc
    
    def nmpc_simulation(self,dover,XXds=None,UUds=None,dtc0_scp=None,x0s=None,xfs=None,xobs=None):
        #self.XXsmpc = np.zeros((self.Nsc,self.N+1,self.n_states))
        #self.UUsmpc = np.zeros((self.Nsc,self.N,self.m_inputs))
        self.XXsmpc = [0]*self.Nsc
        self.UUsmpc = [0]*self.Nsc
        model = load_model("data/sc_obs/neuralnets/model_glas.h5")
        model_r = load_model("data/sc_obs/neuralnets/model_robust_glas.h5")
        #self.generate_environment()
        ncm = self.ncm_train(iTrain=0)
        if dtc0_scp == None:
            tc0 = time.time()
            self.scp()
            dtc0_scp = time.time()-tc0
            XXds = self.XXds
            UUds = self.UUds
        else:
            dtc0_scp = 100
            self.x0s = x0s
            self.xfs = xfs
            self.xobs = xobs
        self.initial_trajectory(self.x0s,self.xfs,self.xobs)
    
        N = int(self.N*3)
        dt = self.dt
        this = []
        Ncon = 4
        Xshis = np.zeros((N+1,Ncon,self.Nsc,self.n_states))
        Ushis = np.zeros((N,Ncon,self.Nsc,self.m_inputs))
        dthis = np.zeros((N,Ncon,self.Nsc))
        X0s = np.hstack((self.x0s,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
        Xfs = np.hstack((self.xfs,np.zeros((self.Nsc,self.q_states+self.n_add_states))))
        t = 0
        this.append(t)
        XXs = np.zeros((Ncon,self.Nsc,self.n_states))
        eps = 0.2
        for c in range(Ncon):
            XXs[c,:,:] = X0s
            Xshis[0,c,:,:] = X0s
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
                Uds = np.zeros((self.Nsc,self.m_inputs))
                for p in range(self.Nsc):
                    if idx_fin[c,p] == 0:
                        X = Xs[p,:]
                        Xf = Xfs[p,:]
                        XX = self.observation_glas(p,X,Xf,Xnow)
                        if c == 0:
                            tc0 = time.time()
                            U = model.predict(XX)
                            dtc0 = time.time()-tc0
                        elif c == 1:
                            tc0 = time.time()
                            U = model_r.predict(XX)
                            dtc0 = time.time()-tc0
                            #print("control time",dc0)
                        elif c == 2:
                            if (i%45 == 0) & (i < self.N):
                                I = self.N-i
                                tc0 = time.time()
                                U = self.online_mpc(p,i,X,Xnow,I)
                                dtc0 = time.time()-tc0
                                Ith = i
                            else:
                                if (i < self.N):
                                    impc = i-Ith
                                    tc0 = time.time()
                                    U = self.clfqp_a_mpc(X,self.XXsmpc[p][impc,:],self.UUsmpc[p][impc,:])
                                    dtc0 = time.time()-tc0
                                else:
                                    tc0 = time.time()
                                    U = self.clfqp_a_mpc(X,self.XXsmpc[p][-1,:],self.UUsmpc[p][-1,:])
                                    dtc0 = time.time()-tc0
                        elif c == 3:
                            tc0 = time.time()
                            if (i < self.N):
                                U = self.clfqp_a_mpc(X,XXds[p][i,:],UUds[p][i,:])
                            else:
                                U = self.clfqp_a_mpc(X,XXds[p][-1,:],UUds[p][-1,:])
                            dtc0 = time.time()-tc0+dtc0_scp
                        U = U.ravel()
                        Uds[p,:] = U
                        dthis[i,c,p] = dtc0
                for p in range(self.Nsc):
                    if idx_fin[c,p] == 0:
                        X = Xs[p,:]
                        ts0 = time.time()
                        U = self.safeu(p,X,Xnow,Uds,dt*10)
                        #U = Uds[p,:]
                        dts0 = time.time()-ts0
                        dthis[i,c,p] += dts0
                        #print("safe time",dc0)
                        #U = self.safeuMPC(p,X,Xnow,Xfs,Uds,model_r,dt*2)
                        #U = Uds[p,:]
                        #print(U)
                        t,X = self.one_step_sim(t,X,U)
                        d = ds[p,:]
                        X += d*dt
                        t = tnow
                        XXs[c,p,:] = X
                        UUs[c,p,:] = U
                        Xf = Xfs[p,:]
                        if np.linalg.norm(X[0:2]-Xf[0:2]) <= eps:
                            idx_fin[c,p] = 1
                            Nends[c,p] = i
                            #print(idx_fin)
                            #print(Nends)
            t += dt
            this.append(t)
            Xshis[i+1,:,:,:] = XXs
            Ushis[i,:,:,:] = UUs
            if np.sum(idx_fin) == Ncon*self.Nsc:
                print("FIN")
                break
        this = np.array(this)
        ueffs = []
        for c in range(Ncon):
            Xhisc = Xshis[:,c,:,:]
            ueffc = 0
            plt.figure()
            thp = np.linspace(0,2*np.pi,100)
            for j in range(self.Nobs):
                c_plot = plt.Circle((self.xobs[j,:][0],self.xobs[j,:][1]),self.Robs,fc="C7",ec="none",alpha=0.5)
                plt.gcf().gca().add_artist(c_plot)
            for p in range(self.Nsc):
                colorp = "C"+str(p)
                Nend = int(Nends[c,p])
                #ueffc += np.sum(np.sqrt(np.sum(Ushis[0:Nend,c,p,:]**2,1)))
                ueffc += np.sum(Ushis[0:Nend,c,p,:]**2)
                #plt.plot(Xshis[:,p,0],Xshis[:,p,1])
                plt.plot(Xhisc[0:Nend+1,p,0],Xhisc[0:Nend+1,p,1],color=colorp)
                plt.scatter(self.x0s[p,:][0],self.x0s[p,:][1],facecolors="none",edgecolors=colorp)
                plt.scatter(self.xfs[p,:][0],self.xfs[p,:][1],color=colorp,marker="x")
            ueffs.append(ueffc)
            plt.xlim(-0.6,5.6)
            plt.ylim(-0.6,5.6)
            plt.grid()
            plt.show()
        for c in range(Ncon):
            Xhisc = Xshis[:,c,:,:]
            plt.figure()
            thp = np.linspace(0,2*np.pi,100)
            for j in range(self.Nobs):
                c_plot = plt.Circle((self.xobs[j,:][0],self.xobs[j,:][1]),self.Robs,fc="C7",ec="none",alpha=0.5)
                plt.gcf().gca().add_artist(c_plot)
            for p in range(self.Nsc):
                colorp = "C"+str(p)
                Nend = int(Nends[c,p])
                plt.plot(Xhisc[0:Nend+1,p,0],Xhisc[0:Nend+1,p,1],color=colorp)
                plt.scatter(self.x0s[p,:][0],self.x0s[p,:][1],facecolors="none",edgecolors=colorp)
                plt.scatter(Xhisc[Nend,p,0],Xhisc[Nend,p,1],color=colorp)
            plt.xlim(-0.6,5.6)
            plt.ylim(-0.6,5.6)
            plt.grid()
            plt.show()
        print(ueffs)
        print(idx_fin)
        """
        figs = []
        ims = [[0]*(N+1) for i in range(Ncon)]
        thp = np.linspace(0,2*np.pi,100)
        for c in range(Ncon):
            fig = plt.figure()
            for k in range(N+1):
                im = []
                for p in range(self.Nsc):                   
                    Nend = int(Nends[c,p])
                    colorp = "C"+str(p)
                    if k <= Nend+1:
                        im.append(plt.scatter(Xshis[k,c,p,0],Xshis[k,c,p,1],color=colorp))
                        im += plt.plot(Xshis[0:k,c,p,0],Xshis[0:k,c,p,1],color=colorp) # plot returns list
                    else:
                        im.append(plt.scatter(Xshis[Nend+1,c,p,0],Xshis[Nend+1,c,p,1],color=colorp))
                        im += plt.plot(Xshis[0:Nend+1,c,p,0],Xshis[0:Nend+1,c,p,1],color=colorp) # plot returns list
                    im.append(plt.scatter(self.x0s[p,0],self.x0s[p,1],facecolors="none",edgecolors=colorp))
                    im.append(plt.scatter(self.xfs[p,0],self.xfs[p,1],color=colorp,marker="x"))
                for o in range(self.Nobs):
                    c_plot = plt.Circle((self.xobs[o,:][0],self.xobs[o,:][1]),self.Robs,fc="C7",ec="none",alpha=0.5)
                    print()
                    im.append(plt.gcf().gca().add_artist(c_plot))
                ims[c][k] = im
            plt.title("sup||d(t))|| = "+str(dover))
            plt.grid()
            plt.xlim(-0.6,5.6)
            plt.ylim(-0.6,5.6)
            figs.append(fig)
        for c in range(Ncon):
            ani = animation.ArtistAnimation(figs[c],ims[c],interval=dt*1000)
            ani.save("movies/sc_obs/output"+str(c)+"_"+str(dover*10)+".mp4",writer="ffmpeg")
        """    
        return this,dthis,Xshis,Ushis,ueffs,idx_fin,Nends
    
def unifrand2(d_over=1,nk=4):
    d_over_out = d_over+1
    while d_over_out > d_over:
            d = np.random.uniform(-d_over,d_over,size=nk)
            d_over_out = np.linalg.norm(d)
    return d
    
if __name__ == "__main__":
    Nobs = 6
    Nsc = 6
    Nsim = 15
    
    """
    xdata = []
    udata = []
    obsdata = []
    fails = []
    for i in range(Nsim):
        start_time = time.time()
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
        obsdata.append(np.array(nmpc.xobs))
        np.save("data/sc_obs/xdata.npy",np.array(xdata))
        np.save("data/sc_obs/udata.npy",np.array(udata))
        np.save("data/sc_obs/obsdata.npy",np.array(obsdata))
        np.save("data/sc_obs/Rscdata.npy",nmpc.Rsc)
        np.save("data/sc_obs/Rsafedata.npy",nmpc.Rsafe)
        t_one_epoch = time.time()-start_time
        print("time:",t_one_epoch,"s")
        print("i =",i)
    xdata = np.array(xdata)
    udata = np.array(udata)
    obsdata = np.array(obsdata)
    np.save("data/sc_obs/xdata.npy",np.array(xdata))
    np.save("data/sc_obs/udata.npy",np.array(udata))
    np.save("data/sc_obs/obsdata.npy",np.array(obsdata))
    np.save("data/sc_obs/Rscdata.npy",nmpc.Rsc)
    np.save("data/sc_obs/Rsafedata.npy",nmpc.Rsafe)
    """
    
    nmpc = NMPC(Nobs,Nsc)
    
    file_path = "data/sc_obs/"
    #nmpc.train_robust_glas(file_path,idx_load=1,iTrain=0)
    #nmpc.train_glas(file_path)
    """
    dovers = [0,0.2,0.4,0.6,0.8,1.0]
    dovers = [0.8]
    for dover in dovers:        
        this,dthis,Xshis,Ushis,ueffs,idx_fin,Nends = nmpc.nmpc_simulation(dover)
        np.save("data/sc_obs/sim/Xshis_"+str(dover*10)+".npy",Xshis)
        np.save("data/sc_obs/sim/Ushis_"+str(dover*10)+".npy",Ushis)
        np.save("data/sc_obs/sim/ueffs_"+str(dover*10)+".npy",ueffs)
        np.save("data/sc_obs/sim/idx_fin_"+str(dover*10)+".npy",idx_fin)
        np.save("data/sc_obs/sim/Nends_"+str(dover*10)+".npy",Nends)
        np.save("data/sc_obs/sim/Robs_"+str(dover*10)+".npy",nmpc.Robs)
        np.save("data/sc_obs/sim/xobs_"+str(dover*10)+".npy",nmpc.xobs)
        np.save("data/sc_obs/sim/x0s_"+str(dover*10)+".npy",nmpc.x0s)
        np.save("data/sc_obs/sim/xfs_"+str(dover*10)+".npy",nmpc.xfs)
    """
    
    """
    nmpc = NMPC(Nobs,Nsc)
    tc0 = time.time()
    nmpc.scp()
    dtc0_scp = time.time()-tc0
    np.save("data/sc_obs/sim/scp/XXds.npy",nmpc.XXds)
    np.save("data/sc_obs/sim/scp/UUds.npy",nmpc.UUds)
    np.save("data/sc_obs/sim/scp/xobs.npy",nmpc.xobs)
    np.save("data/sc_obs/sim/scp/x0s.npy",nmpc.x0s)
    np.save("data/sc_obs/sim/scp/xfs.npy",nmpc.xfs)
    """
    
    """
    XXds = np.load("data/sc_obs/sim/scp/XXds.npy")
    UUds = np.load("data/sc_obs/sim/scp/UUds.npy")
    xobs = np.load("data/sc_obs/sim/scp/xobs.npy")
    x0s = np.load("data/sc_obs/sim/scp/x0s.npy")
    xfs = np.load("data/sc_obs/sim/scp/xfs.npy")
    dtc0_scp = 100
    
    this,dthis,Xshis,Ushis,ueffs,idx_fin,Nends = nmpc.nmpc_simulation(dover,XXds,UUds,dtc0_scp,x0s,xfs,xobs)
    """
    
    Ndata = 5
    dover = 0.8
    dovers = [0.2,0.4,0.6,0.8,1.0]
    np.random.seed(seed=2) # for k = 6-9
    #for k in range(10):
    for k in range(6,10):
        for dover in dovers:
            thiss = []
            dthiss = []
            Xshiss = []
            Ushiss = []
            ueffss = []
            idx_fins = []
            Nendss = []
            Robss = []
            xobss = []
            x0ss = []
            xfss = []
            fails = []
            i = 0
            while len(thiss) < Ndata:
                print("k =",k)
                print("dover =",dover)
                print("Ndata =",i)
                try:
                    nmpc = NMPC(Nobs,Nsc)
                    this,dthis,Xshis,Ushis,ueffs,idx_fin,Nends = nmpc.nmpc_simulation(dover)
                except:
                    continue
                thiss.append(this)
                dthiss.append(dthis)
                Xshiss.append(Xshis)
                Ushiss.append(Ushis)
                ueffss.append(ueffs)
                idx_fins.append(idx_fin)
                Nendss.append(Nends)
                Robss.append(nmpc.Robs)
                xobss.append(nmpc.xobs)
                x0ss.append(nmpc.x0s)
                xfss.append(nmpc.xfs)
                np.save("data/sc_obs/sim/thiss_"+str(k)+str(dover*10)+".npy",np.array(thiss))
                np.save("data/sc_obs/sim/dthiss_"+str(k)+str(dover*10)+".npy",np.array(dthiss))
                np.save("data/sc_obs/sim/Xshiss_"+str(k)+str(dover*10)+".npy",np.array(Xshiss))
                np.save("data/sc_obs/sim/Ushiss_"+str(k)+str(dover*10)+".npy",np.array(Ushiss))
                np.save("data/sc_obs/sim/ueffss_"+str(k)+str(dover*10)+".npy",np.array(ueffss))
                np.save("data/sc_obs/sim/idx_fins_"+str(k)+str(dover*10)+".npy",np.array(idx_fins))
                np.save("data/sc_obs/sim/Nendss_"+str(k)+str(dover*10)+".npy",np.array(Nendss))
                np.save("data/sc_obs/sim/Robss_"+str(k)+str(dover*10)+".npy",np.array(Robss))
                np.save("data/sc_obs/sim/xobss_"+str(k)+str(dover*10)+".npy",np.array(xobss))
                np.save("data/sc_obs/sim/x0ss_"+str(k)+str(dover*10)+".npy",np.array(x0ss))
                np.save("data/sc_obs/sim/xfss_"+str(k)+str(dover*10)+".npy",np.array(xfss))
                i += 1
    
    
        