#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 21:27:54 2020

@author: hiroyasu
"""

import numpy as np
import cvxpy as cp
import scipy as sp
from matplotlib import pyplot as plt
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
import matplotlib.patches as patches

from classncm import NCM

np.random.seed(seed=2268)

class NMPC():
    def __init__(self):
        self.g = 9.8
        self.m_c = 1.0
        self.m = 0.1
        self.mu_c = 0.5
        self.mu_p = 0.002
        self.l = 0.5
        self.n_states = 4
        self.n_add_states = 1
        self.m_inputs = 1
        self.q_states = self.n_states//2
        self.dim = 1
        self.Xlims = np.array([-np.ones(self.n_states),np.ones(self.n_states)])*np.pi/3
        Z = np.zeros((self.q_states,self.q_states))
        eyeq = np.identity(self.q_states)
        self.A = np.vstack((np.hstack((Z,eyeq)),np.hstack((Z,Z))))
        Q = np.identity(self.n_states)*1
        R = np.identity(self.m_inputs)
        self.Q = Q
        self.R = R
        self.dt = 0.1
        self.Nrk = 1
        self.h = self.dt/self.Nrk
        self.N = 100
        #self.th_robust = np.pi/2
        self.th_robust = 1000000
        self.umin = -5
        self.umax = 5
        self.C_u = 20
        self.fname_ncm = "NCM_cartpole"
    
    def dynamicsf(self,X):
        th = X[1]
        v = X[2]
        om = X[3]
        g = self.g
        m_c = self.m_c
        m = self.m
        mu_c = self.mu_c
        mu_p = self.mu_p
        l = self.l
        f1 = v
        f2 = om
        f4 = (g*np.sin(th)+np.cos(th)*(-m*l*om**2*np.sin(th)+mu_c*v)-mu_p*om/(m*l))/(l*(4/3-m*np.cos(th)**2/(m_c+m)))
        f3 = (m*l*om**2*np.sin(th)-mu_c*v)/(m_c+m)-m*l*f4*np.cos(th)/(m_c+m)
        f = np.array([f1,f2,f3,f4])
        return f
    
    def dynamicsg(self,X):
        th = X[1]
        m_c = self.m_c
        m = self.m
        l = self.l
        g1 = 0
        g2 = 0
        g4 = -np.cos(th)/(l*(4/3-m*np.cos(th)**2/(m_c+m)))
        g3 = 1/(m_c+m)-m*l*g4*np.cos(th)/(m_c+m)
        gfun = np.array([[g1],[g2],[g3],[g4]])
        return gfun
    
    def getdfdX(self,X):
        dx = 0.001
        A = np.zeros((self.n_states,self.n_states))
        for i in range(self.n_states):
            ei = np.zeros(self.n_states)
            ei[i] = 1
            f1 = self.dynamicsf(X+ei*dx)
            f0 = self.dynamicsf(X-ei*dx)
            dfdx = (f1-f0)/2/dx
            A[:,i] = dfdx
        return A

    def getdgdX(self,X):
        dx = 0.001
        ei = np.zeros(self.n_states)
        ei[1] = 1 #theta direction
        g1 = self.dynamicsg(X+ei*dx)
        g0 = self.dynamicsg(X-ei*dx)
        dgdx = (g1-g0)/2/dx
        return dgdx
    
    def sdcA(self,X):
        x = X[0]
        th = X[1]
        v = X[2]
        om = X[3]
        g = self.g
        m_c = self.m_c
        m = self.m
        mu_c = self.mu_c
        mu_p = self.mu_p
        l = self.l
        Mth = l*(4/3-m*np.cos(th)**2/(m_c+m))
        # sinc(x) = sin(pi*x)/(pi*x)
        Ath = np.array([0,g*np.sinc(th/np.pi)/Mth,mu_c*np.cos(th)/Mth,(-np.cos(th)*m*l*om*np.sin(th)-mu_p/m/l)/Mth])
        M = m_c+m
        Ax = np.array([0,0,-mu_c/M,m*l*om*np.sin(th)/M])
        Ax = Ax-m*l*np.cos(th)/M*Ath
        A12 = np.hstack((np.zeros((2,2)),np.identity(2)))
        A = np.vstack((A12,Ax))
        A = np.vstack((A,Ath))
        return A
    
    def ncm_train(self,iTrain):
        # enter your choice of CV-STEM sampling period
        dt = 1
        # specify upper and lower bounds of each state
        Xlims = self.Xlims
        # specify upper and lower bound of contraction rate (and we will find optimal alpha within this region)
        alims = np.array([0.1,30])
        # name your NCM
        fname_ncm = self.fname_ncm
        dynamicsf = self.dynamicsf
        dynamicsg = self.dynamicsg
        ncm = NCM(dt,dynamicsf,dynamicsg,Xlims,alims,"con",fname_ncm)
        ncm.Afun = self.sdcA
        # You can use ncm.train(iTrain = 0) instead when using pre-traind NCM models.
        ncm.train(iTrain)
        return ncm
    
    def normalize_rad(self,th):
        if th > np.pi:
            n_rad = th/(2*np.pi)
            th -= n_rad*2*np.pi
        elif th < -np.pi:
            n_rad = -th/(2*np.pi)
            th += n_rad*2*np.pi
        return th

    def safeu_global(self,X,Xd,Ud):
        X[1] = nmpc.normalize_rad(X[1])
        Xd[1] = nmpc.normalize_rad(Xd[1])
        f = self.dynamicsf(X)
        g = self.dynamicsg(X)
        fd = self.dynamicsf(Xd)
        gd = self.dynamicsg(Xd)
        U = cp.Variable(self.m_inputs)
        Q = self.Q
        K,P,_ = control.lqr(self.sdcA(X),self.dynamicsg(X),Q,self.R)
        invR = np.linalg.inv(self.R)
        e = X-Xd
        dVdt = 2*e@P@(f+g@U-fd-gd@Ud)
        constraints = [dVdt <= -e@(P@g@invR@g.T@P+Q)@e]
        #constraints = [dVdt <= -e@(Q)@e]
        prob = cp.Problem(cp.Minimize(cp.norm(U-Ud)**2),constraints)
        prob.solve(solver=cp.MOSEK)
        return U.value
    
    def dynamics(self,t,X,U):
        f = self.dynamicsf(X)
        g = self.dynamicsg(X)
        dXdt = f+g@U
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
    
    def initial_trajectory(self):
        while True:
            #th0 = np.random.uniform(low=-np.pi,high=-2*np.pi/3)*np.random.choice([-1,1])
            th0 = np.random.uniform(low=-np.pi/3,high=np.pi/3)
            #th0 = np.random.uniform(low=-np.pi,high=-np.pi/2)
            X0 = np.zeros(self.n_states)
            X0[1] = th0
            self.X0 = X0
            Xf = np.zeros(self.n_states)
            xf = np.random.uniform(low=-5,high=5)
            Xf[0] = xf
            self.Xf = Xf
            N = self.N
            this = []
            Xhis = np.zeros((N+1,self.n_states))
            Uhis = np.zeros((N,self.m_inputs))
            X = X0
            t = 0
            this.append(t)
            Xhis[0,:] = X
            for i in range(N):
                U = self.clfqp_a(X,self.Xf,np.zeros(1))
                t,X = self.one_step_sim(t,X,U)
                this.append(t)
                Xhis[i+1,:] = X
                Uhis[i,:] = U
            this = np.array(this)
            if np.linalg.norm(Xhis[-1,:]-self.Xf) <= 0.1:
                break
        plt.figure()
        for i in range(self.n_states):
            plt.plot(this,Xhis[:,i])
        return this,Xhis,Uhis

    def scp(self):
        this,Xhis,Uhis = self.initial_trajectory()
        print("ueff clf",np.sum(Uhis**2))
        self.online_mpc(self.X0,self.N)
        """
        #XX,UU,cvx_status = self.SCPnormalized(Xhis,Uhis,100,0)
        uu = Uhis.reshape(self.m_inputs*self.N)
        xx = Xhis.reshape(self.n_states*(self.N+1))
        x0 = np.hstack((uu,xx))
        XX,UU = self.fmincon(x0,self.N)
        """
        self.XXds = self.XXmpc
        self.UUds = self.UUmpc
        pass

    def online_mpc(self,X,I):
        try:
            XXmpc = self.XXmpc
            UUmpc = self.UUmpc
        except:
            XXmpc = np.zeros((I+1,self.n_states))
            UUmpc = np.zeros((I,self.m_inputs))
            Q = X
            XXmpc[0,:] = Q
            t = 0
            for i in range(I):
                U = self.clfqp_a(Q,self.Xf,np.zeros(1))
                t,Q = self.one_step_sim(t,Q,U)
                XXmpc[i+1,:] = Q
                UUmpc[i,:] = U
        XX,UU,cvx_status = self.MPC_SCP(X,XXmpc,UUmpc,100,0,I)
        plt.figure()
        plt.plot(XX)
        plt.show()
        i = self.N-I
        self.XXmpc = XX
        self.UUmpc = UU
        print(i)
        U = self.clfqp_a(X,XX[0,:],UU[0,:])
        return U
    
    def fmincon(self,x0,I):
        N_eqcon = self.n_states*2+I*self.n_states
        N_noncon = I*self.m_inputs
        lb = np.zeros(N_eqcon)
        ub = np.zeros(N_eqcon)
        eqfun = lambda x: self.eq_con(x,I)
        eqc = sp.optimize.NonlinearConstraint(eqfun,lb,ub)
        lb = np.ones(N_noncon)*self.umin
        ub = np.ones(N_noncon)*self.umax
        #nlc = sp.optimize.NonlinearConstraint(self.nl_con,lb,ub)
        #fout = sp.optimize.minimize(self.objective,x0,constraints={eqc,nlc})
        fun = lambda x: self.objective(x,I)
        fout = sp.optimize.minimize(fun,x0,constraints=eqc)
        UU = fout.x[0:I*self.m_inputs].reshape(I,self.m_inputs)
        XX = fout.x[I*self.m_inputs:].reshape(I+1,self.n_states)
        return XX,UU
    
    def objective(self,x,I):
        J = 0
        for k in range(I):
            U = x[k*self.m_inputs:(k+1)*self.m_inputs]
            X = x[I*self.m_inputs+k*self.n_states:I*self.m_inputs+(k+1)*self.n_states]
            J += np.linalg.norm(U)**2#+np.linalg.norm(X)**2
        return J
    
    def eq_con(self,x,I):
        UU = x[0:I*self.m_inputs].reshape(I,self.m_inputs)
        XX = x[I*self.m_inputs:].reshape(I+1,self.n_states)
        X0 = XX[0,:]
        Xf = XX[-1,:]
        eqs = []
        for i in range(self.n_states):
            eqs.append(X0[i]-self.X0[i])
            eqs.append(Xf[i]-self.Xf[i])
        for k in range(I):
            U = UU[k,:]
            X = XX[k,:]
            Xp1 = XX[k+1,:]
            _,Xp1_a = self.one_step_sim(0,X,U)
            #p1_a = X+self.dynamics(0,X,U)*self.dt
            for i in range(self.n_states):
                eqs.append(Xp1_a[i]-Xp1[i])
        return eqs
    
    def nl_con(self,x,I):
        UU = x[0:I*self.m_inputs].reshape(I,self.m_inputs)
        eqs = []
        for k in range(I):
            U = UU[k,:]
            for j in range(self.m_inputs):
                eqs.append(U[j])
        return eqs
    
    def MPC_SCP(self,X,Xhis,Uhis,epochs,Tidx,I):
        dt = self.dt
        X0 = X
        Xf = self.Xf
        cvx_optval = np.inf
        p_prev = 1e16
        for i in range(epochs):
            XX = cp.Variable((I+1,self.n_states))
            UU = cp.Variable((I,self.m_inputs))
            constraints = []
            constraints += [XX[0,:] == X0]
            constraints += [XX[I,:] == Xf]
            Tregion = 0.1
            for k in range(I):
                Xkp1 = XX[k+1,:]
                Xk = XX[k,:]
                Uk = UU[k,:]
                Xs_k = Xhis[k,:]
                Us_k = Uhis[k,:]
                if Tidx == 1:
                    constraints += [cp.norm(Xk-Xs_k) <= Tregion]
                constraints += [Xkp1 == self.DynamicsLD(dt,Xk,Uk,Xs_k,Us_k)]
            J = cp.sum_squares(UU)
            prob = cp.Problem(cp.Minimize(J),constraints)
            try:
                prob.solve(solver=cp.MOSEK)
                print(prob.status)
                print("opt val",J.value)
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
        return XX.value,UU.value,cvx_status
    
    def SCPnormalized(self,Xhis,Uhis,epochs,Tidx):
        dt = self.dt
        N = self.N
        X0 = self.X0
        Xf = self.Xf
        cvx_optval = np.inf
        p_prev = 1e16
        for i in range(epochs):
            XX = cp.Variable((N+1,self.n_states))
            UU = cp.Variable((N,self.m_inputs))
            constraints = []
            constraints += [XX[0,:] == X0]
            constraints += [XX[N,:] == Xf]
            Tregion = 0.1
            for k in range(N):
                Xkp1 = XX[k+1,:]
                Xk = XX[k,:]
                Uk = UU[k,:]
                Xs_k = Xhis[k,:]
                Us_k = Uhis[k,:]
                if Tidx == 1:
                    constraints += [cp.norm(Xk-Xs_k) <= Tregion]
                constraints += [Xkp1 == self.DynamicsLD(dt,Xk,Uk,Xs_k,Us_k)]
            J = cp.sum_squares(UU)
            prob = cp.Problem(cp.Minimize(J),constraints)
            try:
                prob.solve(solver=cp.MOSEK)
                print(prob.status)
                print("opt val",J.value)
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
        return XX.value,UU.value,cvx_status
    
    def DynamicsLD(self,dt,Xk,Uk,Xs_k,Us_k):
        th_k = Xk[1]
        ths_k = Xs_k[1]
        fs = self.dynamicsf(Xs_k)
        dfdXs = self.getdfdX(Xs_k)
        gs = self.dynamicsg(Xs_k)
        dgdths = self.getdgdX(Xs_k)
        dXdt = fs+gs@Uk+dfdXs@(Xk-Xs_k)+(dgdths*(th_k-ths_k))@Us_k
        Xkp1 = Xk+dXdt*dt
        return Xkp1    
    
    def simulation(self):
        XXds = self.XXds
        UUds = self.UUds
        XXd0 = XXds[0,:]
        t = 0
        N = np.size(UUds[:,0])
        Xhis = np.zeros((N+1,self.n_states))
        Uhis = np.zeros((N,self.m_inputs))
        this = np.zeros(N+1)
        Xhis[0,:] = XXd0
        X = XXd0
        for k in range(N):
            Ud = UUds[k,:]
            Xd = XXds[k,:]
            #if np.abs(X[1]) <= self.th_robust:
            #    Ud = self.clfqp_a(X,Xd,Ud)
            #Uclf = self.clfqp_a(X,self.Xf,np.zeros(0))
            t,X = self.one_step_sim(t,X,Ud)
            Xhis[k+1,:] = Xd
            this[k+1] = t
        plt.figure()
        print("ueff sqp",np.sum(UUds**2))
        for i in range(self.n_states):
            plt.plot(this,Xhis[:,i])
        plt.show()
        return Xhis,UUds
    
    def preprocess_data(self,file_path):
        xdata = np.load(file_path+"xdata.npy")
        udata = np.load(file_path+"udata.npy")
        Ndata = np.shape(xdata)[0]
        Nsamp = self.N
        xdata2 = np.zeros((Ndata,Nsamp,self.n_states))
        xfdata = np.zeros((Ndata,Nsamp,self.n_states))
        for k in range(Ndata):
            xf = xdata[k,-1,:]
            for s in range(Nsamp):
                x = xdata[k,s,:]
                xdata2[k,s,:] = x
                xfdata[k,s,:] = xf-x
        xdata2 = xdata2.reshape(-1,self.n_states)
        udata = udata.reshape(-1,self.m_inputs)
        xfdata = xfdata.reshape(-1,self.n_states)
        N = Ndata*Nsamp
        idx_d = np.random.choice(N,N,replace=False)
        xdata2 = xdata2[idx_d,:]
        udata = udata[idx_d,:]
        xfdata = xfdata[idx_d,:]
        return xdata2,udata,xfdata
    
    def preprocess_data_robust(self,file_path,idx_load):
        if idx_load == 0:
            xdata = np.load(file_path+"xdata.npy")
            udata = np.load(file_path+"udata.npy")
            Ndata = np.shape(xdata)[0]
            Nsamp = self.N
            Nrobust = 100
            xdata_r = []
            udata_r = []
            xfdata = []
            for k in range(Ndata):
                for s in range(Nsamp):
                    xd = xdata[k,s,:]
                    ud = udata[k,s,:]
                    xf = xdata[k,-1,:]
                    for r in range(Nrobust):
                        d = unifrand2(5,self.q_states)
                        d = np.hstack((np.zeros(self.q_states),d))
                        x = xd+d*self.dt
                        u = self.clfqp_a(x,xd,ud)
                        if  np.linalg.norm(u) <= 200:
                            u = self.clfqp_a(x,xd,ud)
                            xdata_r.append(x)
                            udata_r.append(u)
                            xfdata.append(xf-x)
                        else:
                            print(u)
            xdata_r = np.array(xdata_r)
            udata_r = np.array(udata_r)
            xfdata = np.array(xfdata)
            xdata_r = xdata_r.reshape(-1,self.n_states)
            udata_r = udata_r.reshape(-1,self.m_inputs)
            xfdata = xfdata.reshape(-1,self.n_states)
            N = xdata_r.shape[0]
            idx_d = np.random.choice(N,N,replace=False)
            xdata_r = xdata_r[idx_d,:]
            udata_r = udata_r[idx_d,:]
            xfdata = xfdata[idx_d,:]
            np.save("data/cartpole/xdata_pp.npy",xdata_r)
            np.save("data/cartpole/udata_pp.npy",udata_r)
            np.save("data/cartpole/xfdata_pp.npy",xfdata)
            self.xdata_r = xdata_r
            self.udata_r = udata_r
            self.xfdata = xfdata
        else:
            self.xdata_r = np.load("data/cartpole/xdata_pp.npy")
            self.udata_r = np.load("data/cartpole/udata_pp.npy")
            self.xfdata = np.load("data/cartpole/xfdata_pp.npy")
        pass
    
    def preprocess_data_robust_ncm(self,file_path,idx_load,iTrain):
        if idx_load == 0:
            xdata = np.load(file_path+"xdata.npy")
            udata = np.load(file_path+"udata.npy")
            Ndata = np.shape(xdata)[0]
            Nsamp = self.N
            Nrobust = 100
            xdata_r = []
            udata_r = []
            xfdata = []
            ncm = self.ncm_train(iTrain)
            for k in range(Ndata):
                print(k)
                for s in range(Nsamp):
                    xd = xdata[k,s,:]
                    ud = udata[k,s,:]
                    xf = xdata[k,-1,:]
                    for r in range(Nrobust):
                        d = unifrand2(ncm.Jcv_opt,self.q_states)
                        d = np.hstack((np.zeros(self.q_states),d))
                        x = xd+d*self.dt
                        u = self.clfqp_a_ncm(x,xd,ud,ncm)
                        if  np.linalg.norm(u) <= 200:
                            u = self.clfqp_a(x,xd,ud)
                            xdata_r.append(x)
                            udata_r.append(u)
                            xfdata.append(xf-x)
                        else:
                            print(u)
            xdata_r = np.array(xdata_r)
            udata_r = np.array(udata_r)
            xfdata = np.array(xfdata)
            xdata_r = xdata_r.reshape(-1,self.n_states)
            udata_r = udata_r.reshape(-1,self.m_inputs)
            xfdata = xfdata.reshape(-1,self.n_states)
            N = xdata_r.shape[0]
            idx_d = np.random.choice(N,N,replace=False)
            xdata_r = xdata_r[idx_d,:]
            udata_r = udata_r[idx_d,:]
            xfdata = xfdata[idx_d,:]
            np.save("data/cartpole/xdata_pp.npy",xdata_r)
            np.save("data/cartpole/udata_pp.npy",udata_r)
            np.save("data/cartpole/xfdata_pp.npy",xfdata)
            self.xdata_r = xdata_r
            self.udata_r = udata_r
            self.xfdata = xfdata
        else:
            self.xdata_r = np.load("data/cartpole/xdata_pp.npy")
            self.udata_r = np.load("data/cartpole/udata_pp.npy")
            self.xfdata = np.load("data/cartpole/xfdata_pp.npy")
        pass

    def train_glas(self,file_path,Nbatch=100,Nphi=32,Nrho=32,Nlayers=3,Nunits=100,Nepochs=1000,
              ValidationSplit=0.1,Patience=2000,Verbose=1):
        xdata2,udata,xfdata = self.preprocess_data(file_path)
        Ndata = 100*self.N
        X = xfdata[0:Ndata,:]
        y = udata[0:Ndata,:]
        
        model = Sequential(name="cartpole")
        model.add(Dense(Nunits,activation='relu',input_shape=(self.n_states,)))
        for l in range(Nlayers-1):
            model.add(Dense(Nunits,activation='relu'))
        model.add(Dense(self.m_inputs,activation=None))
        model.compile(loss="mean_squared_error",optimizer="adam")
        es = EarlyStopping(monitor="val_loss",patience=Patience)
        model.fit(X,y,batch_size=Nbatch,epochs=Nepochs,verbose=Verbose,\
                      callbacks=[es],validation_split=ValidationSplit)
        
        self.model = model
        model.save("data/cartpole/neuralnets/model_glas.h5")
        pass
    def train_robust_glas(self,file_path,idx_load=1,Nbatch=32,Nphi=32,Nrho=32,Nlayers=3,Nunits=100,Nepochs=1000,
    
              ValidationSplit=0.1,Patience=2000,Verbose=1,iTrain=0):
        self.preprocess_data_robust_ncm(file_path,idx_load,iTrain)
        xdata_r = self.xdata_r
        udata_r = self.udata_r
        xfdata = self.xfdata
        Ndata = 100*self.N
        X = xfdata[0:Ndata,:]
        y = udata_r[0:Ndata,:]
        
        model = Sequential(name="cartpole_r")
        model.add(Dense(Nunits,activation='relu',input_shape=(self.n_states,)))
        for l in range(Nlayers-1):
            model.add(Dense(Nunits,activation='relu'))
        model.add(Dense(self.m_inputs,activation=None))
        model.compile(loss="mean_squared_error",optimizer="adam")
        es = EarlyStopping(monitor="val_loss",patience=Patience)
        model.fit(X,y,batch_size=Nbatch,epochs=Nepochs,verbose=Verbose,\
                      callbacks=[es],validation_split=ValidationSplit)
        self.model = model
        model.save("data/cartpole/neuralnets/model_robust_glas.h5")
        pass

    def clfqp_a(self,X,Xd,Ud):
        #Xn = np.array([1,np.pi/3,1,np.pi/3])
        #An = self.sdcA(Xn)
        #Bn = self.dynamicsg(Xn)
        e = X-Xd
        B = self.dynamicsg(X)
        Bd = self.dynamicsg(Xd)
        f = self.dynamicsf(X)+B@Ud
        fd = self.dynamicsf(Xd)+Bd@Ud
        Q = np.identity(self.n_states)*10
        R = np.identity(self.m_inputs)
        A = self.sdcA(X)
        try:
            K,P,_ = control.lqr(A,B,Q,R)
            invR = np.linalg.inv(R)
            phi0 = 2*e@P@(f-fd)+e@(P@B@invR@B.T@P+Q)@e
            phi1 = (2*e@P@B).T
            if phi0 > 0:
                U = -(phi0/(phi1.T@phi1))*phi1
                U = Ud+U.ravel()
            else:
                U = Ud
        except:
            U = Ud
            print("clfqp infeasible")
        return U

    def clfqp_a_ncm(self,X,Xd,Ud,ncm):
        #Xn = np.array([1,np.pi/3,1,np.pi/3])
        #An = self.sdcA(Xn)
        #Bn = self.dynamicsg(Xn)
        e = X-Xd
        B = self.dynamicsg(X)
        Bd = self.dynamicsg(Xd)
        f = self.dynamicsf(X)+B@Ud
        fd = self.dynamicsf(Xd)+Bd@Ud
        Q = np.identity(self.n_states)*5
        R = np.identity(self.m_inputs)
        A = self.sdcA(X)
        try:
            M = ncm.ncm(X,np.empty(0))
            alp = ncm.alp_opt
            phi0 = 2*e@M@(f-fd)+2*alp*e@M@e
            phi1 = (2*e@M@B).T
            if phi0 > 0:
                U = -(phi0/(phi1.T@phi1))*phi1
                U = Ud+U.ravel()
            else:
                U = Ud
        except:
            U = Ud
            print("clfqp infeasible")
        return U
    
    def nmpc_simulation(self,file_path_nn,dover):
        model = load_model(file_path_nn+"/model_glas.h5")
        model_r = load_model(file_path_nn+"/model_robust_glas.h5")
        N = self.N
        dt = self.dt
        this = []
        Ncon = 5
        Xhis = np.zeros((N+1,Ncon,self.n_states))
        Uhis = np.zeros((N,Ncon,self.m_inputs))
        dthis = np.zeros((N,Ncon))
        tc0 = time.time()
        self.scp()
        dtc0_scp = time.time()-tc0
        X0 = self.X0
        Xf = self.Xf
        XXd = self.XXds
        UUd = self.UUds
        t = 0
        this.append(t)
        Xs = np.zeros((Ncon,self.n_states))
        idx_fin = np.zeros(Ncon)
        for c in range(Ncon):
            Xs[c,:] = X0
            Xhis[0,c,:] = X0
        for i in range(N):
            tnow = t
            d = unifrand2(dover,self.q_states)
            d = np.hstack((np.zeros(self.q_states),d))
            Us = np.zeros((Ncon,self.m_inputs))
            for c in range(Ncon):
                X = Xs[c,:]
                Xnn = Xf-X
                if c == 0:
                    tc0 = time.time()
                    U = self.clfqp_a(X,self.Xf,np.zeros(1))
                    dtc0 = time.time()-tc0
                elif c == 1:
                    tc0 = time.time()
                    Xd = XXd[i,:]
                    Ud = UUd[i,:]
                    U = self.clfqp_a(X,Xd,Ud)
                    dtc0 = time.time()-tc0+dtc0_scp
                elif c == 2:
                    tc0 = time.time()
                    U = model.predict(np.array([Xnn]))
                    U = U.ravel()
                    dtc0 = time.time()-tc0
                elif c == 3:
                    tc0 = time.time()
                    U = model_r.predict(np.array([Xnn]))
                    U = U.ravel()
                    dtc0 = time.time()-tc0
                elif c == 4:
                    if i%10 == 0:
                        tc0 = time.time()
                        U = self.online_mpc(X,self.N-i)
                        Ith = i
                        dtc0 = time.time()-tc0
                        print(dtc0)
                    else:
                        impc = i-Ith
                        tc0 = time.time()
                        U = self.clfqp_a(X,self.XXmpc[impc,:],self.UUmpc[impc,:])
                        dtc0 = time.time()-tc0
                    
                #if np.abs(X[1]) <= np.pi/6:
                #Ud = UUd[0,i,:]
                #print(np.linalg.norm(U-Ud))
                t,X = self.one_step_sim(t,X,U)
                X += d*dt
                t = tnow
                Xs[c,:] = X
                Us[c,:] = U
                dthis[i,c] = dtc0
            t += dt
            this.append(t)
            Xhis[i+1,:,:] = Xs
            Uhis[i,:,:] = Us
        this = np.array(this)
        ueffs = []
        for c in range(Ncon):
            Xhisc = Xhis[:,c,:]
            ueffc = 0
            plt.figure()
            for i in range(self.n_states):
                colorp = "C"+str(i)
                plt.plot(this,Xhisc[:,i],color=colorp)
            #ueffc = np.sum(np.sqrt(np.sum(Uhis[:,c,:]**2,1)))
            ueffc = np.sum(Uhis[:,c,:]**2)
            ueffs.append(ueffc)
            plt.grid()
            plt.show()
        print(ueffs)
        eps = 2
        for c in range(Ncon):
            Xend = Xhis[-1,c,:]
            if np.linalg.norm(Xend[0:2]-Xf[0:2]) <= eps:
                idx_fin[c] = 1
        print(idx_fin)
        return this,dthis,Xhis,Uhis,ueffs,idx_fin
    
def unifrand2(d_over=1,nk=4):
    d_over_out = d_over+1
    while d_over_out > d_over:
            d = np.random.uniform(-d_over,d_over,size=nk)
            d_over_out = np.linalg.norm(d)
    return d

if __name__ == "__main__":
    """
    Nsim = 100
    
    xdata = []
    udata = []
    fails = []
    for i in range(Nsim):
        start_time = time.time()
        nmpc = NMPC()
        nmpc.scp()
            #print(nmpc.cvx_status)
            #if nmpc.cvx_status:
            #    break
            #else:
            #    fails.append(i)
        xhis,uhis = nmpc.simulation()
        xdata.append(np.array(nmpc.XXds))
        udata.append(np.array(nmpc.UUds))
        np.save("data/cartpole/xdata.npy",np.array(xdata))
        np.save("data/cartpole/udata.npy",np.array(udata))
        t_one_epoch = time.time()-start_time
        print("time:",t_one_epoch,"s")
        print("i =",i)
    xdata = np.array(xdata)
    udata = np.array(udata)
    np.save("data/cartpole/xdata.npy",np.array(xdata))
    np.save("data/cartpole/udata.npy",np.array(udata))
    
    
    
    nmpc = NMPC()
    nmpc.XXds = np.load("data/cartpole/xdata.npy")[0,:,:]
    nmpc.UUds = np.load("data/cartpole/udata.npy")[0,:,:]
    xhis,uhis = nmpc.simulation()
    """
    
    
    nmpc = NMPC()
    file_path = "data/cartpole/"
    file_path_nn = "data/cartpole/neuralnets/"
    #nmpc.train_robust_glas(file_path,idx_load=0,iTrain=1)
    #nmpc.train_glas(file_path)
    """
    dovers = [0,0.2,0.4,0.6,0.8,1.0]
    dovers = [5]
    for dover in dovers:        
        this,dthis,Xshis,Ushis,ueffs,idx_fin = nmpc.nmpc_simulation(file_path_nn,dover)
        np.save("data/cartpole/sim/this_"+str(dover*10)+".npy",this)
        np.save("data/cartpole/sim/dthis_"+str(dover*10)+".npy",dthis)
        np.save("data/cartpole/sim/Xshis_"+str(dover*10)+".npy",Xshis)
        np.save("data/cartpole/sim/Ushis_"+str(dover*10)+".npy",Ushis)
        np.save("data/cartpole/sim/ueffs_"+str(dover*10)+".npy",ueffs)
        np.save("data/cartpole/sim/X0_"+str(dover*10)+".npy",nmpc.X0)
        np.save("data/cartpole/sim/Xf_"+str(dover*10)+".npy",nmpc.Xf)
    """
    dover = 2
    Ndata = 50
    thiss = []
    dthiss = []
    Xshiss = []
    Ushiss = []
    ueffss = []
    idx_fins = []
    X0s = []
    Xfs = []
    for i in range(Ndata):
        print("Ndata =",i)
        nmpc = NMPC()
        this,dthis,Xshis,Ushis,ueffs,idx_fin = nmpc.nmpc_simulation(file_path_nn,dover)
        thiss.append(this)
        dthiss.append(dthis)
        Xshiss.append(Xshis)
        Ushiss.append(Ushis)
        ueffss.append(ueffs)
        idx_fins.append(idx_fin)
        X0s.append(nmpc.X0)
        Xfs.append(nmpc.Xf)
        np.save("data/cartpole/sim/thiss_"+str(dover*10)+".npy",np.array(thiss))
        np.save("data/cartpole/sim/dthiss_"+str(dover*10)+".npy",np.array(dthiss))
        np.save("data/cartpole/sim/Xshiss_"+str(dover*10)+".npy",np.array(Xshiss))
        np.save("data/cartpole/sim/Ushiss_"+str(dover*10)+".npy",np.array(Ushiss))
        np.save("data/cartpole/sim/ueffss_"+str(dover*10)+".npy",np.array(ueffss))
        np.save("data/cartpole/sim/idx_fins_"+str(dover*10)+".npy",np.array(idx_fins))
        np.save("data/cartpole/sim/X0s_"+str(dover*10)+".npy",np.array(X0s))
        np.save("data/cartpole/sim/Xfs_"+str(dover*10)+".npy",np.array(Xfs))
        
        
    
    