# -*- coding: utf-8 -*-
"""
Created on Wed May 12 08:43:21 2021

@author: carca
"""

#Import des bibliothèques

import numpy as np
import matplotlib.pyplot as py
from math import exp

#Déclarations variables 

Tmax=2
nu=2
h=0.005
N=round(1/h)
tau=0.0002

#Définitions des fonctions

def CalculU0(x):
    return (np.sin(np.pi*x))**10

def CalculMe1(N,c):
    Me=np.eye(N)
    for i in range(N-1):
        Me[i,i+1]=c/2
        Me[i+1,i]=-c/2
    Me[N-1,0]=c/2
    Me[0,N-1]=-c/2
    return Me

def CalculMe2(N,c):
    Me=np.eye(N)
    for i in range(N-1):
        Me[i,i+1]=-c/2
        Me[i+1,i]=c/2
    Me[N-1,0]=-c/2
    Me[0,N-1]=c/2
    return Me

def CalculMe3(N,c):
    Me=(1-c)*np.eye(N)
    for i in range(N-1):
        Me[i+1,i]=c
    Me[0,N-1]=c
    return Me

def CalculMe4(N,c):
    Me=0*np.eye(N)
    for i in range (N-1):
        Me[i+1,i]=(1+c)/2
        Me[i,i+1]=(1-c)/2
    Me[0,N-1]=(1+c)/2
    Me[N-1,0]=(1-c)/2
    return Me

def CalculMe5(N,c):
    Me=(1-c**2)*np.eye(N)
    for i in range(N-1):
        Me[i,i+1]=(c**2-c)/2
        Me[i+1,i]=(c**2+c)/2
    Me[0,N-1]=(c+c**2)/2
    Me[N-1,0]=(c**2-c)/2
    return Me

#Implicite centré
def SolIC(N,tau): 
    X=np.linspace(0.,1.,N+1)
    nmax=int(Tmax/tau)
    T=np.linspace(0.,tau*nmax,nmax+1)
    Xint=X[1:N+1]
    u_n=CalculU0(Xint)
    h=1/(N)
    c=nu*tau/h
    Me=CalculMe1(N,c)
    U=np.zeros((nmax+1,N+1))
    U[0,0:N]=u_n
    for n in range(nmax):
        u_n=np.linalg.solve(Me,u_n) #u^(n+1)
        U[n+1,0:N]=u_n
    U[:,N]=U[:,0]
    return X,T,U

#Explicite centré
def SolEC(N,tau): 
    X=np.linspace(0.,1.,N+1)
    nmax=int(Tmax/tau)
    T=np.linspace(0.,tau*nmax,nmax+1)
    Xint=X[1:N+1]
    u_n=CalculU0(Xint)
    h=1/(N)
    c=nu*tau/h
    Me=CalculMe2(N,c)
    U=np.zeros((nmax+1,N+1))
    U[0,0:N]=u_n
    for n in range(nmax):
        u_n=Me @ u_n #u^(n+1)
        U[n+1,0:N]=u_n
    U[:,N]=U[:,0]
    return X,T,U

#Décentré amont
def SolDA(N,tau):
    X=np.linspace(0.,1.,N+2)
    nmax=int(Tmax/tau)
    T=np.linspace(0.,tau*nmax,nmax+1)
    Xint=X[1:N+1]
    u_n=CalculU0(Xint)
    h=1/(N)
    c=nu*tau/h
    Me=CalculMe3(N,c)
    U=np.zeros((nmax+1,N+2))
    U[0,0:N]=u_n
    for n in range(nmax):
        u_n=Me @ u_n #u^(n+1)
        U[n+1,0:N]=u_n
    U[:,N]=U[:,0]
    return X,T,U

#Lax-Friedrichs
def SolFr(N,tau):
    X=np.linspace(0.,1.,N+2)
    nmax=int(Tmax/tau)
    T=np.linspace(0.,tau*nmax,nmax+1)
    Xint=X[1:N+1]
    u_n=CalculU0(Xint)
    h=1/(N)
    c=nu*tau/h
    Me=CalculMe4(N,c)
    U=np.zeros((nmax+1,N+2))
    U[0,0:N]=u_n
    for n in range(nmax):
        u_n=Me @ u_n #u^(n+1)
        U[n+1,0:N]=u_n
    U[:,N]=U[:,0]
    return X,T,U

#Lax-Wendroff
def SolWen(N,tau):
    X=np.linspace(0.,1.,N+2)
    nmax=int(Tmax/tau)
    T=np.linspace(0.,tau*nmax,nmax+1)
    Xint=X[1:N+1]
    u_n=CalculU0(Xint)
    h=1/(N)
    c=nu*tau/h
    Me=CalculMe5(N,c)
    U=np.zeros((nmax+1,N+2))
    U[0,0:N]=u_n
    for n in range(nmax):
        u_n=Me @ u_n #u^(n+1)
        U[n+1,0:N]=u_n
    U[:,N]=U[:,0]
    return X,T,U

#Tracé des courbes
X1,T1,U1=SolIC(N,tau)
X2,T2,U2=SolEC(N,tau)
X3,T3,U3=SolDA(N,tau)
X4,T4,U4=SolFr(N,tau)
X5,T5,U5=SolWen(N,tau)

for t in np.arange(0.,3.):
    n=int(round(t/tau))
    py.plot(X1,U1[n,:],label='t='+str(t))
py.title("Centré implicite")
py.ylabel("u(x,t)")
py.xlabel("X")
py.legend()
py.grid()
py.show()

for t in np.arange(0.,3.):
    n=int(round(t/tau))
    py.plot(X2,U2[n,:],label='t='+str(t))
py.title("Centré explicite")
py.ylabel("u(x,t)")
py.xlabel("X")
py.legend()
py.grid()
py.show()

for t in np.arange(0.,3.):
    n=int(round(t/tau))
    py.plot(X3,U3[n,:],label='t='+str(t))
py.title("Décentré amont")
py.ylabel("u(x,t)")
py.xlabel("X")
py.legend()
py.grid()
py.show()

for t in np.arange(0.,3.):
    n=int(round(t/tau))
    py.plot(X4,U4[n,:],label='t='+str(t))
py.title("Lax-Friedrichs")
py.ylabel("u(x,t)")
py.xlabel("X")
py.legend()
py.grid()
py.show()

for t in np.arange(0.,3.):
    n=int(round(t/tau))
    py.plot(X5,U5[n,:],label='t='+str(t))
py.title("Lax-Wendroff")
py.ylabel("u(x,t)")
py.xlabel("X")
py.legend()
py.grid()
py.show()