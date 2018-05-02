#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from scipy.special import digamma, polygamma
from decimal import Decimal

def E_step_Plain(alpha, BETA, doc,Phi0, gamma0, max_iter=100, tol=1e-3):
    """
    Latent Dirichlet Allocation: E-step.
    Do to a specific document.
    ------------------------------------
    Input:
    alpha as a k*1 vector;
    BETA as a k*V matrix;
    doc as a Nd*V matrix;
    Phi0 as a Nd*k matrix;
    gamma0 as a k*1 vector;
    max_iter as a float: max iteration;
    tol as a float: tolerance.
    -------------------------------------
    Output:
    optimal Nd*k matrix Phi;
    optimal k*1 vector gamma."""
    
    Nd,V = doc.shape
    k = len(alpha)
    
    Phi=Phi0
    gamma=gamma0
    phi_delta = 1
    gamma_delta = 1

    for iteration in range(max_iter):
        for n in range(Nd): #for each word 
            for i in range(k): #for each topic
                Phi[n, i] = np.sum(BETA[i,]*doc[n,]) * np.exp(digamma(gamma[i])-digamma(sum(gamma)))
            Phi[n,] = Phi[n,] / np.sum(Phi[n,])
        gamma = alpha + np.sum(Phi[n,])
        phi_delta = np.sum((Phi - Phi0) ** 2)
        gamma_delta = np.sum((gamma - gamma0) ** 2)
        Phi0 = Phi
        gamma0 = gamma
        if((phi_delta <= tol) and (gamma_delta <= tol)):
            break
        
    return Phi, gamma

def M_step_Plain(docs,k, tol=1e-3,tol_estep=1e-3,max_iter=100,initial_alpha_shape=100,initial_alpha_scale=0.01):

    """
    Latent Dirichlet Allocation: M-step.
    Do to a list of documnents. -- a list of matrix.
    -------------------------------------------------
    Input:
    docs: a list of one-hot-coding matrix ;
    k: a fixed positive integer indicate the number of topics.
    tol,tol_estep: tolerance;
    max_iter: max iteration;
    inital_alpha_shape,scale;
    -------------------------------------------------
    Output:
    optimal Nd*k matrix Phi;
    optimal k*1 vector gamma;
    optimal k*V matrix BETA;
    optimal k*1 vector alpha.
    """
    
    #get basic iteration
    M=len(docs)
    V=docs[1].shape[1]
    N=[doc.shape[0] for doc in docs]

    
    #initialization
    BETA0=np.random.dirichlet(np.ones(V),k)
    alpha0=np.random.gamma(shape=initial_alpha_shape,scale=initial_alpha_scale,size=k)
    PHI=[np.ones((N[d],k))/k for d in range(M)]
    GAMMA=np.array([alpha0+N[d]/k for d in range(M)])     

    BETA=BETA0
    alpha=alpha0
    alpha_dis = 1
    beta_dis = 1

    for iteration in range(max_iter):
        
        #update PHI,GAMMA
        for d in range(M):
            PHI[d], GAMMA[d,] = E_step_Plain(alpha0, BETA0, docs[d],PHI[d],GAMMA[d,],max_iter,tol_estep)
        
        #update BETA
        for i in range(k):
            for j in range(V):
                beta=0
                for d in range(M): #documents
                    doc = docs[d] 
                    Phi=PHI[d]
                    for n in range(N[d]):
                        beta+=Phi[n,i]*doc[n,j]
                BETA[i,j]=beta
            BETA[i,]=BETA[i,]/sum(BETA[i,])
        
        #update alpha
                        
        GAMMA_rowsum = [sum(gamma) for gamma in GAMMA]
        sum_gamma=0
        for d in range(M):
            sum_gamma+=digamma(GAMMA[d,])-digamma(GAMMA_rowsum[d])
        z = M * polygamma(1, sum(alpha0))
        h=-M*polygamma(1,alpha0)
        g=M*(digamma(sum(alpha0))-digamma(alpha0))+sum_gamma
        c = sum(g / h) / (1/z + sum(1/h))
        alpha = alpha0 - (g - c)/h
        alpha_dis = np.sum((alpha - alpha0) ** 2)
        beta_dis = np.sum((BETA - BETA0) ** 2)
        alpha0 = alpha
        BETA0 = BETA
        iteration+=1
        if((alpha_dis <= tol) and (beta_dis <= tol)):
            break
     
    return alpha, BETA

def E_step_Structure(alpha, BETA, doc, Phi0, gamma0, max_iter=100,tol=1e-3):
    """
    Latent Dirichlet Allocation: E-step.
    Do to a specific document.
    ------------------------------------
    Input:
    alpha as a k*1 vector;
    BETA as a k*V matrix;
    doc as a Nd*1 vector;
    Phi0 as a Nd*k matrix;
    gamma0 as a k*1 vector;
    tol as a float: tolerance.
    -------------------------------------
    Output:
    optimal Nd*k matrix Phi;
    optimal k*1 vector gamma."""
    
    #Initialization
    Phi = Phi0
    gamma = gamma0
    phi_delta = 1
    gamma_delta = 1
    
    #relative tolerance is for each element in the matrix
    tol=tol**2
 

    for iteration in range(max_iter):
        ##update Phi
        Phi=(BETA.T[doc,])*np.exp(digamma(gamma)-digamma(sum(gamma))) #Here changed
        Phi=Phi/(Phi.sum(axis=1)[:,None]) #row sum to 1
        
        ##update gamma
        gamma = alpha + Phi.sum(axis = 0)
        
        ##check the convergence
        phi_delta = np.mean((Phi - Phi0) ** 2)
        gamma_delta = np.mean((gamma - gamma0) ** 2)
        
        ##refill
        Phi0 = Phi
        gamma0 = gamma
        if((phi_delta <= tol) and (gamma_delta <= tol)):
            break
        
    return Phi, gamma

def M_step_Structure(docs,k, tol=1e-3,tol_estep=1e-3,max_iter=100,initial_alpha_shape=100,initial_alpha_scale=0.01):
    """
    Latent Dirichlet Allocation: M-step.
    Do to a list of documnents. -- a list of matrix.
    -------------------------------------------------
    Input:
    docs: a list of Nd*1 vector ;
    k: a fixed positive integer indicate the number of topics.
    -------------------------------------------------
    Output:
    optimal Nd*k matrix Phi;
    optimal k*1 vector gamma;
    optimal k*V matrix BETA;
    optimal k*1 vector alpha.
    """
    
    #get basic iteration
    M=len(docs)
    V=docs[1].shape[1]
    N=[doc.shape[0] for doc in docs]

    
    #initialization
    BETA0=np.random.dirichlet(np.ones(V),k)
    alpha0=np.random.gamma(shape=initial_alpha_shape,scale=initial_alpha_scale,size=k)
    PHI=[np.ones((N[d],k))/k for d in range(M)]
    GAMMA=np.array([alpha0+N[d]/k for d in range(M)])       
    

    BETA=BETA0
    alpha=alpha0
    alpha_dis = 1
    beta_dis = 1
    
    #relative tolerance: tolerance for each element
    tol=tol**2
    
    for iteration in range(max_iter):
    
        #update PHI,GAMMA,BETA
        BETA = np.zeros((k,V))
        for d in range(M): #documents
            PHI[d], GAMMA[d,] = E_step_Structure(alpha0, BETA0, np.nonzero(docs[d])[1],PHI[d],GAMMA[d,],max_iter,tol_estep) #here changed
            BETA += PHI[d].T @ docs[d]   
        BETA=BETA/(BETA.sum(axis=1)[:,None])   #rowsum=1
        
        #update alpha
                        
        z = M * polygamma(1, sum(alpha0))
        h=-M*polygamma(1,alpha0)
        g=M*(digamma(sum(alpha0))-digamma(alpha0))+(digamma(GAMMA)-digamma(GAMMA.sum(axis=1))[:,None]).sum(axis=0)
        c = (sum(g / h)) / (1/z + sum(1/h))
        alpha = alpha0 - (g - c)/h
        
        alpha_dis = np.mean((alpha - alpha0) ** 2)
        beta_dis = np.mean((BETA - BETA0) ** 2)
        alpha0 = alpha
        BETA0 = BETA
        if((alpha_dis <= tol) and (beta_dis <= tol)):
            break
        
        
    return alpha, BETA


def E_step_Realdata(alpha, BETA, doc, Phi0, gamma0, max_iter=100, tol=1e-3):
    """
    Latent Dirichlet Allocation: E-step.
    Do to a specific document.
    ------------------------------------
    Input:
    alpha as a k*1 vector;
    BETA as a k*V matrix;
    doc as a Nd*V matrix;
    Phi0 as a Nd*k matrix;
    gamma0 as a k*1 vector;
    tol as a float: tolerance.
    -------------------------------------
    Output:
    optimal Nd*k matrix Phi;
    optimal k*1 vector gamma."""

    
    #Initialization
    Phi = Phi0
    gamma = gamma0
    phi_delta = 1
    gamma_delta = 1
    
    Phi = Phi.astype(np.float128)
    Phi0 = Phi0.astype(np.float128)
    
    #relative tolerance is for each element in the matrix
    tol=tol**2
    for iteration in range(max_iter):
        ##update Phi
        gamma = gamma / min(gamma)
        Phi=(doc@BETA.T)*np.exp(digamma(gamma)-digamma(sum(gamma)))
        Phi=Phi/(Phi.sum(axis=1)[:,None]) #row sum to 1
        
        ##update gamma
        gamma = alpha + Phi.sum(axis = 0)
        
        ##check the convergence
        phi_delta = np.mean((Phi - Phi0) ** 2)
        gamma_delta = np.mean((gamma - gamma0) ** 2)
        
        ##refill
        Phi0 = Phi
        gamma0 = gamma
        if ((phi_delta<=tol) and (gamma_delta<=tol)):
            break
        
    return Phi, gamma

def M_step_Realdata(docs,k, tol=1e-3,tol_estep=1e-3,max_iter=100,initial_alpha_shape=5,initial_alpha_scale=2):
    """
    Latent Dirichlet Allocation: M-step.
    Do to a list of documnents. -- a list of matrix.
    -------------------------------------------------
    Input:
    docs: a list of one-hot-coding matrix ;
    k: a fixed positive integer indicate the number of topics.
    -------------------------------------------------
    Output:
    optimal Nd*k matrix Phi;
    optimal k*1 vector gamma;
    optimal k*V matrix BETA;
    optimal k*1 vector alpha.
    """
    
    #get basic iteration
    M=len(docs)
    V=docs[1].shape[1]
    N=[doc.shape[0] for doc in docs]

    
    #initialization
    BETA0=np.random.dirichlet(np.ones(V),k)
    alpha0=np.random.gamma(shape=initial_alpha_shape,scale=initial_alpha_scale,size=k)
    PHI=[np.ones((N[d],k))/k for d in range(M)]
    GAMMA=np.array([alpha0+N[d]/k for d in range(M)])       
    

    BETA=BETA0
    alpha=alpha0
    alpha_dis = 1
    beta_dis = 1
    
    #relative tolerance: tolerance for each element
    tol=tol**2
    
    for iteration in range(max_iter):
    
        #update PHI,GAMMA,BETA
        BETA = np.zeros((k,V))
        for d in range(M): #documents
            PHI[d], GAMMA[d,] = E_step_Realdata(alpha0, BETA0, docs[d],PHI[d],GAMMA[d,],max_iter,tol_estep)
            BETA += PHI[d].T @ docs[d]    
        BETA=BETA/(BETA.sum(axis=1)[:,None])   #rowsum=1
        
        #update alpha
                        
        z = M * polygamma(1, sum(alpha0))
        h=-M*polygamma(1,alpha0)
        g=M*(digamma(sum(alpha0))-digamma(alpha0))+(digamma(GAMMA)-digamma(GAMMA.sum(axis=1))[:,None]).sum(axis=0)
        c = (sum(g / h)) / (1/z + sum(1/h))
        alpha = alpha0 - (g - c)/h
        
        alpha_dis = np.mean((alpha - alpha0) ** 2)
        beta_dis = np.mean((BETA - BETA0) ** 2)
        alpha0 = alpha
        BETA0 = BETA
        if((alpha_dis <= tol) and (beta_dis <= tol)):
            break
          
    return alpha, BETA
