#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from scipy.special import digamma, polygamma


def E_step_Smoothing(alpha, LAMBDA, doc, Phi0, gamma0, max_iter=100,tol=1e-6):
    """
    Smoothing Latent Dirichlet Allocation: E-step.
    Do to a specific document.
    ------------------------------------
    Input:
    alpha as a k*1 vector;
    LAMBDA as a k*V matrix;
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
    phi_delta = Nd*k
    gamma_delta = k
    
    #relative tolerance is for each element in the matrix
    tol=tol**2
 
    for iteration in range(max_iter):
        ##update Phi
        Phi=(doc@(np.exp(digamma(LAMBDA)-(digamma(LAMBDA.sum(axis=1))[:,None]))).T)*np.exp(digamma(gamma)-digamma(sum(gamma)))
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


def M_step_Smoothing(docs,k, tol=1e-3,tol_estep=1e-3,initial_alpha_shape,initial_alpha_scale,initial_eta_shape,initial_eta_scale,max_iter=100):
    """
    Smoothing Latent Dirichlet Allocation: M-step.
    Do to a list of documnents. -- a list of matrix.
    -------------------------------------------------
    Input:
    docs: a list of one-hot-coding matrix ;
    k: a fixed positive integer indicate the number of topics;
    tol: tolerance for the M_step
    tol_estep: tolerance for E_step;
    max_iter: max iteration for M_step;
    initial_eta: initialization for eta;
    -------------------------------------------------
    Output:
    optimal Nd*k matrix Phi;
    optimal k*1 vector gamma;
    optimal k*V matrix LAMBDA;
    optimal k*1 vector alpha;
    optimal V*1 vector eta. 
    """
    
    #get basic iteration
    M=len(docs)
    V=docs[1].shape[1]
    N=[doc.shape[0] for doc in docs]

    
    #initialization
    eta0=np.random.gamma(shape=initial_eta_shape,scale=initial_eta_scale,size=V)
    alpha0=np.random.gamma(shape=initial_alpha_shape,scale=initial_alpha_scale,size=k)
    PHI=[np.ones((N[d],k))/k for d in range(M)]
    GAMMA=np.array([alpha0+N[d]/k for d in range(M)])
    LAMBDA0=np.random.gamma(shape=initial_eta_shape,scale=initial_eta_scale,size=(k,V))
    

    eta=eta0
    alpha=alpha0
    alpha_dis = 1
    lambda_dis = 1
    eta_dis=1
    
    
    #relative tolerance: tolerance for each element
    tol=tol**2
    
    for iteration in range(max_iter):
    
        #update PHI,GAMMA,LAMBDA
        LAMBDA = np.zeros((k,V))
        for d in range(M): #documents
            PHI[d], GAMMA[d,] = E_step_Smoothing(alpha0, LAMBDA0, docs[d],PHI[d],GAMMA[d,],max_iter,tol_estep)
            LAMBDA += PHI[d].T @ docs[d]
        LAMBDA+=eta
        
        #update alpha
                        
        z = M * polygamma(1, sum(alpha0))
        h=-M*polygamma(1,alpha0)
        g=M*(digamma(sum(alpha0))-digamma(alpha0))+(digamma(GAMMA)-digamma(GAMMA.sum(axis=1))[:,None]).sum(axis=0)
        c = (sum(g / h)) / (1/z + sum(1/h))
        alpha = alpha0 - (g - c)/h
        
        z1 = k * polygamma(1, sum(eta0))
        h1=-k*polygamma(1,eta0)
        g1=k*(digamma(sum(eta0))-digamma(eta0))+(digamma(LAMBDA)-digamma(LAMBDA.sum(axis=1))[:,None]).sum(axis=0)
        c1 = (sum(g1 / h1)) / (1/z1 + sum(1/h1))
        eta = eta0 - (g1 - c1)/h1
        
        alpha_dis = np.mean((alpha - alpha0) ** 2)
        eta_dis = np.mean((eta - eta0) ** 2)
        lambda_dis = np.mean((LAMBDA - LAMBDA0) ** 2)
        
        alpha0 = alpha
        eta0 = eta
        LAMBDA0=LAMBDA
        
        if((alpha_dis <= tol) and (eta_dis <= tol) and (lambda_dis <= tol)):
            break
        
        
    return alpha, eta