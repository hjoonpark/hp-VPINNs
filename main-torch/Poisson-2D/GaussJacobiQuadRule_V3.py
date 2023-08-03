# -*- coding: utf-8 -*-
"""
Gauss Quadrature Rules

Created on Fri Apr 12 15:06:19 2019
@author: Ehsan
"""

import numpy as np
import torch
from scipy.special import gamma
from scipy.special import jacobi
from scipy.special import roots_jacobi
#from scipy.special import legendre
#import matplotlib.pyplot as plt

##################################################################
# Recursive generation of the Jacobi polynomial of order n
def Jacobi(n,a,b,x):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    j = jacobi(n,a,b)(x.cpu().detach())
    if not isinstance(j, torch.FloatTensor):
        j = torch.tensor(j).float()
    return j
    
##################################################################
# Derivative of the Jacobi polynomials
def DJacobi(n,a,b,x,k: int):
    ctemp = gamma(a+b+n+1+k)/(2**k)/gamma(a+b+n+1)
    return (ctemp*Jacobi(n-k,a+k,b+k,x))

    
##################################################################
# Weight coefficients
def GaussJacobiWeights(Q: int,a,b):
    [X , W] = roots_jacobi(Q,a,b)
    return [X, W]
	


##################################################################
# Weight coefficients
def GaussLobattoJacobiWeights(Q: int,a,b):
    W = []
    X = roots_jacobi(Q-2,a+1,b+1)[0]
    if a == 0 and b==0:
        # Gauss–Legendre quadrature is a special case of Gauss–Jacobi quadrature with α = β = 0. 
        W = 2/( (Q-1)*(Q)*(Jacobi(Q-1,0,0,X)**2) )
        Wl = 2/( (Q-1)*(Q)*(Jacobi(Q-1,0,0,-1)**2) )
        Wr = 2/( (Q-1)*(Q)*(Jacobi(Q-1,0,0,1)**2) )
    else:
        W = 2**(a+b+1)*gamma(a+Q)*gamma(b+Q)/( (Q-1)*gamma(Q)*gamma(a+b+Q+1)*(Jacobi(Q-1,a,b,X)**2) )
        Wl = (b+1)*2**(a+b+1)*gamma(a+Q)*gamma(b+Q)/( (Q-1)*gamma(Q)*gamma(a+b+Q+1)*(Jacobi(Q-1,a,b,-1)**2) )
        Wr = (a+1)*2**(a+b+1)*gamma(a+Q)*gamma(b+Q)/( (Q-1)*gamma(Q)*gamma(a+b+Q+1)*(Jacobi(Q-1,a,b,1)**2) )
    W = np.append(W , Wr)
    W = np.append(Wl , W)
    X = np.append(X , 1)
    X = np.append(-1 , X)    
    return [torch.from_numpy(X), torch.from_numpy(W)]
##################################################################


    
