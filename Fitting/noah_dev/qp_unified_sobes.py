#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:27:18 2023

@author: sobesv
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2pdf
from scipy.stats.distributions import chi2
from cvxopt import matrix, solvers, lapack, blas, mul, div

def getD(xi,res_par_avg):    
    return res_par_avg['<D>' ]*2*np.sqrt(np.log(1/(1-xi))/np.pi)

def sample_Gn(sample_size,res_par_avg):
    return res_par_avg['<Gn>']*np.random.chisquare(df=res_par_avg['n_dof'],size=sample_size)/res_par_avg['n_dof']   

def sample_Gg(sample_size,res_par_avg):
    return res_par_avg['<Gg>']*np.random.chisquare(df=res_par_avg['g_dof'],size=sample_size)/res_par_avg['g_dof'] 

def get_feature(E,E_res,Gt,Gn=1):
    return Gt*Gn/((E-E_res)**2 + Gt**2)

def make_res_par_avg(D_avg, Gn_avg, n_dof, Gg_avg, g_dof, print):
    
    res_par_avg = {'<D>'  :D_avg,'<Gn>' :Gn_avg,'n_dof':n_dof,'<Gg>' :Gg_avg,'g_dof':g_dof}

    res_par_avg['D01']  = getD(0.01,res_par_avg)
    res_par_avg['D99']  = getD(0.99,res_par_avg)
    res_par_avg['Gn01'] = res_par_avg['<Gn>']*chi2.ppf(0.01, df=res_par_avg['n_dof'])/res_par_avg['n_dof']
    res_par_avg['Gn99'] = res_par_avg['<Gn>']*chi2.ppf(0.99, df=res_par_avg['n_dof'])/res_par_avg['n_dof']
    res_par_avg['Gg01'] = res_par_avg['<Gg>']*chi2.ppf(0.01, df=res_par_avg['g_dof'])/res_par_avg['g_dof']
    res_par_avg['Gg99'] = res_par_avg['<Gg>']*chi2.ppf(0.99, df=res_par_avg['g_dof'])/res_par_avg['g_dof']
    res_par_avg['Gt01'] = res_par_avg['Gn01'] + res_par_avg['Gg01']
    res_par_avg['Gt99'] = res_par_avg['Gn99'] + res_par_avg['Gg99']

    if print:
        print('D99  =',res_par_avg['D99'])
        print('Gn01 =',res_par_avg['Gn01'])
        print('Gn99 =',res_par_avg['Gn99'])
        print('Gg01 =',res_par_avg['Gg01'])
        print('Gg99 =',res_par_avg['Gg99'])
        print('Gt01 =',res_par_avg['Gt01'])
        print('Gt99 =',res_par_avg['Gt99'])

    return res_par_avg

def sample_res_par(res_par_avg,E_max):
        
    Er = -np.random.rand(1)*getD(np.random.rand(1),res_par_avg)
    while True:
        Er = np.append(Er,Er[-1] + getD(np.random.rand(1),res_par_avg))
        if Er[-1] > E_max:
            break
    Er = np.extract(            0 < Er, Er)
    Er = np.extract(Er < E_max        , Er)
    Gn = sample_Gn(Er.shape,res_par_avg) 
    Gg = sample_Gg(Er.shape,res_par_avg)
    Gt = Gn + Gg
    
    return Er, Gn, Gt

def make_sigma(E, Er, Gn, Gt):          
    sigma = np.zeros((len(E),1))       
    for iEr in range(len(Er_true)):    
        sigma += np.reshape((get_feature(E,Er[iEr],Gt[iEr],Gn[iEr])), (len(E), 1))
    return sigma

def make_data(sigma, unc):
    data = sigma + unc*np.random.randn(len(sigma),1)
    return data

def build_feature_vectors(E, option, num_Er, num_Gt):
    
    if option < 2:
        num_Gn = 1
    else:
        num_Gn = num_Gt
                
    Er = np.linspace(                  0,              max(E), num_Er)
    Gt = np.logspace(np.log10(res_par_avg['Gt01']), np.log10(res_par_avg['Gt99']), num_Gt)    
    Gn = np.logspace(np.log10(res_par_avg['Gn01']), np.log10(res_par_avg['Gn99']), num_Gn)
    
    return Er, Gt, Gn

def refine_feature_vectors(Er, x, thr, option, Gt):            
    
    x_bin = x > thr['ub']  
    if option < 2:
        x_bin = x_bin.reshape((len(Er),len(Gt)        ))  
        x_bin = np.any(x_bin,axis=1)
    else:
        x_bin = x_bin.reshape((len(Er),len(Gt),len(Gt)))
        x_bin = np.any(x_bin,axis=(1,2))
        
    Er = Er[x_bin] 
    
    deltaE = min(np.diff(Er))/3 # divide by 3 such that neighboring resonances won't split into a repeating resonance
    Er_new = np.zeros(2*len(Er))
    
    for iEr in range(len(Er)):
        Er_new[2*iEr  ] = Er[iEr] - deltaE
        Er_new[2*iEr+1] = Er[iEr] + deltaE
        
    if option < 2:
        num_Gt = int(        len(x)/len(Er_new) ) + 1 
        num_Gn = 1
    else:
        num_Gt = int(np.sqrt(len(x)/len(Er_new))) + 1 
        num_Gn = num_Gt
                        
    Gt = np.logspace(np.log10(res_par_avg['Gt01']), np.log10(res_par_avg['Gt99']), num_Gt)    
    Gn = np.logspace(np.log10(res_par_avg['Gn01']), np.log10(res_par_avg['Gn99']), num_Gn)
            
    return Er_new, Gt, Gn

def collapse_Er(Er,x,res_par_avg):    

    x_bin = x > thr['ub']  
    if option < 2:
        x_bin = x_bin.reshape((len(Er),len(Gt)        ))
        x_bin = np.any(x_bin,axis=1)
    else:
        x_bin = x_bin.reshape((len(Er),len(Gt),len(Gt)))
        x_bin = np.any(x_bin,axis=(1,2))
        
    Er = Er[ x_bin]        
    diff_Er = np.diff(Er)
    
    Er_lb = Er - res_par_avg['D01']/2
    Er_ub = Er + res_par_avg['D01']/2    

    index = np.argwhere(diff_Er < res_par_avg['D01'])
    for iEr in np.flipud(index):       
        Er_lb = np.delete(Er_lb, iEr+1        )
        Er_ub = np.delete(Er_ub, iEr          ) 
    
    return Er_lb, Er_ub

def collapse_feature_bank(Er,A,x,option,res_par_avg):    

    if option < 2:
        inc = len(Gt)        
    else:
        inc = len(Gt)*len(Gn)
    
    for iEr in range(len(Er)):
        index = slice(inc*iEr,inc*(iEr+1))
        A[:,iEr] = (A[:,index].dot(x[index])).flatten()
    A = A[:,slice(len(Er))]
        
    x_bin = x > thr['ub']  
    if option < 2:
        x_bin = x_bin.reshape((len(Er),len(Gt)        ))
        x_bin = np.any(x_bin,axis=1)
    else:
        x_bin = x_bin.reshape((len(Er),len(Gt),len(Gt)))
        x_bin = np.any(x_bin,axis=(1,2))
        
    Er = Er[  x_bin]
    A  =  A[:,x_bin]
      
    diff_Er = np.diff(Er)    
    index = np.argwhere(diff_Er < res_par_avg['D01'])
    for iEr in np.flipud(index): 
        A[:,iEr] = A[:,iEr] + A[:,iEr+1]
        A = np.delete(A, iEr+1, axis=1)  
    
    return A
    
def build_feature_bank(E, res_par_avg, option, Er_true, Gt_true, Gn_true, Er, Gt, Gn):

    num_Er = len(Er)
    num_Gt = len(Gt)
    num_Gn = len(Gn)    

    num_F = num_Er*num_Gt*num_Gn
    
    Er_index = np.zeros((len(Er_true),1))
    Gt_index = np.zeros((len(Er_true),1))
    Gn_index = np.zeros((len(Er_true),1))
    for i in range(len(Er_true)):
        Er_index[i] = np.argmin(np.abs(Er_true[i] - Er))
        Gt_index[i] = np.argmin(np.abs(Gt_true[i] - Gt))
        Gn_index[i] = np.argmin(np.abs(Gn_true[i] - Gn))
    x_true = np.zeros((num_F,1))
    
    A = np.zeros((len(E),num_F))    
    col = 0;    
    for iEr in range(num_Er):
        for iGt in range(num_Gt): 
            for iGn in range(num_Gn):                
                if   option == 0:
                    A[:,col] = get_feature(E, Er[iEr], Gt[iGt])
                elif option == 1:
                    A[:,col] = get_feature(E, Er[iEr], Gt[iGt])*Gt[iGt]
                elif option == 2:                    
                    A[:,col] = get_feature(E, Er[iEr], Gt[iGt], Gn[iGn])
                for i in range(len(Er_true)):
                    if Er_index[i] == iEr and Gt_index[i] == iGt and Gn_index[i] == iGn:
                        if   option == 0:
                            x_true[col] = Gn_true[i]
                        elif option == 1:
                            x_true[col] = Gn_true[i]/Gt_true[i]
                        elif option == 2:
                            x_true[col] = 1
                col += 1              

    # 0 < Gn and Gn < 1 or Gn99
    C = np.concatenate((-np.eye(num_F),np.eye(num_F)))
    if option == 0:
        d = np.concatenate((np.zeros((num_F,1)),res_par_avg['Gn99']*np.ones((num_F,1))))  
    else:
        d = np.concatenate((np.zeros((num_F,1)),                    np.ones((num_F,1))))       
    
    if False:
        if option < 2:
            inc = num_Gt        
        else:
            inc = num_Gt*num_Gn
            
        # there can only be one resonance in every D01     
        D01_step = int(np.ceil(len(Er)*res_par_avg['D01']/max(E)))
        step = inc*D01_step
        C_row = np.ones((1,step))
        num_rows = int((num_F-step)/inc+1)
        C_sub = np.zeros((num_rows,num_F))  
        for i_row in range(0,num_rows):
            cols = slice(i_row*inc,i_row*inc+step)
            C_sub[i_row,cols] = C_row                     
        C = np.concatenate((C,C_sub))
        
        if option == 0:
            d = np.concatenate((d,res_par_avg['Gn99']*np.ones((num_rows,1))))
        else:
            d = np.concatenate((d,                    np.ones((num_rows,1))))
          
        # make RPCM
        if False: # option == 2:        
            p = np.zeros(step)
            col = 0
            for iEr in range(D01_step):
                for iGt in range(num_Gt): 
                    for iGn in range(num_Gn):                                   
                        x = Gn[iGn]/res_par_avg['<Gn>']*res_par_avg['n_dof']
                        p[col] = chi2.pdf(x, df=res_par_avg['n_dof'])
                                        
                        # x = (Gt[iGt]-Gn[iGn])/res_par_avg['<Gg>']*res_par_avg['g_dof']
                        # assert(x > 0)
                        # p[col] *= chi2.pdf(x, df=res_par_avg['g_dof'])               
                        
                        col += 1                
            p = p/sum(p)
            # p is also the mean
            mu = p
            
            RPCM_sub = np.zeros((inc,step))
            for i in range(inc):
                for j in range(i,step):
                    for k in range(step):
                        RPCM_sub[i,j] += (int(i==k)-mu[i])*(int(j==k)-mu[j])*p[k] 
            
            # make bigger for easier assembly, then cut down
            RPCM = np.zeros((num_F,num_F+step-inc))
            i_start = 0
            for iEr in range(len(Er)):
                rows = slice(i_start,i_start+inc )
                cols = slice(i_start,i_start+step)
                RPCM[rows,cols] = RPCM_sub
                i_start += inc
                
            RPCM = RPCM[slice(num_F),slice(num_F)]  
            RPCM = RPCM + RPCM.T - np.diag(np.diag(RPCM))
            # A = np.concatenate((A,RPCM))  
          
        # there must be a resonance in every D99
        if max(E)/res_par_avg['D99'] > 1:
            D99_step = int(np.ceil(len(Er)*res_par_avg['D99']/max(E)))
            step = inc*D99_step
            C_row = np.ones((1,step))
            num_rows = int((num_F-step)/inc+1)
            C_sub = np.zeros((num_rows,num_F))  
            for i_row in range(0,num_rows):
                cols = slice(i_row*inc,i_row*inc+step)
                C_sub[i_row,cols] = C_row
            C = np.concatenate((C,-C_sub))
            
            if option == 0:                
                d = np.concatenate((d,-res_par_avg['Gn01']*np.ones((num_rows,1))))
            else:
                d = np.concatenate((d,                    -np.ones((num_rows,1))))
    
    # lasso constraint
    C =  np.concatenate((C,np.ones( (1,num_F))))
    d =  np.concatenate((d,np.zeros((1,1    ))))
  
    return A, C, d, Er, Gt, Gn, x_true

def make_qp_input(A,data,C,d):
    P = matrix( np.transpose(A).dot(A))
    q = matrix(-np.transpose(A).dot(data))
    G = matrix(C)
    h = matrix(d)
    return P, q, G, h

def qp_solve(P,q,G,h,theta=None,x0=None):        
    
    if theta is not None:
        h[-1] = theta
    
    if x0 is not None:
        x0 = {'x':matrix(x0)}        
    
    prob = solvers.qp(P, q, G, h, initvals = x0)
    x = np.array(prob['x'])                  
    
    return x  

def qcl1(A, b):
    """
    Returns the solution u, z of

        (primal)  minimize    || u ||_1
                  subject to  || A * u - b ||_2  <= 1

        (dual)    maximize    b^T z - ||z||_2
                  subject to  || A'*z ||_inf <= 1.

    Exploits structure, assuming A is m by n with m >= n.
    """

    m, n = A.size

    # Solve equivalent cone LP with variables x = [u; v].
    #
    #     minimize    [0; 1]' * x
    #     subject to  [ I  -I ] * x <=  [  0 ]   (componentwise)
    #                 [-I  -I ] * x <=  [  0 ]   (componentwise)
    #                 [ 0   0 ] * x <=  [  1 ]   (SOC)
    #                 [-A   0 ]         [ -b ]
    #
    #     maximize    -t + b' * w
    #     subject to  z1 - z2 = A'*w
    #                 z1 + z2 = 1
    #                 z1 >= 0,  z2 >=0,  ||w||_2 <= t.

    c = matrix(n*[0.0] + n*[1.0])
    h = matrix( 0.0, (2*n + m + 1, 1))
    h[2*n] = 1.0
    h[2*n+1:] = -b

    def G(x, y, alpha = 1.0, beta = 0.0, trans = 'N'):
        y *= beta
        if trans=='N':
            # y += alpha * G * x
            y[:n] += alpha * (x[:n] - x[n:2*n])
            y[n:2*n] += alpha * (-x[:n] - x[n:2*n])
            y[2*n+1:] -= alpha * A*x[:n]

        else:
            # y += alpha * G'*x
            y[:n] += alpha * (x[:n] - x[n:2*n] - A.T * x[-m:])
            y[n:] -= alpha * (x[:n] + x[n:2*n])


    def Fkkt(W):
        """
        Returns a function f(x, y, z) that solves

            [ 0   G'   ] [ x ] = [ bx ]
            [ G  -W'*W ] [ z ]   [ bz ].
        """

        # First factor
        #
        #     S = G' * W**-1 * W**-T * G
        #       = [0; -A]' * W3^-2 * [0; -A] + 4 * (W1**2 + W2**2)**-1
        #
        # where
        #
        #     W1 = diag(d1) with d1 = W['d'][:n] = 1 ./ W['di'][:n]
        #     W2 = diag(d2) with d2 = W['d'][n:] = 1 ./ W['di'][n:]
        #     W3 = beta * (2*v*v' - J),  W3^-1 = 1/beta * (2*J*v*v'*J - J)
        #        with beta = W['beta'][0], v = W['v'][0], J = [1, 0; 0, -I].

        # As = W3^-1 * [ 0 ; -A ] = 1/beta * ( 2*J*v * v' - I ) * [0; A]
        beta, v = W['beta'][0], W['v'][0]
        As = 2 * v * (v[1:].T * A)
        As[1:,:] *= -1.0
        As[1:,:] -= A
        As /= beta

        # S = As'*As + 4 * (W1**2 + W2**2)**-1
        S = As.T * As
        d1, d2 = W['d'][:n], W['d'][n:]
        d = 4.0 * (d1**2 + d2**2)**-1
        S[::n+1] += d
        lapack.potrf(S)

        def f(x, y, z):

            # z := - W**-T * z
            z[:n] = -div( z[:n], d1 )
            z[n:2*n] = -div( z[n:2*n], d2 )
            z[2*n:] -= 2.0*v*( v[0]*z[2*n] - blas.dot(v[1:], z[2*n+1:]) )
            z[2*n+1:] *= -1.0
            z[2*n:] /= beta

            # x := x - G' * W**-1 * z
            x[:n] -= div(z[:n], d1) - div(z[n:2*n], d2) + As.T * z[-(m+1):]
            x[n:] += div(z[:n], d1) + div(z[n:2*n], d2)

            # Solve for x[:n]:
            #
            #    S*x[:n] = x[:n] - (W1**2 - W2**2)(W1**2 + W2**2)^-1 * x[n:]

            x[:n] -= mul( div(d1**2 - d2**2, d1**2 + d2**2), x[n:])
            lapack.potrs(S, x)

            # Solve for x[n:]:
            #
            #    (d1**-2 + d2**-2) * x[n:] = x[n:] + (d1**-2 - d2**-2)*x[:n]

            x[n:] += mul( d1**-2 - d2**-2, x[:n])
            x[n:] = div( x[n:], d1**-2 + d2**-2)

            # z := z + W^-T * G*x
            z[:n] += div( x[:n] - x[n:2*n], d1)
            z[n:2*n] += div( -x[:n] - x[n:2*n], d2)
            z[2*n:] += As*x[:n]

        return f

    dims = {'l': 2*n, 'q': [m+1], 's': []}
    sol = solvers.conelp(c, G, h, dims, kktsolver = Fkkt)
    if sol['status'] == 'optimal':
        return sol['x'][:n],  sol['z'][-m:]
    else:
        return None, None

def split_panel(theta, x, qp_info, thr):
    
    # theta = [0,thr.theta];
    # x = [x(:,1),x(:,I)];
    # [theta,x] = split_panel(theta,x,qp,thr);
    # subplot(2,1,2)
    # plot(theta,sum(x > thr.ub),'rs')
    
    def add_theta(theta, x, theta_int, x_int):
        theta.append(theta_int)
        x.append(x_int)
        theta, indices = np.unique(theta, return_index=True)
        x = x[:, indices]
        return theta, x

    def get_dx(H, x, thr):
        index = x.flatten() > thr['ub']

        Hr = np.block([
            [H[index,index]             , np.ones((sum(index), 1))],
            [np.ones((1, sum(index))), 0                          ]
        ])

        db = np.block([
            [np.zeros((sum(index), 1))],
            [1]
        ])

        dx = np.linalg.lstsq(Hr, db)
        dx = dx[0:-1]
        return dx, index

    def predict_boundary(H, theta, x, theta_int, thr):
        dx, index = get_dx(H, x, thr)
        x_int = x[index] + (theta_int - theta)*dx
        return x_int, index

    def predict_intersect(H, theta, x, thr, direction):
        dx, index = get_dx(H, x, thr)
        dx[direction * dx < 0] = 0
        step = np.min(np.abs((x[index] - thr['lb']) / dx))
        theta_int = theta - direction * step
        if theta_int < 0 or theta_int > thr['theta_max']:
            theta_int = None
        else:
            x_int = np.zeros(x.shape)
            x_int[index] = x[index] + (theta_int - theta)*dx
        return theta_int, x_int
    
    x_left, index = predict_boundary(qp_info['H'], theta[0], x[:,0], theta[1], thr)
    if any(abs(x_left - x[1]) > thr['ub']): # does not predict left boundary
        direction = -1
        theta_int, x_int = predict_intersect(qp_info['H'], theta[0], x[:,0], thr, direction)
        if theta_int is not None and theta_int < theta[1]: # intercept is internal 
            # split           
            x_int = qp_solve(qp_info,theta_int,x_int) 
            theta_int, x_int = split_panel(theta_int.append(theta[1]),x_int.append(x[:,1]), qp_info, thr)
            theta, x = add_theta(theta, x, theta_int, x_int)
            theta_int, x_int = split_panel(theta[0].append(theta_int),x[:,0].append(x_int), qp_info, thr)
            theta, x = add_theta(theta, x, theta_int, x_int)
            return theta, x
       
    # work on the right boundary        
    x_right, index_right = predict_boundary(qp_info['H'], theta[1], x[:,1], theta[0], thr)
    index_left = x[:,0].flatten() > thr['ub']
    if any(abs(x[index_right,0] - x_right) > thr['ub']) or np.sum(np.logical_xor(index_left, index_right)) > 1:
        # right derviative does not predict left boundary
        direction = 1
        theta_int, x_int = predict_intersect(qp_info['H'], theta[1], x[:,1], thr, direction)
        if theta[0] >= theta_int: # if true, bisect else, intercept is internal, split  
            theta_int = np.mean(theta)
            x_int = np.mean(x,axis=1)
        x_int = qp_solve(qp_info,theta_int,x_int)
        theta_int, x_int = split_panel([theta_int, theta[1]], [x_int, x[:, 1]], qp_info, thr)
        theta, x = add_theta(theta, x, theta_int, x_int)
        theta_int, x_int = split_panel([theta[0], theta_int], [x[:, 0], x_int], qp_info, thr)
        theta, x = add_theta(theta, x, theta_int, x_int)
        return theta, x
    
    plt.subplot(2, 1, 1)
    x_plot = np.linspace(theta[0], theta[1])
    y_plot = np.interp(x_plot, theta, x.T)
    plt.plot(x_plot, y_plot, 'r')
    plt.axvline(theta[0])
    
    plt.subplot(2, 1, 2)
    plt.axvline(theta[0])
    plt.pause(1e-2)

def plot(A, x, data, unc, Er, Gt, Gn, Er_true, Gn_true, Gt_true, res_par_avg, thr, option):    
    
    plt.subplot(2,2,1)                
    plt.plot(E,data,'.')
    plt.plot(E,A.dot(x))
    plt.xlim(0,max(E))
    plt.title(str(np.linalg.norm((A.dot(x)-data)/unc)**2/len(E)))  
    
    x_bin = x > thr['ub']  
    if option < 2:
        x_bin = x_bin.reshape((len(Er),len(Gt)        ))
        num_Er = np.sum(x_bin,axis=(1  ))
    else:
        x_bin = x_bin.reshape((len(Er),len(Gt),len(Gn)))
        num_Er = np.sum(x_bin,axis=(1,2))
                                   
    plt.subplot(2,2,3)
    for iEr in range(len(Er_true)):
        plt.axvline(Er_true[iEr],color='red',alpha=0.2)        
    plt.plot(Er,num_Er,'.')    
    plt.ylabel('Num Res')
    plt.axhline(1,alpha=0.1)
    plt.ylim(0)  
    plt.xlim(0,max(E))
    Er_lb, Er_ub = collapse_Er(Er,x,res_par_avg)
    for iEr in range(len(Er_lb)):
        plt.fill_between(np.linspace(Er_lb[iEr],Er_ub[iEr]), 0, max(num_Er), alpha=0.1, color='green')
        
    plt.subplot(2,2,2)
    plt.semilogy(Er_true,Gn_true,'rs')
    plt.ylabel('Gn')
    plt.ylim(res_par_avg['Gn01'],res_par_avg['Gn99'])
    plt.xlim(0,max(E))
    if   option == 0:        
        plt.semilogy(Er,x.reshape((len(Er),len(Gt)))                 ,'.')
    elif option == 1:
        plt.semilogy(Er,x.reshape((len(Er),len(Gt))).dot(np.diag(Gt)),'.') 
    elif option == 2:
        for iEr in range(len(Er)):
            for iGt in range(len(Gt)):
                for iGn in range(len(Gn)):
                    if x_bin[iEr,iGt,iGn]:
                        plt.semilogy(Er[iEr],Gn[iGn],'.')                        
        
    plt.subplot(2,2,4)
    plt.semilogy(Er_true,Gt_true,'rs')   
    plt.ylabel('Gt')
    plt.ylim(res_par_avg['Gt01'],res_par_avg['Gt99'])
    plt.xlim(0,max(E))           
    for iEr in range(len(Er)):
        for iGt in range(len(Gt)):
            for iGn in range(len(Gn)):
                if option < 2:
                    if x_bin[iEr,iGt]:
                        plt.semilogy(Er[iEr],Gt[iGt],'.')
                else:
                    if x_bin[iEr,iGt,iGn]:
                        plt.semilogy(Er[iEr],Gt[iGt],'.')                        
   
    plt.show
    plt.pause(1e-2)

############

"""
Fix Wigner constraint for dynamic feature bank updating
KKT sparsty
"""

solvers.options['show_progress'] = False
solvers.options['abstol'] = 1e-7
solvers.options['reltol'] = 1e-6
np.random.seed(0)
max_iter = int(1e1)

# option
# 0: w = Gn
# 1: w = Gn/Gt
# 2: w = binary
option = 0

unc = 0.01
chi2_step = 0.05

res_par_avg = make_res_par_avg(D_avg = 1, Gn_avg = 1e-2, n_dof = 1, Gg_avg = 1e-2, g_dof = 1e2, print = False)

if option == 0:
    thr = {'ub':res_par_avg['Gn01'],'lb':res_par_avg['Gn01']/10}
else:
    thr = {'ub':1e-5,'lb':1e-6}
thr['chi2_tol'] = 1e-3

E = np.transpose(np.linspace(0, res_par_avg['D99']*2, num = int(5e2)))

# sample true resonance parameter ladders until you get one with 3+ resonances
while True:
    Er_true, Gn_true, Gt_true = sample_res_par(res_par_avg,max(E))
    if len(Er_true) > 2:
        break
# calculate true cross section and make syndat
sigma = make_sigma(E,Er_true, Gn_true, Gt_true)
data = make_data(sigma, unc)

### Initial solve 
chi2_old = 0
for i in range(max_iter):
        
    if chi2_old == 0:
        # for first itteration, build a feature bank based on some grid
        Er, Gt, Gn = build_feature_vectors(E, option, num_Er = int(1e2), num_Gt = int(3))
    else:
        # all subsequent itterations refine the feature bank
        Er, Gt, Gn = refine_feature_vectors(Er, x, thr, option, Gt)

    # build actual feature bank matrices and constraints
    A, C, d, Er, Gt, Gn, x_true = build_feature_bank(E, res_par_avg, option, Er_true, Gt_true, Gn_true, Er, Gt, Gn)
    
    ### make inputs for solve min(chi2) s.t. sum(weights)< upper_lim_theta with quadratic program 
    P, q, G, h = make_qp_input(A,data,C,d)  
    # print(np.linalg.norm((sigma-data)/unc)**2/len(E))
    # plot(A, x_true, data, unc, Er, Gt, Gn, Er_true, Gn_true, Gt_true, res_par_avg, thr, option)  

    # theta option gives upper bound on sum of all weights
    if option == 0:       
        theta = len(Er_true)*res_par_avg['Gn99']
    else:
        theta = len(Er_true)
    
    # solve
    x = qp_solve(P, q, G, h, theta, x0 = None)
    plot(A, x, data, unc, Er, Gt, Gn, Er_true, Gn_true, Gt_true, res_par_avg, thr, option)  
    chi2_new = np.linalg.norm(A.dot(x)-data)

    ### solve min(|weights|) with constraint on chi2
    x1, z = qcl1(matrix(A/chi2_new), matrix(data/chi2_new))        
    print(len(Er)); print(np.count_nonzero(x1>thr['ub']))
    ### Handle failed solve
    if x1 is not None:      
        x1 = np.array(x1)                 
        plot(A, x1, data, unc, Er, Gt, Gn, Er_true, Gn_true, Gt_true, res_par_avg, thr, option)        
        x = x1        
    else:
        print('KKT Failed')
    
    ### evaluate new chi2, if improvement is greater than some threshold -> loop with refined feature bank
    if abs(chi2_new-chi2_old)/chi2_new < thr['chi2_tol']:
        break
    else:
        chi2_old = chi2_new

### After initial solve, begin to step up in chi2 constraint on min(|weights|) solve 
steps=0
while True:
    chi2_new += chi2_step
    x1, z = qcl1(matrix(A/chi2_new), matrix(data/chi2_new))        
    
    if x1 is not None:      
        x1 = np.array(x1 )         
        x = x1 
        
        Er_lb, Er_ub = collapse_Er(Er,x,res_par_avg)
        if Er_lb.size == 0:
            break

        A_new = collapse_feature_bank(Er,A,x,option,res_par_avg)    

        plt.plot(E,data,'.') 
        plt.plot(E,A_new)
        plt.title(str(('num res=',len(Er_lb),'chi2=',np.linalg.norm((A.dot(x)-data)/unc)**2/len(E))))
        for iEr in range(len(Er_true)):
            plt.axvline(Er_true[iEr],color='red'  ,alpha=0.2)    
        for iEr in range(len(Er_lb)):
            plt.fill_between(np.linspace(Er_lb[iEr],Er_ub[iEr]), 0, max(sigma), alpha=0.1, color='green')
        plt.show() 
              
    else:
        print('KKT Failed')

    steps += 1
    if steps ==10:
        break
    
   

  
