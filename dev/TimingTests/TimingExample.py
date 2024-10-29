import numpy as np
#import pandas as pd
import time as time
import cupy as cp
from numba import cuda
from cupyx.scipy.sparse import csr_matrix


N_prot=int(20e3);
N_flus=int(150e3);
N_sparsity=int( (30/124)*N_flus);
N_flus_per_prot_avg=50; #As an average of how many flus per prot we have.
sparse_flag=False;

cp.random.seed(0)
n_flus_per_prot=cp.floor(cp.random.normal(N_flus_per_prot_avg,2,size=N_prot)).astype(int);
flus_per_prot=[cp.random.choice(N_prot,int(n_flus_per_prot[i])) for i in range(N_prot)];

#flus_mask=cp.zeros((N_prot,N_flus), dtype=bool)
#p_flu_exp_per_prot=cp.zeros((N_prot,N_flus))
#for i in range(N_prot):
#    flus_mask[i,flus_per_prot[i,:]]=True;
#    p_flu_exp_per_prot[i,flus_per_prot[i,:]]=1/len(flus_per_prot[i,:]);

p_rho_init=cp.ones((N_prot,))/N_prot;

def create_random_sparse_mat(N_sparsity,N_flus,N_samples):
    data=cp.random.normal(size=(N_samples,N_sparsity)).flatten();
    col_idxs=cp.concatenate([cp.random.choice(N_flus,N_sparsity,replace=False) for i in range(N_samples)])
    row_idxs=cp.repeat(cp.arange(N_samples),int(N_sparsity))
    S=csr_matrix((data, (row_idxs, col_idxs)))
    Perr=cp.ones((N_samples,))*0.5;
    return S,Perr


def calc_P_X_given_rho_norm(p_X_given_flu,N_prot,flus_per_prot,Perr=None):
    P_X_given_rho = cp.zeros(shape=(cp.shape(p_X_given_flu)[0], N_prot))  # Matrix where we store P(X|Rho)
    for prot_iz in range(N_prot):
        flu_iz_interest = flus_per_prot[prot_iz]
        # Indexes where P(flu_exp|Prot)!=0
        P_X_given_prot = p_X_given_flu[:, flu_iz_interest] @ cp.reshape(
            flu_iz_interest, (-1, 1)
        )  # P(X|P)= sum_{f} (P(X|f) P(f|Prot))
        
        if sparse_flag:
            P_X_given_rho[:, prot_iz] = P_X_given_prot.flatten() + Perr
        else:
            P_X_given_rho[:, prot_iz] = P_X_given_prot.flatten()
        
    # Do I have to normalize P_X_given_rho?
    return P_X_given_rho

#@cuda.jit
def calc_EM_step(P_X_given_rho_norm, p_rho_init, n_epochs=20):
    p_rho = cp.copy(p_rho_init)
    p_rho_given_X_unnorm = P_X_given_rho_norm * p_rho
    p_rho_given_X = cp.transpose(
        p_rho_given_X_unnorm.T / cp.sum(p_rho_given_X_unnorm, axis=1)
    )
    p_rho_unnorm = cp.sum(p_rho_given_X, axis=0)
    p_rho = p_rho_unnorm / p_rho_unnorm.sum()
    return p_rho

n_samples_try=[500,1000,5000];
for n_samples in n_samples_try:
    if sparse_flag:
        S,Perr=create_random_sparse_mat(N_sparsity,N_flus,int(n_samples))
    else:
        P_x_given_f_e=cp.random.normal(size=(n_samples,N_flus));
    #print(cp.shape(S))
    t_start=time.time();
    if sparse_flag:
        P_X_given_rho_norm=calc_P_X_given_rho_norm(S,N_prot,flus_per_prot, Perr=Perr)
    else:
        P_X_given_rho_norm=calc_P_X_given_rho_norm(P_x_given_f_e,N_prot,flus_per_prot)
    t_inter=time.time();
    p_rho_init_upd=calc_EM_step(P_X_given_rho_norm, p_rho_init)
    t_finish=time.time();
    t_per_read_per_it=(t_finish-t_start)/n_samples;
    t_per_read_per_it
    n_epochs=30;
    t_per_read=t_per_read_per_it*n_epochs
    print("Hours computing for 30 epochs and 1M samples with batches of "+ str(n_samples)+" reads: " + str(t_per_read*1e6/3600))
    perc_first=(t_inter-t_start)/(t_finish-t_start);
    perc_sec=(t_finish-t_inter)/(t_finish-t_start);
    print("Perc time in first function: " +"{:.2f}".format(perc_first*100));
