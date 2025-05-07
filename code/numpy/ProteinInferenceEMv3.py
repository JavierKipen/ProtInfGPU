import numpy as np
import pandas as pd
import copy
import time


class ProteinInferenceEMv3:
    def __init__(self, df_path_csv, p_miss = 0.07):
        self.df=pd.read_csv(df_path_csv);
        self.p_miss=p_miss;
        self.parse_df();
        self.true_P_Y=None;
        # Contains the accumulated p_I to be normalized later. (for partial EM)

    def set_true_P_Y(self, true_P_Y):
        self.true_P_Y=true_P_Y;
    # Score preprocessing:
    
    def calc_etas_oracle(self, true_iz, p_err=0):
        t_start = time.time()
        self.Etas= p_err/self.n_prot + (1 - p_err) * self.PFgI[:, true_iz].T
        self.t_eta = time.time()-t_start
        return self.Etas;
    
    def calc_etas(self, PFo_given_X, R=None):   #When sparse, we sum up the residues.
        t_start = time.time()
        self.Etas= PFo_given_X @ self.PFgI.T
        if not (R is None):
            self.Etas = self.Etas + R;
        self.t_eta = time.time()-t_start
        return self.Etas;
    
    def fit(self, n_epochs=30):
        self.err_hist=np.zeros(n_epochs);
        t_start = time.time()
        for e in range(n_epochs):
            if not (self.true_P_Y is None):
                self.err_hist[e] = self.calc_err();
            aux_unnorm = self.Etas*self.P_I_est;
            aux=(aux_unnorm.T / aux_unnorm.sum(axis=1)).T; #Normalization
            P_I_est_new_unnorm=np.sum(aux, axis=0)
            self.P_I_est = P_I_est_new_unnorm / P_I_est_new_unnorm.sum();
        self.t_fit = time.time()-t_start
        return self.err_hist
    
    def reset(self):
        self.P_I_est = copy.deepcopy(self.P_I_equidist)
    
    def calc_err(self):
        return self.MAE(self.P_I_to_P_Y(self.P_I_est),self.true_P_Y);
    
    def MAE(self, P_est, P_true):
        return np.mean(np.abs(P_est - P_true))

    def P_I_to_P_Y(self, p_I):
        p_Y_unnorm = p_I / self.flu_exp_count_per_prot;
        p_Y = p_Y_unnorm / p_Y_unnorm.sum()
        return p_Y

    def P_Y_to_P_I(self, p_Y):
        p_I_unnorm = p_Y * self.flu_exp_count_per_prot
        P_I = p_I_unnorm / p_I_unnorm.sum()
        return P_I

   
    
    ##Internal functions
    def parse_df(self):
        self.n_prot = np.max(self.df["Original Protein Id"].values)+1; 
        self.n_exp_flus = np.max(self.df["Flustring Id"].values)+1; 
        
        
        df_one_row_per_flu = self.df.groupby("Flustring Id").first().reset_index() #1 row per flustring!
        df_one_row_per_flu["Flustring"]=df_one_row_per_flu["Flustring"].apply(lambda x: "" if x is None else x) #Replaces None by an empty str.
        flu_n_dyes=df_one_row_per_flu["Flustring"].apply(lambda x: sum(c.isdigit() for c in x)).to_numpy()
        flu_exp_n_dyes=flu_n_dyes
        #flu_exp_n_dyes=flu_n_dyes[1:];
        
        self.p_flu_exp_is_obs = np.array([(1 - self.p_miss**i) for i in flu_exp_n_dyes]);
        
        self.list_flu_exp_iz_per_I=[]#list of lists containing the flu exp each prot can generate
        self.list_p_flu_exp_per_I=[]#list of lists containing the flu exp each prot can generate
        self.flu_exp_count_per_prot=np.zeros(self.n_prot)
        
        for prot_iz in range(self.n_prot):
            df_prot = self.df[self.df["Original Protein Id"] == prot_iz]  
            flu_exp_for_prot_i = df_prot["Flustring Id"].values  # here indexes of flus -1 because we dont count null flu.
            u, indices = np.unique(flu_exp_for_prot_i, return_index=True) #We see the unique values
            self.list_flu_exp_iz_per_I.append(u)# Appends the flu exps generalated by the protein i
            auxCountList = [] ##Stores the expected number of each flustring per protein 
            for curr_flu_exp in u:
                n_flu_exp = len(np.argwhere(flu_exp_for_prot_i == curr_flu_exp)[:, 0])
                auxCountList.append(n_flu_exp * self.p_flu_exp_is_obs[curr_flu_exp])
                
            self.flu_exp_count_per_prot[prot_iz] = np.sum(auxCountList) # Expected flu exp per protein
            self.list_p_flu_exp_per_I.append(np.asarray(auxCountList) / self.flu_exp_count_per_prot[prot_iz]) #Normalizes probs!
        
        self.PFgI=np.zeros((self.n_prot,self.n_exp_flus))
        for prot_idx in range(self.n_prot): #Converts the lists to the matrix of pFgI
            for list_idx,flu_exp_of_prot in enumerate(self.list_flu_exp_iz_per_I[prot_idx]):
                self.PFgI[prot_idx,flu_exp_of_prot]=self.list_p_flu_exp_per_I[prot_idx][list_idx];
                
        P_I_equidist_prot_unnorm = ((1 / self.n_prot)* np.ones(shape=(self.n_prot,))* self.flu_exp_count_per_prot)
        # To calculate the probability of rho when assuming all proteins equally distributed
        self.P_I_equidist = P_I_equidist_prot_unnorm / P_I_equidist_prot_unnorm.sum() 
        self.P_I_est = copy.deepcopy(self.P_I_equidist)
    

