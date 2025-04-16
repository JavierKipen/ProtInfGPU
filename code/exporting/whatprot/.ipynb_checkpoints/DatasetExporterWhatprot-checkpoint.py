# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:28:00 2025

@author: JK-WORK
"""
import pandas as pd
import os
import numpy as np


class DatasetExporterWhatprot():
    def __init__(self,df_path_csv,out_folder,n_cross_val=10,p_miss = 0.25):
        self.df=pd.read_csv(df_path_csv);
        self.out_folder=out_folder+"/binary";
        self.p_miss=p_miss;
        self.n_cross_val=n_cross_val;
        self.parse_df();
        aux=np.array([len(i) for i in self.list_flu_exp_iz_per_I])
        print("Each protein produces in avg " +str(np.mean(aux)) + " and there is one protein that can generate "+ str(np.max(aux)) + " different flus")
        self.gen_vars_for_binary_export();
        
    def save_dataset_common_vars(self): #Common variables that are always saved.
        common_subfolder=self.out_folder+"/Common/";
        if not os.path.exists(common_subfolder):  ##Creating Classifier subfolders!
            os.makedirs(common_subfolder)
        
        self.nFluExpForI.tofile(common_subfolder+"nFluExpForI.bin")
        self.probFluExpForI.tofile(common_subfolder+"probFluExpForI.bin")
        self.fluExpIdForI.tofile(common_subfolder+"fluExpIdForI.bin")
        self.expNFluExpGenByI.tofile(common_subfolder+"expNFluExpGenByI.bin")
        
    def export_oracle(self,n_samples_per_flu=1000,n_samples_cv=5e5):
        self.save_dataset_common_vars()
        
        if not os.path.exists(self.out_folder+"/Common/trueIds.bin"): ##Generates trueIds if they didnt exists (doesnt overwrite others)
            true_ids=np.repeat(np.arange(0, self.n_exp_flus), n_samples_per_flu);
            true_ids.astype(np.uint32).tofile(self.out_folder+"/Common/trueIds.bin")
        
        if not os.path.exists(self.out_folder+"/Common/nSparsity.bin"): ##Generates trueIds if they didnt exists (doesnt overwrite others)
            n_sparsity=np.asarray(10);
            n_sparsity.astype(np.uint32).tofile(self.out_folder+"/Common/nSparsity.bin")
        
        oracle_subfolder=self.out_folder+"/Oracle"
        if not os.path.exists(oracle_subfolder): #Creates oracle subf if it doesnt exist!
            os.makedirs(oracle_subfolder)    
        self.gen_cv_data(oracle_subfolder,n_samples=n_samples_cv);
        
    def export_classifier(self,classifier_name,n_samples_per_flu=100,n_samples_cv=5e5):
        self.save_dataset_common_vars()
        
        classifier_folder=self.out_folder+"/"+classifier_name;
        
        if not os.path.exists(classifier_folder): #Creates oracle subf if it doesnt exist!
            os.makedirs(classifier_folder)
        
       # if not os.path.exists(self.out_folder+"/Common/trueIds.bin"): ##Generates trueIds if they didnt exists (doesnt overwrite others)
        true_ids=np.repeat(np.arange(0, self.n_exp_flus), n_samples_per_flu);
        true_ids.astype(np.uint32).tofile(classifier_folder+"/Common/trueIds.bin")
        
        if not os.path.exists(classifier_folder+"/Common/nSparsity.bin"): ##Generates trueIds if they didnt exists (doesnt overwrite others)
            n_sparsity=np.asarray(1000);
            n_sparsity.astype(np.uint32).tofile(classifier_folder+"/Common/nSparsity.bin")
        
            
        self.gen_cv_data(classifier_folder,n_samples=n_samples_cv);
        
    def gen_cv_data(self,clasificator_folder,n_samples=5e5):
        cv_folder=clasificator_folder+"/CrossVal/";
        if not os.path.exists(cv_folder): #Creates cv subf if it doesnt exist!
            os.makedirs(cv_folder)  
        prot_dists=self.gen_random_prot_dists();
        
        for i in range(self.n_cross_val):
            curr_p_flu_exp=self.p_prot_to_p_flu_exp(prot_dists[i,:])
            scoreIds=self.sample_score_ids(curr_p_flu_exp,n_samples=n_samples);
            scoreIds.astype(np.uint32).tofile(cv_folder+"ScoreIds"+str(i)+".bin")
            prot_dists[i,:].astype(np.float32).tofile(cv_folder+"TrueProtDist"+str(i)+".bin")
        
        
    ##Internal functions
    def parse_df(self):
        self.n_prot = np.max(self.df["Original Protein Id"].values)+1; 
        n_flustr = np.max(self.df["Flustring Id"].values)+1; 
        self.n_exp_flus = n_flustr-1;#No +1 because of the null flustring
        
        df_one_row_per_flu = self.df.groupby("Flustring Id").first().reset_index() #1 row per flustring!
        df_one_row_per_flu["Flustring"]=df_one_row_per_flu["Flustring"].apply(lambda x: "" if x is None else x) #Replaces None by an empty str.
        flu_n_dyes=df_one_row_per_flu["Flustring"].apply(lambda x: sum(c.isdigit() for c in x)).to_numpy()
        flu_exp_n_dyes=flu_n_dyes[1:];
        
        self.p_flu_exp_is_obs = np.array([(1 - self.p_miss**i) for i in flu_exp_n_dyes]);
        
        self.list_flu_exp_iz_per_I=[]#list of lists containing the flu exp each prot can generate
        self.list_p_flu_exp_per_I=[]#list of lists containing the flu exp each prot can generate
        self.flu_exp_count_per_prot=np.zeros(self.n_prot)
        
        for prot_iz in range(self.n_prot):
          if prot_iz==1:
              print("Debug")
          df_prot = self.df[self.df["Original Protein Id"] == prot_iz]  
          flu_exp_for_prot_i = np.asarray([i - 1 for i in df_prot["Flustring Id"].values if i != 0])  # here indexes of flus -1 because we dont count null flu.
          u, indices = np.unique(flu_exp_for_prot_i, return_index=True) #We see the unique values
          self.list_flu_exp_iz_per_I.append(u)# Appends the flu exps generalated by the protein i
          auxCountList = [] ##Stores the expected number of each flustring per protein 
          for curr_flu_exp in u:
              n_flu_exp = len(np.argwhere(flu_exp_for_prot_i == curr_flu_exp)[:, 0])
              auxCountList.append(n_flu_exp * self.p_flu_exp_is_obs[curr_flu_exp])
              
          self.flu_exp_count_per_prot[prot_iz] = np.sum(auxCountList) # Expected flu exp per protein
          self.list_p_flu_exp_per_I.append(np.asarray(auxCountList) / self.flu_exp_count_per_prot[prot_iz]) #Normalizes probs!

    
    def gen_vars_for_binary_export(self):#Variables for the binary export
        self.nFluExpForI = np.asarray([len(i) for i in self.list_flu_exp_iz_per_I]).astype(np.uint32)
        self.probFluExpForI = np.concatenate(self.list_p_flu_exp_per_I).astype(np.float32)
        self.fluExpIdForI = np.concatenate(self.list_flu_exp_iz_per_I).astype(np.uint32)
        self.expNFluExpGenByI = self.flu_exp_count_per_prot.astype(np.float32);
       
        
    def gen_random_prot_dists(self):
        exponential_vars = np.random.exponential(size=(self.n_cross_val, self.n_prot))
        prot_dists = exponential_vars / exponential_vars.sum(axis=1, keepdims=True)
        return prot_dists
    
    def p_prot_to_p_flu_exp(self,p_prot): #for a given P prot, returns which is the P flu exp.
        p_I_unnorm=p_prot*self.expNFluExpGenByI;
        p_I=p_I_unnorm / p_I_unnorm.sum();
        p_flu_exp=np.zeros(self.n_exp_flus);
        for prot in range(self.n_prot):
            for index,flu_exp_id in enumerate(self.list_flu_exp_iz_per_I[prot]):
                p_flu_exp[flu_exp_id] += self.list_p_flu_exp_per_I[prot][index]*p_I[prot];
        
        p_flu_exp = p_flu_exp / p_flu_exp.sum() ##Should sum up to 1 though.
        return p_flu_exp
    
    def sample_score_ids(self,curr_p_flu_exp,n_samples=5e5):
        true_ids=np.fromfile(self.out_folder+"/Common/trueIds.bin", dtype=np.uint32);
        u_true_ids, idx_start_u_true_ids= np.unique(true_ids, return_index=True) #We generate the outputs to be contiguous with the same class
        n_samples_per_flu=idx_start_u_true_ids[1]-idx_start_u_true_ids[0];
        true_ids_w_dist = np.random.choice(self.n_exp_flus, size=int(n_samples), p=curr_p_flu_exp).astype(np.uint32)
        true_ids_w_dist.sort(); ##Sort so we know they are in order.
        unique_flus, idx_start_unique_flus = np.unique(true_ids_w_dist, return_index=True) #We generate the outputs to be contiguous with the same class
        score_ids=(-1)*np.ones(int(n_samples),dtype=np.int32)+self.n_exp_flus; ##If one value still is negative is an error 
        
        for idx,u_flu in enumerate(unique_flus):
            if idx < len(unique_flus)-1: #While its not the last u_flue
                occ_of_flu = np.arange(idx_start_unique_flus[idx],idx_start_unique_flus[idx+1])
            else:
                occ_of_flu = np.arange(idx_start_unique_flus[idx],len(unique_flus))
            true_id_start=idx_start_u_true_ids[np.argwhere(u_true_ids == u_flu)[0][0]];
            same_id_scores=np.arange(true_id_start,true_id_start+n_samples_per_flu);
            score_ids[occ_of_flu] = np.random.choice(same_id_scores, len(occ_of_flu))  # (j+1) cause it is real flu.Picks len(idxs_flu) random scores for the flu j.
        score_ids.sort() #In dataset format they have to be sorted.
        return score_ids;
        
        
if __name__ == "__main__":
    n_proteins=20642;
    path_datasets="/home/jkipen/raid_storage/ProtInfGPU/data/20642_Prot";
    exp_csv_path=path_datasets+"/binary/ProbeamBetterConfig/ExpTable.csv"
    classifier_name="ProbeamBetterConfig";
    DEW=DatasetExporterWhatprot(exp_csv_path,path_datasets,n_cross_val=10,p_miss=0.0007);
    DEW.export_classifier(classifier_name,n_samples_cv=10e6,n_samples_per_flu=2);