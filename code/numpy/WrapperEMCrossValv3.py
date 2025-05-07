from ProteinInferenceEMv3 import ProteinInferenceEMv3
import numpy as np
from scipy.sparse import csr_matrix


class WrapperEMCrossValv3:
    def __init__(self,classifier_path,n_crossval=10,oracle=False,oracle_perr=0.0,n_sparsity=None,n_epochs=30,n_samples=1e6):
        self.classifier_path=classifier_path;
        self.load_vars(); #Loads the variables from the path, including df_path_csv
        self.n_crossval=n_crossval;
        self.oracle=oracle;
        self.oracle_perr=oracle_perr;
        self.n_sparsity=n_sparsity;
        self.PIEM=ProteinInferenceEMv3(self.df_path_csv);
        self.n_epochs=n_epochs;
        self.n_samples=n_samples;
        self.PFo_given_X_db=self.PFo_given_X_db_flatten.reshape(-1,self.PIEM.n_exp_flus)
    #Main functions    
    def compute_cv(self):
        self.gen_crossval_dists();
        self.errs=np.zeros((self.n_crossval,self.n_epochs))
        self.compute_time=0;
        for i in range(self.n_crossval):
            self.PIEM.reset();
            self.PIEM.set_true_P_Y(self.P_Ys_true[i])
            self.compute_etas(i); #Computes the equivalent etas for the PIEM (depends if oracle, sparse, etc)
            self.errs[i,:]=self.PIEM.fit(n_epochs=self.n_epochs);
            self.compute_time+= (self.PIEM.t_eta + self.PIEM.t_fit); #Keeps the time used for computation
    
    
    def compute_etas(self,index_crossval):
        if(self.oracle):
            true_iz=self.true_ids_db[self.Idxs_cv[index_crossval]]
            self.PIEM.calc_etas_oracle(true_iz, p_err=self.oracle_perr);
        else:
            
            if self.n_sparsity is None:
                PFo_given_X=self.PFo_given_X_db[self.Idxs_cv[index_crossval],:]
                self.PIEM.calc_etas(PFo_given_X, R=None)
            else:
                PFo_given_X_sparse,R=self.get_sparse_mat(index_crossval);
                self.PIEM.calc_etas(PFo_given_X_sparse, R=R)
    
    
    
    def gen_crossval_dists(self): #Generates n_crossval random distributions, and the samples ids to get from the classifier. 
        exponential_vars = np.random.exponential(size=(self.n_crossval, self.PIEM.n_prot))
        self.P_Ys_true = exponential_vars / exponential_vars.sum(axis=1, keepdims=True)    
        self.Idxs_cv=[];
        for i in range(self.n_crossval):
            curr_p_flu_exp=self.p_flu_o_given_p_Y(self.P_Ys_true[i,:])
            self.Idxs_cv.append(self.sample_score_ids(curr_p_flu_exp));
        
        
    # Internal functions
    
    def sample_score_ids(self,curr_p_flu_exp):
        u_true_ids, idx_start_u_true_ids= np.unique(self.true_ids_db, return_index=True) #We generate the outputs to be contiguous with the same class
        self.n_samples_per_flu_db=idx_start_u_true_ids[1]-idx_start_u_true_ids[0];
        true_ids_w_dist = np.random.choice(self.PIEM.n_exp_flus, size=int(self.n_samples), p=curr_p_flu_exp).astype(np.uint32)
        true_ids_w_dist.sort(); ##Sort so we know they are in order.
        unique_flus, idx_start_unique_flus = np.unique(true_ids_w_dist, return_index=True) #We generate the outputs to be contiguous with the same class
        score_ids=(-1)*np.ones(int(self.n_samples),dtype=np.int32); ##If one value still is negative is an error 
        
        for idx,u_flu in enumerate(unique_flus):
            if idx < len(unique_flus)-1: #While its not the last u_flue
                occ_of_flu = np.arange(idx_start_unique_flus[idx],idx_start_unique_flus[idx+1])
            else:
                occ_of_flu = np.arange(idx_start_unique_flus[idx],len(true_ids_w_dist))
            true_id_start=idx_start_u_true_ids[np.argwhere(u_true_ids == u_flu)[0][0]];
            same_id_scores=np.arange(true_id_start,true_id_start+self.n_samples_per_flu_db);
            score_ids[occ_of_flu] = np.random.choice(same_id_scores, len(occ_of_flu))  # (j+1) cause it is real flu.Picks len(idxs_flu) random scores for the flu j.
        score_ids.sort() #In dataset format they have to be sorted.
        return score_ids;
    
    def load_vars(self):
        self.PFo_given_X_db_flatten=np.fromfile(self.classifier_path+"/Common/TopNScores.bin", dtype=np.float32);
        self.true_ids_db=np.fromfile(self.classifier_path+"/Common/trueIds.bin", dtype=np.uint32);
        self.df_path_csv=self.classifier_path + "/ExpTable.csv";
        
    def p_flu_o_given_p_Y(self,p_Y): #for a given P prot, returns which is the P flu exp.
        p_I_unnorm=p_Y*self.PIEM.flu_exp_count_per_prot;
        p_I=p_I_unnorm / p_I_unnorm.sum();
        p_flu_exp=np.zeros(self.PIEM.n_exp_flus);
        for prot in range(self.PIEM.n_prot):
            for index,flu_exp_id in enumerate(self.PIEM.list_flu_exp_iz_per_I[prot]):
                p_flu_exp[flu_exp_id] += self.PIEM.list_p_flu_exp_per_I[prot][index]*p_I[prot];
        
        p_flu_exp = p_flu_exp / p_flu_exp.sum() ##Should sum up to 1 though.
        return p_flu_exp
    
    def get_sparse_mat(self,index_crossval):
        PFo_given_X=self.PFo_given_X_db[self.Idxs_cv[index_crossval],:]
        topk_idx = np.argpartition(PFo_given_X, -self.n_sparsity, axis=1)[:, -self.n_sparsity:]
        
        data, cols, ptr = [], [], [0];
        for i, idx in enumerate(topk_idx):
            vals = PFo_given_X[i, idx]
            data.extend(vals)
            cols.extend(idx)
            ptr.append(ptr[-1] + self.n_sparsity)
    
        PFo_given_X_sparse = csr_matrix((np.array(data), np.array(cols), np.array(ptr)), shape=(self.n_reads, self.PIEM.n_exp_flus))
        R = (1.0 - np.array(PFo_given_X_sparse.sum(axis=1)).ravel())/self.PIEM.n_exp_flus;
        
        return PFo_given_X_sparse,R;
    
    
if __name__ == "__main__":
    WEMV=WrapperEMCrossValv3("C:/Users/JK-WORK/Desktop/DatasetsProtInf/5_Prot/",oracle=False)
    WEMV.compute_cv()
