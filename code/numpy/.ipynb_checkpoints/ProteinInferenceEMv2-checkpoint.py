import time

import numpy as np
import pandas as pd
#from numba import float32, int32, jit
from scipy.sparse import csr_matrix


#@jit(nopython=True, parallel=True)
def calc_P_I(
    P_X_given_I_rel, P_I_init, n_epochs=30
):  # Does the whole loop of the EM algorithm (only used for all memory present)
    p_I = np.copy(P_I_init)
    for epoch_em in range(n_epochs):  ##Expectation maximization of p_flu
        p_joint_I_X_unnorm = P_X_given_I_rel * p_I
        alpha = np.transpose(p_joint_I_X_unnorm.T / np.sum(p_joint_I_X_unnorm, axis=1))
        p_I_unnorm = np.sum(alpha, axis=0)
        p_I = p_I_unnorm / p_I_unnorm.sum()
    return p_I


#@jit(nopython=True, parallel=True)
def sum_alpha(
    P_X_given_I_rel, P_I_init
):  # Numba Optimized version of the function inside the class.
    p_joint_I_X_unnorm = P_X_given_I_rel * P_I_init
    # P(X|I)*P(I)
    alpha = (p_joint_I_X_unnorm.T / p_joint_I_X_unnorm.sum(axis=1)).T
    # This is the term inside the sum over X
    p_I_upd = np.sum(alpha, axis=0)
    return p_I_upd  # Returns the unnormalized p_I, since it will be normalized later when all the contributions of all X are summed.


class ProteinInferenceEMv2:
    def __init__(self, df_sim_path, use_numba=True):
        self.df_sim = pd.read_csv(df_sim_path)
        self.get_general_variables()
        self.calc_internal_variables()
        self.use_numba = use_numba
        self.normalize_sparse = True
        self.p_I_accumulated = 0
        # Contains the accumulated p_I to be normalized later. (for partial EM)

    # Score preprocessing:

    def oracle_p_X_given_flu_exp(
        self, true_iz, p_err=0
    ):  # Imaginary classifier that picks the true flu_exp with probability 1-p_err. Rest is uniformly distributed.
        n_rads = len(true_iz)
        p_X_given_flu_exp = np.zeros(shape=(n_rads, self.num_exp_flus))
        for i in range(n_rads):
            p_X_given_flu_exp[i, :] = p_err / (self.num_exp_flus - 1)
            p_X_given_flu_exp[i, true_iz[i] - 1] = 1 - p_err
        return p_X_given_flu_exp

    def build_sparse_oracle(self, true_iz, oracle_error_rate):
        N_rads = len(true_iz)
        p_aux = oracle_error_rate / self.num_exp_flus
        # value of the uniform normalizing prob
        p_rem = np.ones(N_rads) * p_aux
        # Remaining probability
        S_row_idxs = np.arange(N_rads)
        S_col_idxs = true_iz.flatten()
        S_data = np.ones(N_rads) * (1 - oracle_error_rate) - p_aux
        S = csr_matrix(
            (S_data, (S_row_idxs, S_col_idxs)), shape=(N_rads, self.num_exp_flus)
        )
        return S, p_rem

    def construct_p_X_given_flu_exp(
        self, top_n_flu_scores, top_n_flu_iz
    ):  # To get p_X_given_flu from the RF classifications
        p_X_given_flu_exp = np.zeros(
            np.shape(top_n_flu_scores)
        )  # Assumes RF classificator returns score for all.
        for i in range(np.shape(p_X_given_flu_exp)[0]):
            p_X_given_flu_exp[i, top_n_flu_iz[i, :]] = top_n_flu_scores[i, :]
        return p_X_given_flu_exp

    def build_sparse_matrix_N_Best(
        self, top_n_scores, top_n_flu_iz, N=10, dense_scores=True
    ):  # Dense scores says whether the given scores are already sparse or not
        # Sparses the vector so the scores P(X|fe) = S + p_rem @ ones(1,N_rad)= S + A, where S is the sparse matrix and A is the normalization of the remaining prob
        if dense_scores:
            top_n_scores = top_n_scores[:, 0:N]
            top_n_flu_iz = top_n_flu_iz[:, 0:N]
        N_rads = len(top_n_flu_iz)
        # We split the remaining prob among all flus.
        p_rem = (
            1 - np.sum(top_n_scores, axis=1)
        ) / self.num_exp_flus  # probability missing in each row div by num_exp_flus
        # A=np.reshape(p_rem, (-1, 1)) @ np.ones((1,N_rads));
        S_data = top_n_scores.flatten()  ##Variables to generate the sparse matrix.
        S_row_idxs = np.repeat(np.arange(0, N_rads), N)
        S_col_idxs = top_n_flu_iz.flatten()
        # Finally we construct the sparse matrix
        S = csr_matrix(
            (S_data, (S_row_idxs, S_col_idxs)), shape=(N_rads, self.num_exp_flus)
        )
        return S, p_rem

    def build_sparse_matrix_accumulated_th(
        self, top_n_scores, top_n_flu_iz, Th=0.7
    ):  # Dense scores says whether the given scores are already sparse or not
        # Sparses the vector so the scores P(X|f) = S + p_rem @ ones(1,N_rad)= S + A, where S is the sparse matrix and B is
        N_rads = np.shape(top_n_scores)[0]
        acc_sum = np.cumsum(top_n_scores, axis=1)
        score_cut_idxs = [
            np.argwhere(acc_sum[i, :] > Th)[0, 0] for i in range(len(acc_sum))
        ]
        # print(score_cut_idxs)
        p_rem = (1 - acc_sum[np.arange(N_rads), score_cut_idxs]) / self.num_exp_flus
        # A=np.reshape(p_rem, (-1, 1)) @ np.ones((1,N_rads));
        list_top_scores = [
            top_n_scores[i, : (score_cut_idxs[i] + 1)] for i in range(N_rads)
        ]
        list_top_scores_iz = [
            top_n_flu_iz[i, : (score_cut_idxs[i] + 1)] for i in range(N_rads)
        ]
        S_data = np.concatenate(list_top_scores)
        S_row_idxs = np.repeat(np.arange(0, N_rads), np.asarray(score_cut_idxs) + 1)
        S_col_idxs = np.concatenate(list_top_scores_iz)
        # Finally we construct the sparse matrix
        S = csr_matrix(
            (S_data, (S_row_idxs, S_col_idxs)), shape=(N_rads, self.num_exp_flus)
        )
        return S, p_rem

    def build_sparse_matrix_score_th(
        self, top_n_scores, top_n_flu_iz, Th=0.7
    ):  # Dense scores says whether the given scores are already sparse or not
        # Sparses the vector so the scores P(X|f) = S + p_rem @ ones(1,N_rad)= S + A, where S is the sparse matrix and B is
        N_rads = np.shape(top_n_scores)[0]
        score_cut_idxs = np.zeros(N_rads, dtype=int)
        p_rem = np.zeros(N_rads)
        for i in range(N_rads):
            a = np.argwhere(top_n_scores[i, :] < Th)
            score_cut_idxs[i] = 1 if np.size(a) == 0 else a[0, 0]
            # print(score_cut_idxs[i])
            p_rem[i] = (
                np.sum(top_n_scores[i, : score_cut_idxs[i]])
                if score_cut_idxs[i] != 0
                else 1
            )
            # Saves the acc prob
        # print(score_cut_idxs)
        p_rem = (1 - p_rem) / self.num_exp_flus
        # Calculates the normalizing prob.
        list_top_scores = [
            top_n_scores[i, : (score_cut_idxs[i] + 1)] for i in range(N_rads)
        ]
        list_top_scores_iz = [
            top_n_flu_iz[i, : (score_cut_idxs[i] + 1)] for i in range(N_rads)
        ]
        S_data = np.concatenate(list_top_scores)
        S_row_idxs = np.repeat(np.arange(0, N_rads), np.asarray(score_cut_idxs) + 1)
        S_col_idxs = np.concatenate(list_top_scores_iz)
        # Finally we construct the sparse matrix
        S = csr_matrix(
            (S_data, (S_row_idxs, S_col_idxs)), shape=(N_rads, self.num_exp_flus)
        )
        return S, p_rem

    # Fitting partially (Does the EM algorithm but for the given Xis, which are not the whole dataset).
    # This allows to compute big datasets that maybe can not fit in the program memory:

    def fit_p(self, p_X_given_flu_exp, p_rem=None, p_I_init=None):
        if p_I_init is None:
            p_I_init = self.P_I_equidist_prot
        self.t_start = time.time()
        if p_rem is None:  # Non sparse case
            P_X_given_I_rel = self.get_P_X_given_I_rel(p_X_given_flu_exp)
        else:  # Sparse case
            P_X_given_I_rel = self.get_P_X_given_I_rel_sparse(p_X_given_flu_exp, p_rem)
        t_inter = time.time()
        if self.use_numba:
            p_I_upd = sum_alpha(P_X_given_I_rel, p_I_init)
        else:
            p_I_upd = self.sum_alpha(P_X_given_I_rel, p_I_init)
        self.t_end = time.time()
        self.t_calc_P_X_given_rho_norm = t_inter - self.t_start
        self.t_calc_EM_steps_opt = self.t_end - t_inter
        self.p_I_accumulated += p_I_upd

    def update_P_I(self):
        p_I_new = self.p_I_accumulated / self.p_I_accumulated.sum()
        self.p_I_accumulated = 0
        return p_I_new

    def P_I_to_P_prot(self, p_I):
        p_prot_unnorm = p_I / self.flu_exp_count_per_prot
        p_prot = p_prot_unnorm / p_prot_unnorm.sum()
        return p_prot

    def P_prot_to_P_I(self, p_prot):
        p_I_unnorm = p_prot * self.flu_exp_count_per_prot
        P_I = p_I_unnorm / p_I_unnorm.sum()
        return P_I

    # Could generate a function with a callback that does the whole fitting process and returns the p_prot_est.

    # fit function that does the whole fitting process. Only works when all data can be stored in memory

    def fit(self, p_X_given_flu_exp, p_rem=None, p_I_init=None, n_epochs=30):
        # Assumes p_X_given_flu_exp is sparse when p_rem is given.
        if p_I_init is None:
            p_I_init = self.P_I_equidist_prot
        p_I = p_I_init
        self.t_start = time.time()
        if p_rem is None:  # Non sparse case
            P_X_given_I_rel = self.get_P_X_given_I_rel(p_X_given_flu_exp)
        else:  # Sparse case
            P_X_given_I_rel = self.get_P_X_given_I_rel_sparse(p_X_given_flu_exp, p_rem)
        t_inter = time.time()
        if self.use_numba:
            p_I = calc_P_I(P_X_given_I_rel, p_I, n_epochs)
        else:
            print("TBD: This function is not implemented yet")
        self.t_end = time.time()
        self.t_calc_P_X_given_rho_norm = t_inter - self.t_start
        self.t_calc_EM_steps_opt = self.t_end - t_inter
        return p_I

    # Non jit version of functions in the EM algorithm

    def get_P_X_given_I_rel(self, p_X_given_flu):
        P_X_given_I_rel = np.zeros(
            shape=(len(p_X_given_flu), self.n_prot)
        )  # Matrix where we store P(X|Rho)
        for prot_iz in range(self.n_prot):
            flu_iz_interest = self.non_zero_flu_idz_per_prot[prot_iz]
            # Indexes where P(flu_exp|Prot)!=0
            P_X_given_prot = p_X_given_flu[:, flu_iz_interest] @ np.reshape(
                self.P_flu_exp_given_prot_i[prot_iz, flu_iz_interest], (-1, 1)
            )  # P(X|P)= sum_{f} (P(X|f) P(f|Prot))
            P_X_given_I_rel[:, prot_iz] = P_X_given_prot.flatten()
        return P_X_given_I_rel

    def get_P_X_given_I_rel_sparse(self, p_X_given_flu_exp_sparse, p_rem):
        P_X_given_I_rel = np.zeros(
            shape=(np.shape(p_X_given_flu_exp_sparse)[0], self.n_prot)
        )  # Matrix where we store P(X|Rho)
        for prot_iz in range(self.n_prot):
            flu_iz_interest = self.non_zero_flu_idz_per_prot[prot_iz]
            # Indexes where P(flu_exp|Prot)!=0
            P_X_given_prot = p_X_given_flu_exp_sparse[:, flu_iz_interest] @ np.reshape(
                self.P_flu_exp_given_prot_i[prot_iz, flu_iz_interest], (-1, 1)
            )  # Computes for S, but we have to add the remaining probability due to the truncation.
            if self.normalize_sparse:
                P_X_given_I_rel[:, prot_iz] = P_X_given_prot.flatten() + p_rem
            else:
                P_X_given_I_rel[:, prot_iz] = P_X_given_prot.flatten()
        return P_X_given_I_rel

    def sum_alpha(self, P_X_given_I_rel, P_I_init):
        p_I = np.copy(P_I_init)
        p_joint_I_X_unnorm = P_X_given_I_rel * p_I
        # P(X|I)*P(I)
        alpha = p_joint_I_X_unnorm / p_joint_I_X_unnorm.sum(axis=1, keepdims=True)
        # This is the term inside the sum over X
        p_I_upd = np.sum(alpha, axis=0)
        return p_I_upd  # Returns the unnormalized p_I, since it will be normalized later when all the contributions of all X are summed.

    ### Mem usage calculation

    def get_perc_mem_usage(self, S, spar_mode="N"):
        # All usages are in bytes
        float_32_bytes = 4
        uint_16_bytes = 2
        N_rads = np.shape(S)[0]
        N_flus = np.shape(S)[1]
        no_sparsity_usage = N_rads * N_flus * float_32_bytes
        # Number of floats32 to represent the dense scores.
        if spar_mode == "N":
            method_usage = len(S.data) * (float_32_bytes + uint_16_bytes)
        elif spar_mode == "AccTh":
            method_usage = (
                len(S.data) * (float_32_bytes + uint_16_bytes) + N_rads * uint_16_bytes
            )
            # Each data point and its index + vector saying the amount of scores per row
        elif spar_mode == "Th":
            method_usage = (
                len(S.data) * (float_32_bytes + uint_16_bytes) + N_rads * uint_16_bytes
            )
            # Each data point and its index + vector saying the amount of scores per row

        return method_usage / no_sparsity_usage * 100

    ### Error metric functions

    def MAE(self, p_prot_est, p_prot_true):
        return np.mean(np.abs(p_prot_est - p_prot_true))

    def get_calc_times(self):
        return np.copy(self.t_calc_P_X_given_rho_norm), np.copy(
            self.t_calc_EM_steps_opt
        )

    ###All setup functions
    def get_general_variables(self):
        self.num_unique_flus = self.df_sim["flu_i"].max() + 1  ##Num of flus
        self.num_exp_flus = self.num_unique_flus - 1  ##Num of experimental flus.
        self.n_prot = np.max(self.df_sim["pro_i"].values)  ##Num of proteins
        ##Gets the total number of fluorophores per flustring, to obtain the correction values for experimental flustrings
        self.flu_n_dyes = np.array(
            [
                self.df_sim.loc[
                    self.df_sim.index[np.where(self.df_sim["flu_i"] == i)[0][0]],
                    "n_dyes_all_ch",
                ]
                for i in np.arange(1, self.num_unique_flus)
            ]
        )
        self.p_dud = 0.25
        # From the default Marker class... (seems pretty high).
        self.p_flu_is_obs = np.array(
            [(1 - self.p_dud**i) for i in self.flu_n_dyes]
        )  # Prob that each flu will be observable
        self.correction_p_flu = np.array(
            [(1 / i) for i in self.p_flu_is_obs]
        )  ##Inverse of the previous one is used to correct other vectors

    def calc_internal_variables(self):
        # Obtains P(f^e|Prot), P(\rho) and the counts of f^e per prot.
        P_flu_given_prot_i = np.zeros((self.n_prot, self.num_unique_flus - 1))
        # Rows show P(f|Prot_i)
        flu_count_per_prot = np.zeros(
            self.n_prot
        )  # Counts how many flustrings produces a protein.
        self.flu_exp_count_per_prot = np.zeros(
            self.n_prot
        )  # Counts how many experimental flustrings produces a protein.
        self.non_zero_flu_idz_per_prot = []
        # List of indexes of flus that can be generated by each protein
        for prot_iz in range(self.n_prot):
            df_prot = self.df_sim[
                self.df_sim["pro_i"] == (prot_iz + 1)
            ]  # Protein index 0 is null.
            flu_iz_list_prot_i = [
                i - 1 for i in df_prot["flu_i"].values if i != 0
            ]  # here indexes of flus -1 because we dont count null flu.

            self.non_zero_flu_idz_per_prot.append(
                np.unique(flu_iz_list_prot_i)
            )  # Stores which flus can be generated by each protein
            for i in flu_iz_list_prot_i:
                # This flu can be generated by this protein
                P_flu_given_prot_i[prot_iz, i] = P_flu_given_prot_i[prot_iz, i] + 1
                ##Sums up the flus given
                self.flu_exp_count_per_prot[prot_iz] += self.p_flu_is_obs[i]
            flu_count_per_prot[prot_iz] = len(
                flu_iz_list_prot_i
            )  # Counts the flus of the protein
        P_flu_given_prot_i = np.divide(P_flu_given_prot_i.T, flu_count_per_prot).T
        P_flu_exp_given_prot_i_unnorm = P_flu_given_prot_i * self.p_flu_is_obs
        self.P_flu_exp_given_prot_i = (
            P_flu_exp_given_prot_i_unnorm.T / P_flu_exp_given_prot_i_unnorm.sum(axis=1)
        ).T

        P_I_equidist_prot_unnorm = (
            (1 / self.n_prot)
            * np.ones(shape=(self.n_prot,))
            * self.flu_exp_count_per_prot
        )
        # To calculate the probability of rho when assuming all proteins equally distributed
        self.P_I_equidist_prot = (
            P_I_equidist_prot_unnorm / P_I_equidist_prot_unnorm.sum()
        )
        # Then normalized

        # self.p_flue_exp_for_equidist_prot = (
        #    self.P_flu_exp_given_prot_i.T @ self.P_rho_equidist_prot
        # f)
