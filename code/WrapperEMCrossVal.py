import os

import ipdb
import numpy as np
import pandas as pd
from ProteinInferenceEMv2 import ProteinInferenceEMv2


class WrapperEMCrossVal:
    def __init__(
        self,
        classifier_path,
        n_sparse=None,
        n_epochs=30,
        n_datasets=10,
        Gb_max_compute=2,
        verbose=False,
        oracle_error_rate=0,
        test_local_minimum=False,
    ):  # Exp folder points towards the classifier folder of the dataset
        self.exp_folder = os.path.dirname(classifier_path)
        self.n_sparse = n_sparse
        self.df_sim = pd.read_csv(self.exp_folder + "/Common/df_info.csv")
        self.n_flu_exp = np.max(
            self.df_sim["flu_i"].values
        )  ##Num of exp flus (null flu not counted).
        self.classifier_path = classifier_path
        self.curr_score_ds = 0
        # Always start with the first dataset for the cross val
        self.only_one_dataset = not os.path.isfile(
            classifier_path + "/Common/Scores_1.npz"
        )
        # If there is only Scores_0, then there is only one dataset.
        self.n_epochs = n_epochs
        self.n_datasets = n_datasets
        self.Gb_max_compute = Gb_max_compute
        # Max Gb to compute EM step at once.
        self.n_prot = np.max(self.df_sim["pro_i"].values)
        self.load_scores()
        self.load_cross_val_data()
        self.oracle_error_rate = oracle_error_rate
        if self.n_sparse is not None:
            bytes_per_read = self.n_sparse * (4 + 2)
            # Each float32 (score value) takes 4 bytes, each int16 (score idx) takes 2 bytes.
        else:
            bytes_per_read = self.top_n_flu_iz.shape[1] * (4 + 2)
        bytes_per_read = np.max(
            [bytes_per_read, self.n_prot * (4)]
        )  # In the middle a matrix of N_reads x N_prot is created, so the biggest is the bottleneck.
        self.max_n_samples_compute = int(
            self.Gb_max_compute * (2**30) / bytes_per_read
        )
        # Each float32 takes 4 bytes.
        self.verbose = verbose
        self.test_local_minimum = test_local_minimum

    # Fitting for big datasets: only loading to memory the scores to work with.
    def cross_val_em_fit(self, oracle=False):
        if self.test_local_minimum:  # Uses the true dist as starting dist
            self.est_P_Is = self.get_cv_true_P_Is()
        else:  # If not, uses equidist protein distribution
            self.est_P_Is = [self.get_P_I_for_eqdist_prot()] * self.n_datasets
        # For this method, we load the some reads and update the model for the different cross-validations,
        # then the order of the loops is different!
        for epoch in range(self.n_epochs):  # Loop through epochs
            print("Epoch " + str(epoch))
            self.n_reads_used = 0
            # Restarts the counter of reads used
            for self.curr_score_ds in range(
                self.n_ds_scores
            ):  # Loop through the score datasets
                self.load_scores()
                # Loads some of the reads to the whole memory.
                for cross_val_i in range(
                    self.n_datasets
                ):  # For each cross validation dataset
                    self.cv_unit_em_fit(cross_val_i, oracle=oracle)
                    # Computes the partial fits for the given data and the given P_I estimates
                self.n_reads_used += self.ds_n_reads
                # We save that we have processed this amount of reads.
            # Here the whole dataset was processed, so now the probs have to be updated.
            for cross_val_i in range(self.n_datasets):
                self.est_P_Is[cross_val_i] = self.cv_PIEMs[cross_val_i].update_P_I()
                # (P_I is updated in each PIEM)
        self.est_P_prot = [
            self.cv_PIEMs[0].P_I_to_P_prot(self.est_P_Is[i])
            for i in range(self.n_datasets)
        ]  # Converts P_I to P_prot

    def cv_unit_em_fit(self, cross_val_i, oracle=False):
        curr_PIEM = self.cv_PIEMs[cross_val_i]
        read_iz_in_mem = [self.n_reads_used, self.n_reads_used + self.ds_n_reads]
        # These are the reads present in memory
        curr_id_scores = self.cv_id_scores[cross_val_i]
        curr_ids = curr_id_scores[
            np.logical_and(
                (curr_id_scores >= read_iz_in_mem[0]),
                (curr_id_scores < read_iz_in_mem[1]),
            )
        ]
        # Scores to use for the computations
        # Since curr_ids can be very large, it has to be partitioned in smaller chunks to compute the EM step.
        part_curr_ids = self.partition_idxs(self.max_n_samples_compute, curr_ids)
        for i in range(len(part_curr_ids)):
            idxs_use = part_curr_ids[i] - self.n_reads_used
            # Transforms the ids of the general dataset to the idxs of the memory
            if (
                oracle
            ):  # Uses the true ids to get perfect P(I|X) (only for debug/analysis)
                S, p_err = curr_PIEM.build_sparse_oracle(
                    self.cv_true_flu_exp_izs[cross_val_i][idxs_use],
                    self.oracle_error_rate,
                )
            else:
                top_n_flu_scores = self.curr_top_n_flu_scores[idxs_use]
                top_n_flu_iz = self.top_n_flu_iz[idxs_use]
                S, p_err = curr_PIEM.build_sparse_matrix_N_Best(
                    top_n_flu_scores, top_n_flu_iz, N=self.n_sparse
                )
                # Generates sparse representation for faster compute
            curr_PIEM.fit_p(S, p_err, p_I_init=self.est_P_Is[cross_val_i])

    ##Easiest fitting case: One dataset and all compute fits in memory:
    ##Note that this is more efficient in calculations since it calculates P(X|I) only once.
    ##When oracle is true, uses an oracle classifier with error rate self.oracle_error_rate
    def cv_fit_all_mem_load(self, oracle=False):
        if self.test_local_minimum:  # Uses the true dist as starting dist
            self.est_P_Is = self.get_cv_true_P_Is()
        else:  # If not, uses equidist protein distribution
            self.est_P_Is = [self.get_P_I_for_eqdist_prot()] * self.n_datasets
        p_rem = None
        # Information of how to complete sparse matrix. Is None when dense scores are used.
        for i in range(self.n_datasets):
            if self.verbose:
                print("Fitting dataset " + str(i))
            curr_PIEM = self.cv_PIEMs[i]
            if oracle:
                S, p_rem = curr_PIEM.build_sparse_oracle(
                    self.cv_true_flu_exp_izs[i], self.oracle_error_rate
                )
            else:
                score_iz_iz = self.cv_id_scores[i]
                # Gets the datasets pointers to scores
                top_n_flu_scores = self.curr_top_n_flu_scores[score_iz_iz]
                # Loads the ones to use for the computations
                top_n_flu_iz = self.top_n_flu_iz[score_iz_iz]
                S, p_rem = curr_PIEM.build_sparse_matrix_N_Best(
                    top_n_flu_scores, top_n_flu_iz, N=self.n_sparse
                )
                # Generates sparse representation for faster compute
            self.est_P_Is[i] = curr_PIEM.fit(S, p_rem=p_rem, n_epochs=self.n_epochs)
            # Fits the model
        self.est_P_prot = [
            self.cv_PIEMs[0].P_I_to_P_prot(self.est_P_Is[i])
            for i in range(self.n_datasets)
        ]
        if self.verbose:
            print(
                "Time of the last fit "
                + "{:.2f}".format(self.cv_PIEMs[-1].t_end - self.cv_PIEMs[-1].t_start)
                + " seconds"
            )
            # Just print the time of the last fit
        return self.est_P_prot

    ##Error measurement
    def cv_MAEs(self):  # Calculates the MAE in each cross validation run.
        MAE_results = [
            np.mean(np.abs(self.est_P_prot[i] - self.cv_true_prot_dists[i]))
            for i in range(self.n_datasets)
        ]
        return MAE_results

    def cv_random_guess_MAEs(
        self,
    ):  # Calculates the MAE when guessing without the algorithm
        est_P_prot = (
            np.ones(np.shape(self.cv_true_prot_dists[0]))
            / np.shape(self.cv_true_prot_dists[0])[0]
        )
        MAE_results = [
            np.mean(np.abs(est_P_prot - self.cv_true_prot_dists[i]))
            for i in range(self.n_datasets)
        ]
        return MAE_results

    ##Loading and setting up wrapper:
    def load_scores(self):
        data = np.load(
            self.classifier_path + "/Common/Scores_" + str(self.curr_score_ds) + ".npz"
        )
        self.n_ds_scores = len(
            [
                name
                for name in os.listdir(self.classifier_path + "/Common/")
                if name.endswith(".npz")
            ]
        )
        self.curr_top_n_flu_scores = data["top_n_flu_scores"]
        self.top_n_flu_iz = data["top_n_flu_iz"]
        self.ds_n_reads = np.shape(self.curr_top_n_flu_scores)[0]
        # Number of reads in the dataset
        if (
            self.n_sparse is not None
        ):  # If we sparsify further, its carried in this step
            self.top_n_flu_iz = self.top_n_flu_iz[:, : self.n_sparse]
            self.curr_top_n_flu_scores = self.curr_top_n_flu_scores[:, : self.n_sparse]

    def load_cross_val_data(self):
        self.cv_true_prot_dists = []
        self.cv_true_flu_exp_izs = []
        self.cv_id_scores = []
        self.cv_PIEMs = []
        for i in range(self.n_datasets):
            self.cv_PIEMs.append(
                ProteinInferenceEMv2(self.exp_folder + "/Common/df_info.csv")
            )
            data = np.load(self.classifier_path + "/CrossVal/ds_" + str(i) + ".npz")
            self.cv_id_scores.append(data["scores_iz"])
            self.cv_true_flu_exp_izs.append(data["true_flu_exp_iz"])
            self.cv_true_prot_dists.append(data["true_prot_dist"])

    ## Useful internal functions:
    def partition_idxs(self, n_samples_max, idxs_to_partition):
        n_partitions = int(np.ceil(len(idxs_to_partition) / n_samples_max))
        idxs = np.array_split(idxs_to_partition, n_partitions)
        return idxs

    def get_P_I_for_eqdist_prot(self):
        return self.cv_PIEMs[0].P_I_equidist_prot

    def get_cv_true_P_Is(self):
        return [
            self.cv_PIEMs[0].P_prot_to_P_I(self.cv_true_prot_dists[i])
            for i in range(self.n_datasets)
        ]


def test_cv_all_in_mem():
    classifier_path = "/home/coder/erisyon/jobs_folder/javier/01_ScaleupTest/5_Prot/numpy/rf_n_est_30_depth_30"

    wrapper = WrapperEMCrossVal(classifier_path, n_sparse=30, n_epochs=30, verbose=True)
    # ipdb.set_trace()
    wrapper.cv_fit_all_mem_load()
    MAEs = wrapper.cv_MAEs()
    print(
        "The obtained MAEs have mean "
        + str(np.mean(MAEs))
        + " and std "
        + str(np.std(MAEs))
    )
    wrapper.cv_fit_all_mem_load(oracle=True)
    MAEs = wrapper.cv_MAEs()
    print(
        "The obtained MAEs with Oracle (p_error=0) have mean "
        + str(np.mean(MAEs))
        + " and std "
        + str(np.std(MAEs))
    )
    RG_MAEs = wrapper.cv_random_guess_MAEs()
    print(
        "Random guess MAEs have mean "
        + str(np.mean(RG_MAEs))
        + " and std "
        + str(np.std(RG_MAEs))
    )
    # Random guess is worst baseline, oracle should achieve near optimal results, and the algorithm should be in between.


def test_cv_partial_fit():
    classifier_path = "/home/coder/erisyon/jobs_folder/javier/01_ScaleupTest/5_Prot/numpy/RF_NEst_10_Depth_10"

    # wrapper=WrapperEMCrossVal(classifier_path,n_sparse=30,n_epochs=30,verbose=True)
    wrapper = WrapperEMCrossVal(
        classifier_path, n_sparse=30, n_epochs=30, verbose=True, Gb_max_compute=0.001
    )  # This is to test the partitioning of the idxs

    ##ipdb.set_trace()
    wrapper.cross_val_em_fit()
    MAEs = wrapper.cv_MAEs()
    print(
        "The obtained MAEs have mean "
        + str(np.mean(MAEs))
        + " and std "
        + str(np.std(MAEs))
    )


if __name__ == "__main__":
    test_cv_all_in_mem()
    # test_cv_partial_fit();
