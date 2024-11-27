#ifndef DEVICEDATA_H
#define DEVICEDATA_H

//Definition of the device variables for calculation
typedef struct {
    //Input variables
    unsigned int * d_FexpIdForI, * d_NFexpForI, * d_TopNFluExpId;
	float * d_PFexpForI,* d_TopNFluExpScores;
    //Variables used for calculation
    float * d_pRem, *d_MatAux, *d_ones, *d_PIEst, *d_VecAux; //d_MatAux is a n_readsxn_prot matrix used in the intermediary steps in the calculation. d_ones has a vector of ones of size max(n_prot,n_sparsity,n_reads). pI est is a 1d array of nprots that has the estimated probability. d_VecAux is a 1d N_prot auxiliar array to store the sum of the PXIRels rows for normalization, and we also will store the output there!
    //Other used variables on the process:
    unsigned int n_sparsity, n_reads,n_flu_exp,n_prot;
    
} DeviceDataPXgICalc; //Inputs that are needed calculate P(X/I) but on the GPU (device)



#endif