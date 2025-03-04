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
    unsigned int nSparsity, nReadsMax,nReadsProcess,nFluExp,nProt;
    
} DeviceData; //Variables and inputs required to calculate the step on the GPU

typedef struct {
    float * pTopNFluExpScores,* pPIEst;
    unsigned int * pTopNFluExpIds;
    unsigned int nReads;
} PNewData; //Pointers to newer data that will be changed in "every" calculation, to make the functions easier to read

#endif