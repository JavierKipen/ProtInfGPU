#include "DataManager.h"

//extern "C" {
#include <cublas_v2.h>
//#include "cublas_utils.h"
//}

using namespace std;

void getPRem(DataManager &DM, DeviceDataPXgICalc *d_pDataPXgI);

int main()
{
    DataManager DM;
    float * pRem; //pRem probs in computer.
    DeviceDataPXgICalc *d_pDataPXgI; //Data to use for calculations.
    
    DM.loadDataFromCSV("/home/jkipen/ProtInfGPU/dev/P_X_giv_I_rel_w_GPU/RepVars/"); //Loads the test data
    DM.dataToGPU(d_pDataPXgI); //Copies input data to GPU and allocates for internal variables 
    getPRem(DM, d_pDataPXgI); //Calculation of Prem
    pRem = (float *) malloc(sizeof(float)*DM.n_reads);
    cudaMemcpy(d_pDataPXgI->d_pRem,pRem, sizeof(float)*DM.n_reads, cudaMemcpyDeviceToHost );
    for(unsigned int i=0;i<DM.n_reads;i++)
        cout << to_string(pRem[i]) << ", ";
    DM.free(d_pDataPXgI);
    free(pRem);
    return 0;
}

void getPRem(DataManager &DM, DeviceDataPXgICalc *d_pDataPXgI)
{
    float alpha,beta;
    alpha=1;beta=0; //Parameters for gemv
    cublasHandle_t handle;
    cublasOperation_t trans=CUBLAS_OP_N;
    cublasStatus_t status;
    float *d_x;
    float *x;
    
    status=cublasCreate(&handle);
    //Sets ones vector in device x:
    cudaMalloc(&d_x, sizeof(float)*DM.n_reads);
    x= (float *) malloc(sizeof(float)*DM.n_reads);
    for(unsigned int i=0;i<DM.n_reads;i++)
        x[i]=1;
    cudaMemcpy(d_x, x, sizeof(float)*DM.n_reads, cudaMemcpyHostToDevice);
    //gemv: y= (alpha)*op(A)@x+ beta*y; where A is mxn matrix, x and y are vectors nx1
    status = cublasSgemv( handle, trans,
                                DM.n_reads, DM.n_sparsity,
                                &alpha,
                                d_pDataPXgI->d_TopNFluExpScores, DM.n_reads, 
                                d_x, 1,
                                &beta,
                                d_pDataPXgI->d_pRem, 1);//lda is number of columns
                                
    cublasDestroy(handle);
}