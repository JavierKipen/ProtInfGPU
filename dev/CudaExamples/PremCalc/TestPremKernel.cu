#include "DataManager.h"

//extern "C" {
#include <cublas_v2.h>
//#include "cublas_utils.h"
//}

using namespace std;

#define MAX(a, b)  (((a) > (b)) ? (a) : (b)) 


void getPRem(DataManager &DM, DeviceDataPXgICalc *d_pDataPXgI);


int main()
{
    DataManager DM;
    float * pRem; //pRem probs in computer.
    DeviceDataPXgICalc d_pDataPXgI; //Data to use for calculations.
    
    DM.loadDataFromCSV("/home/jkipen/ProtInfGPU/dev/P_X_giv_I_rel_w_GPU/RepVars/"); //Loads the test data
    //cout << to_string(DM.InputData.NFexpForI[0]) << "This was N first ";
    DM.dataToGPU(&d_pDataPXgI); //Copies input data to GPU and allocates for internal variables 
    
    getPRem(DM, &d_pDataPXgI); //Calculation of Prem
    pRem = (float *) malloc(sizeof(float)*DM.n_reads);
    cudaMemcpy(pRem,d_pDataPXgI.d_pRem, sizeof(float)*DM.n_reads, cudaMemcpyDeviceToHost );
    for(unsigned int i=0;i<DM.n_reads;i++)
        cout << to_string(pRem[i]) << ", ";
    DM.free(&d_pDataPXgI);
    free(pRem);
    return 0;
}

void getPRem(DataManager &DM, DeviceDataPXgICalc *d_pDataPXgI)
{
    float alpha,beta;
    alpha=-1;beta=1; //Parameters for gemv
    unsigned int m,n,max; //Matrix size. CUDA-BLAS is on column major format so its a bit tricky
    cublasHandle_t handle;
    cublasOperation_t trans=CUBLAS_OP_T; //Transpose to have column-major format of the transposed ( A is n_sparxn_reads)
    cublasStatus_t status;
    float *d_x;
    float *x;
    
    m=DM.n_sparsity;
    n=DM.n_reads;
    max=MAX(m,n);
    
    
    status=cublasCreate(&handle);
    //Sets ones vector in device x:
    cudaMalloc(&d_x, sizeof(float)*m);
    x= (float *) malloc(sizeof(float)*max);
    for(unsigned int i=0;i<max;i++)
        x[i]=1;
    cudaMemcpy(d_x, x, sizeof(float)*m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pDataPXgI->d_pRem, x, sizeof(float)*n, cudaMemcpyHostToDevice); //ones in beta, so we do 1-sum(ps).
    //gemv: y= (alpha)*op(A)@x+ beta*y; where A is mxn matrix, x and y are vectors nx1
    status = cublasSgemv( handle, trans,
                                m, n,
                                &alpha,
                                d_pDataPXgI->d_TopNFluExpScores, m, 
                                d_x, 1,
                                &beta,
                                d_pDataPXgI->d_pRem, 1);//lda is number of columns
                                
    cublasDestroy(handle);
}