#include <iostream>

#define N_COLS 1000
#define N_ROWS 1000
#define LEN (N_COLS*N_ROWS)

#define N_THREADS 100000
#define BLOCK_SIZE 256 

void fillMat(float *matP,unsigned int len, float val);
__global__ void sum_cuda(float * matA,float * matB,unsigned int len);

using namespace std;


int main(int argc, char *argv[])
{
    float matA[N_COLS][N_ROWS];
    float matB[N_COLS][N_ROWS];
    fillMat((float *)matA,LEN,1);
    fillMat((float *)matB,LEN,2);
    float * d_matA;float * d_matB;
    
    cudaMalloc(&d_matA, LEN*sizeof(float));
    cudaMalloc(&d_matB, LEN*sizeof(float));
    
    cudaMemcpy(d_matA,(float *)matA,LEN*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB,(float *)matB,LEN*sizeof(float),cudaMemcpyHostToDevice);
    
    
    
    sum_cuda<<<N_THREADS/BLOCK_SIZE+1,BLOCK_SIZE>>>((float *)d_matA,(float *)d_matB,LEN);
    cudaDeviceSynchronize();
    
    cudaMemcpy((float *)matA,d_matA,LEN*sizeof(float),cudaMemcpyDeviceToHost);
    
    cudaFree(d_matA);
    cudaFree(d_matB);
    
    cout << matA[0][0] <<endl;
    float * matAP= (float *)matA;
    
    bool matIsThree=1; 
    for(unsigned int i=0; i<LEN;i++)
    {
        if(matAP[i]!=3)
        {
            cout << i << endl;
            matIsThree=0;
            break;
        }
    
    } 
    cout << matIsThree << endl;
    return 0;
}


__global__ void sum_cuda(float * matA,float * matB,unsigned int len)
{
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N_THREADS)
    {
        unsigned int n_points_per_thread= len/N_THREADS; //10 In this case
        for(unsigned int j= (i*n_points_per_thread);j<(i+1)*n_points_per_thread;j++)
            matA[j] += matB[j];
    }
}



void fillMat(float *matP,unsigned int len, float val)
{
    for(unsigned int i=0;i<len;i++)
        matP[i]=val;
}