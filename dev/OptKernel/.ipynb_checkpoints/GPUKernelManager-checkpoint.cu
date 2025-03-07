#include "GPUKernelManager.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <limits>
#include <chrono>



using namespace std;



#define MAX_ELEMS_SHARED 5600 //Max amount of elements of sparse matrix that can be stored in share memory ( [sharedMemSize]/(size_of(uint)+size_of(float)). For 48kB, we get around 6100 elements, but we use a bit less
#define MAX_PROTS_IN_READS_N_PROT_PER_BLOCK 170 //We checked and as it is ordered it can only be less than 196 proteins on this mode
//Its tricky because we also need in shared memory some related to MAX_PROTS_IN_READS_N_PROT_PER_BLOCK, so when we use 196 we get overflow. therefore we pick less elements and less proteins (in a combination such that fits the shared memory!)

/********************* Kernel declarations ************************/
__global__ void pRemKernel(DeviceData *d_devData);
__global__ void PIgXRelThreadPerReadPerProt(DeviceData *d_devData);
__global__ void PIgXRelBlockFewProtsFewReads(DeviceData *d_devData,unsigned int nProtsPerBlock, unsigned int nReadsPerBlock);
__global__ void PIgXReLPRemContribution(DeviceData *d_devData);

//Class functions

GPUKernelManager::GPUKernelManager()
{
    NThreadsPerBlock=512; //default value!
}


void GPUKernelManager::init()
{
    cudaSetDevice(GPU_DEVICE); //Sets the device to do the calculations
    
}



void GPUKernelManager::runBaseKernel(DeviceData *pdevData, DeviceData *d_pdevData) //Assumes P_rem already calculated.
{ 
    unsigned long n_threads = ((unsigned long)pdevData->nReadsProcess * (unsigned long)pdevData->nProt);
    unsigned long n_blocks = (n_threads/NThreadsPerBlock)+1;
    PIgXRelThreadPerReadPerProt<<<n_blocks,NThreadsPerBlock>>>(d_pdevData);
    cudaDeviceSynchronize();
}

void GPUKernelManager::runFewProtFewReadPerBlockOracle(DeviceData *pdevData, DeviceData *d_pdevData, unsigned int nProtPerBlock, unsigned int nReadPerBlock) 
{ 
    unsigned long nBlocksPerGroupOfProteins = (pdevData->nReadsProcess/nReadPerBlock)+1;
    unsigned long nGroupOfProteins = (pdevData->nProt/nProtPerBlock)+1;
    unsigned long nBlocks = nBlocksPerGroupOfProteins*nGroupOfProteins;
    PIgXRelBlockFewProtsFewReads<<<nBlocks,NThreadsPerBlock>>>(d_pdevData,nProtPerBlock,nReadPerBlock);
    cudaDeviceSynchronize();
}

void GPUKernelManager::initCublas()
{
    //cublasCreate(&cuBlasHandle);
}

void GPUKernelManager::calcPRem(DeviceData *pdevData, DeviceData *d_pdevData)
{
    unsigned long n_threads = ((unsigned long)pdevData->nReadsProcess);
    unsigned long n_blocks = (n_threads/NThreadsPerBlock)+1;
    pRemKernel<<<n_blocks,NThreadsPerBlock>>>(d_pdevData);
    cudaDeviceSynchronize();
    
    /*
    cublasStatus_t result= cublasSgbmv(cuBlasHandle, cublasOperation_t trans,
            int m, int n, int kl, int ku,
            const float *alpha,
            const float *A, int lda,
            const float *x, int incx,
            const float *beta,
            float *y, int incy)*/
}


void GPUKernelManager::setPRemContribution(DeviceData *pdevData, DeviceData *d_pdevData)
{/*
    unsigned long n_threads = ((unsigned long)pdevData->nReadsProcess);
    unsigned long n_blocks = (n_threads/NThreadsPerBlock)+1;
    PIgXReLPRemContribution<<<n_blocks,NThreadsPerBlock>>>(d_pdevData);
    cudaDeviceSynchronize();*/
    
    unsigned int nBlocks = (pdevData->nReadsProcess/100)+1;
    PIgXReLPRemContribution<<<nBlocks,NThreadsPerBlock>>>(d_pdevData);
    cudaDeviceSynchronize();
}

GPUKernelManager::~GPUKernelManager()
{
}


/********************* Kernel definitions ************************/
__global__ void pRemKernel(DeviceData *d_devData)
{
    const unsigned long threadId = (unsigned long)blockIdx.x*(unsigned long)blockDim.x + (unsigned long)threadIdx.x;
    float * d_TopNFluExpScoresRead;
    float norm_factor=1.0f/((float)d_devData->nFluExp); //To have Prem normalized dividing by number of flu exps.
    if( threadId< (unsigned long)d_devData->nReadsProcess) //One thread per read (threadId=currRead)
    {
        float aux=0;
        d_TopNFluExpScoresRead=&(d_devData->d_TopNFluExpScores[(unsigned long)threadId*(unsigned long)d_devData->nSparsity]); //Pointing towards current read flu scores
        for(unsigned int i=0;i<d_devData->nSparsity;i++)
            aux+=d_TopNFluExpScoresRead[i];
        d_devData->d_pRem[threadId]=norm_factor*(1-aux);
    }
}



__global__ void PIgXRelThreadPerReadPerProt(DeviceData *d_devData)
{
    const unsigned long threadId = (unsigned long)blockIdx.x*(unsigned long)blockDim.x + (unsigned long)threadIdx.x;
    
    float pRemRead,PCurrProtGivenReadRel;
    float * d_TopNFluExpScoresRead, *d_PIgXRelRead;
    unsigned int *d_TopNFluExpIdRead,FluExpIdOffset;
    

    if( threadId< ((unsigned long)d_devData->nReadsProcess*(unsigned long)d_devData->nProt) ) //Only threads within the desired range.
    {
        const unsigned int currRead = threadId/d_devData->nProt; //gets currRead and currProt from thread Id (could still have mix within same block, but it is small)
        const unsigned int currProt = threadId%d_devData->nProt;
        
        pRemRead=d_devData->d_pRem[currRead];
        d_TopNFluExpScoresRead=&(d_devData->d_TopNFluExpScores[(unsigned long)currRead*(unsigned long)d_devData->nSparsity]); //Pointing towards current read flu scores
        d_TopNFluExpIdRead=&(d_devData->d_TopNFluExpId[(unsigned long)currRead*(unsigned long)d_devData->nSparsity]); //Pointing towards current read flu scores ids
        d_PIgXRelRead=&(d_devData->d_MatAux[((unsigned long)currRead*(unsigned long)d_devData->nProt)+(unsigned long)currProt]); //Pointing towards the point in the matrix to be calculated.
        
        //Getting the start of the flu exps prob for the curr protein
        FluExpIdOffset=0; //Starts from the beggining of the fluexp array
        for(unsigned int i=0;i<currProt;i++)
            FluExpIdOffset+=d_devData->d_NFexpForI[i];
       

        PCurrProtGivenReadRel=0;
        for(unsigned int currReadScoreId=0;currReadScoreId<d_devData->nSparsity;currReadScoreId++) //Sparse matrix calculation!
            for(unsigned int currFluExpOfProtId=FluExpIdOffset;currFluExpOfProtId<FluExpIdOffset+(d_devData->d_NFexpForI[currProt]);currFluExpOfProtId++) //For every possible flu exp that the given protein can produce:
                if(d_TopNFluExpIdRead[currReadScoreId]== d_devData->d_FexpIdForI[currFluExpOfProtId]) //If the score Id could also be generated by the current protein
                    PCurrProtGivenReadRel += (d_TopNFluExpScoresRead[currReadScoreId] * d_devData->d_PFexpForI[currFluExpOfProtId]); //Accumulates prob
        PCurrProtGivenReadRel += pRemRead; //Adding normalizing error
        *d_PIgXRelRead=PCurrProtGivenReadRel;
    }
}



__global__ void PIgXRelBlockFewProtsFewReads(DeviceData *d_devData,unsigned int nProtsPerBlock, unsigned int nReadsPerBlock) //ONLY FOR ORACLE FOR NOW
{
    unsigned int nBlocksPerGroupOfProteins = (d_devData->nReadsProcess/nReadsPerBlock)+1; //For easier notation
    unsigned int nGroupOfProteins = (d_devData->nProt/nProtsPerBlock)+1;
    
    unsigned int protGroupBlockIdx = blockIdx.x % nBlocksPerGroupOfProteins; //This idx says which reads will the block deal with.
    unsigned int protGroupIdx = blockIdx.x / nBlocksPerGroupOfProteins; //This idx says which group of proteins this block will work with.
    unsigned int firstProtFromGroup=nProtsPerBlock*protGroupIdx; //Absolute first protein of the group
    unsigned int firstReadFromGroup=nReadsPerBlock*protGroupBlockIdx; //Absolute first read of the group
    unsigned int nProts = (protGroupIdx==(nGroupOfProteins-1)) ? (d_devData->nProt-firstProtFromGroup): nProtsPerBlock; //We always have nGroupOfProteins unless we are in the last group
    unsigned int nReads = (protGroupBlockIdx==(nBlocksPerGroupOfProteins-1)) ? (d_devData->nReadsProcess-firstReadFromGroup): nReadsPerBlock; //We always have nGroupOfProteins unless we are in the last group
    
    __shared__ unsigned int nExpIdForProt[MAX_PROTS_IN_READS_N_PROT_PER_BLOCK];
    __shared__ unsigned int accNExpIdForProt[MAX_PROTS_IN_READS_N_PROT_PER_BLOCK];
    __shared__ unsigned int fluExpIdForProt[MAX_ELEMS_SHARED];
    __shared__ float probFluExpIdForProt[MAX_ELEMS_SHARED];
    
    /********************** Loading the shared memory ***************************/
    if(threadIdx.x==0) //Thread 0 loading shared memory.
    {
        unsigned int FluExpIdOffset=0; //Could be precalculated instead of calculated now, but effect should be minimal.
        for(unsigned int i=0;i<firstProtFromGroup;i++)
            FluExpIdOffset+=d_devData->d_NFexpForI[i];
            
        unsigned int fluExpNToCopy=0;
        for(unsigned int i=0;i<nProts;i++)//Now we load the shared memory
        {
            nExpIdForProt[i]=d_devData->d_NFexpForI[firstProtFromGroup+i];
            accNExpIdForProt[i]= (i==0) ? 0: (accNExpIdForProt[i-1]+nExpIdForProt[i-1]); //Accumulates to have the offsets for calculations
            fluExpNToCopy+=nExpIdForProt[i];
        } //Copies nExpIdForProt, and obtains how many elements will be copied to the shared memory
        
        for(unsigned int i=0;i<fluExpNToCopy;i++)//Now we load the shared memory
        {
            probFluExpIdForProt[i]=d_devData->d_PFexpForI[i+FluExpIdOffset];
            fluExpIdForProt[i]=d_devData->d_FexpIdForI[i+FluExpIdOffset];
        }
    }
    __syncthreads();
    /********************** Operating the kernel ***************************/
    unsigned int nOutElems=nProts*nReads; //PIgXRel is n_protxn_reads in total, here is a small part of it.
    unsigned int outMaxElemsPerThread=(nOutElems/blockDim.x)+1; //We divide the calculations per threads, but this can be with comma, so we take the max 
    //using outMaxElemsPerThread per thread, there will be inactive threads, but can be minimized if the nOutElems>>nThreads.
    unsigned int threadStartIdx = threadIdx.x * outMaxElemsPerThread; //The thread starts with this element
    unsigned long PIgXRelIdx; //Used for the indexing later
    
    if(threadStartIdx< nOutElems) //Keeps only threads that make sense.
    {
        unsigned int threadEndIdx = min(threadStartIdx + outMaxElemsPerThread, nOutElems); //Indicates the elements that this thread will calculate!
        for(unsigned int element=threadStartIdx;element<threadEndIdx;element++)
        {
            unsigned int relRead=element/nProts; //Read index within the ones that are in the group. We use / so we minimize global memory reads!. % for prot.
            unsigned int absoluteReadIdx=firstReadFromGroup+relRead;
            
            unsigned int fluExpIdRead = d_devData->d_TopNFluExpId[absoluteReadIdx]; //Loads the read for all the proteins to test. Global memory retrieving!
            float probFluExpIdRead = d_devData->d_TopNFluExpScores[absoluteReadIdx];
            //float pRemRead = d_devData->d_pRem[absoluteReadIdx];
            
            while((relRead == element/nProts) && (element<threadEndIdx)) //For the same read, we perform the sparsity additions for diff proteins! and we dont have to go of our range of elements!
            {
                unsigned int relProt=element%nProts; //Protein index within the ones that are in the group
                unsigned int FluExpRelIdOffset=accNExpIdForProt[relProt]; //Offset to get the flus of that prot.
                unsigned int absoluteProtIdx=firstProtFromGroup+relProt;
                PIgXRelIdx=((unsigned long)absoluteReadIdx)*((unsigned long)d_devData->nProt) + ((unsigned long)absoluteProtIdx);
                
                //float PCurrProtGivenReadRel=pRemRead;//Adding normalizing error
                float PCurrProtGivenReadRel;//Adding normalizing error
                for(unsigned int currFluExpOfRelProt=FluExpRelIdOffset;currFluExpOfRelProt<FluExpRelIdOffset+nExpIdForProt[relProt];currFluExpOfRelProt++)
                {    
                    unsigned int currFluExp = fluExpIdForProt[currFluExpOfRelProt]; //Shared memory to register!
                    if(fluExpIdRead < currFluExp) //fluExpIdForProt are ordered, so if we have a lower id we are not gonna match.
                        break;
                    else if(fluExpIdRead == currFluExp) //If match, add probability!
                    {
                        PCurrProtGivenReadRel = (probFluExpIdRead * probFluExpIdForProt[currFluExpOfRelProt]); //When a match, adds prob and quits the search.
                        d_devData->d_MatAux[PIgXRelIdx]+=PCurrProtGivenReadRel;
                        break;
                    }
                }
                //d_devData->d_MatAux[PIgXRelIdx]=PCurrProtGivenReadRel; //Saves output! In global memory.
                element++;
            }
            element--; //For loop will increment in one, but we dont want to jump!
        }
    }
}

__global__ void PIgXReLPRemContribution(DeviceData *d_devData)
{
//(unsigned long)blockIdx.x*(unsigned long)blockDim.x + (unsigned long)threadIdx.x
    unsigned long nProt= d_devData->nProt;
    
    float regArray[100];
    unsigned int nBlocks = (d_devData->nReadsProcess/100)+1;
    unsigned int nReadStart=blockIdx.x*100;
    unsigned int nReadsToProcess =  (blockIdx.x== nBlocks-1) ? (d_devData->nReadsProcess-nReadStart):100;
    
    for(unsigned long i=0;i<nReadsToProcess;i++)
        regArray[i]=d_devData->d_pRem[blockIdx.x*100+i];
    
    unsigned int nChunks= ((nReadsToProcess*d_devData->nProt)/blockDim.x)+1;
    
    unsigned int i;
    for(i=0;i<nChunks-1;i++)
    {
        unsigned int idxRelRead = i*blockDim.x + threadIdx.x;
        unsigned int relProt = idxRelRead%d_devData->nProt;
        unsigned int relRead = idxRelRead/d_devData->nProt;
        unsigned long absoluteRead = relRead + nReadStart;
        unsigned long idxAbsolute = absoluteRead*nProt+(unsigned long)relProt;
        d_devData->d_MatAux[idxAbsolute]=regArray[relRead];
        //d_devData->d_MatAux[idxAbsolute]=blockIdx.x;
    }
    //For the last case we introduce the branch, only the corresponding threads should copy
    
    
    unsigned int nFloatsFinalChunk= (nReadsToProcess*d_devData->nProt)-(blockDim.x*(nChunks-1));
    if(threadIdx.x<nFloatsFinalChunk)
    {
        unsigned int idxRelRead = (nChunks-1)*blockDim.x + threadIdx.x;
        unsigned int relProt = idxRelRead%d_devData->nProt;
        unsigned int relRead = idxRelRead/d_devData->nProt;
        unsigned long absoluteRead = relRead + nReadStart;
        unsigned long idxAbsolute = absoluteRead*nProt+relProt;
        d_devData->d_MatAux[idxAbsolute]=regArray[relRead];
        //d_devData->d_MatAux[idxAbsolute]=blockIdx.x;
    }
}