#include "GPUCalcManager.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <limits>
#include <chrono>

/*********** Defines used for the kernels mostly ***********/
#define ORACLE_MAX_ELEMS_SHARED 5600
#define ORACLE_MAX_PROTS_IN_READS_N_PROT_PER_BLOCK 170 
#define ORACLE_N_READS_MAX_PREM 100

#define OPT_KERNEL_MAX_PROTS ORACLE_MAX_PROTS_IN_READS_N_PROT_PER_BLOCK
#define OPT_KERNEL_N_READS 50000

using namespace std;


/*********** Declarations ***********/
__global__ void PIgXRelThreadPerReadPerProt(DeviceData *d_devData);
__global__ void invertVect(float * vec,unsigned int len);
__global__ void PIgXRelBlockFewProtsFewReads(DeviceData *d_devData,unsigned int nProtsPerBlock, unsigned int nReadsPerBlock);
__global__ void PIgXRelBlockFewProtsFewReadsNonOracle(DeviceData *d_devData,unsigned int nProtsPerBlock, unsigned int nReadsPerBlock); //ONLY FOR NON ORACLE
__global__ void PIgXReLPRemContribution(DeviceData *d_devData);


//Class functions

GPUCalcManager::GPUCalcManager()
{
    NThreadsPerBlock=DEFAULT_N_THREADS_PER_BLOCK;
    
    map<unsigned int, unsigned long> aux={
            { 152   ,    1000000 },
            { 50  ,      1000000 },
            { 1000  ,     100000 },
            { 20660  ,     10000 }
        };
    batchingLenForNProt=aux; //Set the batching map, these values were set after trial and error.
    
    minBatchSize = std::numeric_limits<unsigned long>::max();
    for (const auto& pair : batchingLenForNProt) {
        if (pair.second < minBatchSize) {
            minBatchSize = pair.second;
        }
    }
}

unsigned long GPUCalcManager::retrieveBatch(unsigned int nProt) //I made a map for certain protein numbers I tried, but to cover unexpected cases is this function
{
    auto it = batchingLenForNProt.find(nProt);
    if (it != batchingLenForNProt.end()) {
        return it->second;  // Return value if key exists
    }
    // If key doesnt exist, minimum batch size is returned
    return minBatchSize;

}

void GPUCalcManager::init()
{
    cublasCreate(&cuBlasHandle);
    
}


void GPUCalcManager::calculateUpdate(DeviceData *pdevData, DeviceData *d_pdevData) //Whenever data is ready, this process is run to process the data contribution.
{
    this->pdevData=pdevData;
    this->d_pdevData=d_pdevData;
    auto t1 = chrono::high_resolution_clock::now();
    
    calcPRem(); //Gets normalization factor of sparse matrix
    //checkNan(pdevData->d_pRem,pdevData->nReadsProcess);
    
    auto t2 = chrono::high_resolution_clock::now();
    calcPXgIRel(); //PXgIRel is obtained
    auto t3 = chrono::high_resolution_clock::now();
    
    //checkZeroRows(pdevData->d_MatAux, pdevData->nProt,pdevData->nReadsProcess);
    
    //checkNan(pdevData->d_MatAux,pdevData->nReadsProcess*pdevData->nProt);
    calcPXIRel(); //The joint relative matrix is normalized
    //checkNan(pdevData->d_MatAux,pdevData->nReadsProcess*pdevData->nProt);
    PXIRelSumRows(); //The sums over the the rows are calculated for normalization and alpha calc.
    //checkNan(pdevData->d_VecAux,pdevData->nReadsProcess);
    calcAlphas(); //Multiplies the normalized matrix with the 1/sum for normalizations
    //checkNan(pdevData->d_MatAux,pdevData->nReadsProcess*pdevData->nProt);
    sumAlphas(); //Sums all alphas through reads for the updates to p_I estimation!
    //checkNan(pdevData->d_MatAux,pdevData->nReadsProcess*pdevData->nProt);
    auto t4 = chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_total = t4 - t1;
    std::chrono::duration<double, std::milli> ms_kernel = t3 - t2;
    
    //cout << "Calc total time: " << ms_total.count() << "ms. Kernel time: " << ms_kernel.count() <<"ms\n"; //Comment when not timing!
    
}
void GPUCalcManager::sumAlphas()
{

    float alpha,beta;
    unsigned int m;
    cublasOperation_t trans;
    
    //Parameters fixing
    alpha=1;
    trans=CUBLAS_OP_N; //No transpose
    m=pdevData->nProt; //Cublas uses column-major notation, so we using this notation we can do our original operation
    beta=1;
    cudaError_t err= cudaMemset(pdevData->d_VecAux, 0, m * sizeof(float)); //Sets aux vec to zero to continue accumulating sums
    
    
    unsigned long batchSize=retrieveBatch(pdevData->nProt); 
    for (unsigned int i = 0; i < pdevData->nReadsProcess; i += batchSize) {
        unsigned int currentBatchSize = min( (unsigned int)batchSize, pdevData->nReadsProcess - i); // Handle last batch

        cuBlasStatus = cublasSgemv( cuBlasHandle, trans,
                                    m, currentBatchSize,
                                    &alpha,
                                    pdevData->d_MatAux + (unsigned long)i * (unsigned long)m, m,  // Offset matrix by i * m
                                    pdevData->d_ones , 1,         // Offset ones vector
                                    &beta,
                                    pdevData->d_VecAux, 1);

        assert(cuBlasStatus == CUBLAS_STATUS_SUCCESS && "Error in cuBlas calculation!");
    }

}

/* Approach with batching!
    beta=1;
    cudaError_t err= cudaMemset(pdevData->d_VecAux, 0, m * sizeof(float)); //Sets aux vec to zero to continue accumulating sums
    
    
    unsigned long batchSize=retrieveBatch(pdevData->nProt); 
    for (unsigned int i = 0; i < pdevData->nReadsProcess; i += batchSize) {
        unsigned int currentBatchSize = min( (unsigned int)batchSize, pdevData->nReadsProcess - i); // Handle last batch

        cuBlasStatus = cublasSgemv( cuBlasHandle, trans,
                                    m, currentBatchSize,
                                    &alpha,
                                    pdevData->d_MatAux + i * m, m,  // Offset matrix by i * m
                                    pdevData->d_ones , 1,         // Offset ones vector
                                    &beta,
                                    pdevData->d_VecAux, 1);

        assert(cuBlasStatus == CUBLAS_STATUS_SUCCESS && "Error in cuBlas calculation!");
    }
*/

void GPUCalcManager::calcAlphas()
{
    int m,n;
    cublasSideMode_t mode;
    
    mode=CUBLAS_SIDE_RIGHT;
    m = pdevData->nProt; //Cublas uses column-major notation, so we using this notation we can do our original operation
    n = pdevData->nReadsProcess;
    
    cuBlasStatus = cublasSdgmm(cuBlasHandle, mode,
                m, n,
                pdevData->d_MatAux, m,
                pdevData->d_VecAux, 1,
                pdevData->d_MatAux, m); //Documentation says that it is "in-place" if lda=ldc!
    
    assert(cuBlasStatus == CUBLAS_STATUS_SUCCESS && "Error in cuBlas calculation lib!4 Try reducing the number of reads on GPU by reducing memory usage.");
}

void GPUCalcManager::PXIRelSumRows()
{
    float alpha,beta;
    unsigned int m;
    cublasOperation_t trans;
    
    //Parameters fixing
    alpha=1.0f;beta=0; 
    trans=CUBLAS_OP_T; //Transpose  
    m=pdevData->nProt; //Cublas uses column-major notation, so we using this notation we can do our original operation
    //n=pdevData->nReadsProcess; //This would be the real size of the matrix, but we will batch it.
    

    //This operation is batched because gemv failed for very high N when using 152Prot!
    unsigned long batchSize=retrieveBatch(pdevData->nProt); 
    for (unsigned long j = 0; j < pdevData->nReadsProcess; j += batchSize) 
    {
        unsigned int current_n = std::min(batchSize, pdevData->nReadsProcess - j); //n of the matrix to process

        cuBlasStatus = cublasSgemv( cuBlasHandle, trans,
                                m, current_n,
                                &alpha,
                                pdevData->d_MatAux + (j*(unsigned long)m), m, 
                                pdevData->d_ones, 1,
                                &beta,
                                pdevData->d_VecAux + j, 1);
        assert(cuBlasStatus == CUBLAS_STATUS_SUCCESS && "Error in cuBlas calculation lib!3 Try reducing the number of reads on GPU by reducing memory usage.");
    }
    //Since we need 1/sum, we now invert the obtained vector with a custom kernel:
    unsigned int n_threads = pdevData->nReadsProcess;
    unsigned int n_blocks = (n_threads/NThreadsPerBlock)+1;
    invertVect<<<n_blocks,NThreadsPerBlock>>>(pdevData->d_VecAux,pdevData->nReadsProcess);
    cudaDeviceSynchronize();
}

void GPUCalcManager::calcPXIRel()
{
    int m,n;
    cublasSideMode_t mode;
    
    mode=CUBLAS_SIDE_LEFT;
    m = pdevData->nProt;
    n = pdevData->nReadsProcess;
    
    cuBlasStatus = cublasSdgmm(cuBlasHandle, mode,
                m, n,
                pdevData->d_MatAux, m,
                pdevData->d_PIEst, 1,
                pdevData->d_MatAux, m); //Documentation says that it is "in-place" if lda=ldc!
    assert(cuBlasStatus == CUBLAS_STATUS_SUCCESS && "Error in cuBlas calculation lib!2 Try reducing the number of reads on GPU by reducing memory usage.");
}


void GPUCalcManager::calcPXgIRel() //Assumes P_rem already calculated.
{ 
    if(pdevData->nSparsity==1) //Case of the oracle, especialized fast kernel
    {
        unsigned int nBlocks = (pdevData->nReadsProcess/ORACLE_N_READS_MAX_PREM)+1;
        PIgXReLPRemContribution<<<nBlocks,NThreadsPerBlock>>>(d_pdevData);
        cudaDeviceSynchronize();
        unsigned int nReadPerBlock=50000; //Could be parameters to tune, but this values are good for whole proteome!
        unsigned int nProtPerBlock=170; //Max protein per group
        unsigned long nBlocksPerGroupOfProteins = (pdevData->nReadsProcess/nReadPerBlock)+1;
        unsigned long nGroupOfProteins = (pdevData->nProt/nProtPerBlock)+1;
        nBlocks = nBlocksPerGroupOfProteins*nGroupOfProteins;
        PIgXRelBlockFewProtsFewReads<<<nBlocks,NThreadsPerBlock>>>(d_pdevData,nProtPerBlock,nReadPerBlock);
    
    }
    else
    {
        if(pdevData->nProt>100) //For low amount of proteins this is okay fast
        {
            unsigned long n_threads = ((unsigned long)pdevData->nReadsProcess * (unsigned long)pdevData->nProt);
            unsigned long n_blocks = (n_threads/NThreadsPerBlock)+1;
            PIgXRelThreadPerReadPerProt<<<n_blocks,NThreadsPerBlock>>>(d_pdevData);
        }
        else //Optimized kernel!
        {
            unsigned int nBlocks = (pdevData->nReadsProcess/ORACLE_N_READS_MAX_PREM)+1;
            PIgXReLPRemContribution<<<nBlocks,NThreadsPerBlock>>>(d_pdevData);
            cudaDeviceSynchronize(); //Copies the PRem contribution into the matrix!
            runFewProtFewReadPerBlockNonOracle(OPT_KERNEL_MAX_PROTS,OPT_KERNEL_N_READS); //Optimized kernel!
        }
            
    }

    cudaDeviceSynchronize();
}



void GPUCalcManager::calcPRem()
{
    //Variable declarations
    float alpha,beta;
    unsigned int m,n;
    cublasOperation_t trans;
    float norm_factor=1.0f/((float)pdevData->nFluExp); //To have Prem normalized dividing by number of flu exps.
    
    //Parameters fixing
    alpha=(-1)*norm_factor;beta=1*norm_factor; //Parameters for gemv
    trans=CUBLAS_OP_T; //Transpose to have column-major format of the transposed ( A is n_sparxn_reads)
    m=pdevData->nSparsity; //The matrix A is mxn but we use this representation + transpose because of cublas column-major format.
    n=pdevData->nReadsProcess;
    cudaMemcpy(pdevData->d_pRem, pdevData->d_ones, sizeof(float)*n, cudaMemcpyDeviceToDevice); //ones in beta, so we do 1-sum(ps).
    
    
    if(m==1) //easier calculation when one unique value is given
    {
        alpha= (-1);
        cuBlasStatus =  cublasSaxpy( cuBlasHandle, n,
                                &alpha,
                                pdevData->d_TopNFluExpScores, 1,
                                pdevData->d_pRem, 1);
        assert(cuBlasStatus == CUBLAS_STATUS_SUCCESS && "Error in cuBlas calculation lib! Try reducing the number of reads on GPU by reducing memory usage.");
        cuBlasStatus =  cublasSscal(cuBlasHandle, n,
                                &norm_factor,
                                pdevData->d_pRem, 1);
    }
    else//gemv: y= (alpha)*op(A)@x+ beta*y; where A is mxn matrix, x and y are vectors nx1. With the parameters set we get y=1-np.sum(topNFluExpScores,axis=1)
        cuBlasStatus = cublasSgemv( cuBlasHandle, trans,
                                m, n,
                                &alpha,
                                pdevData->d_TopNFluExpScores, m, 
                                pdevData->d_ones, 1,
                                &beta,
                                pdevData->d_pRem, 1);//lda is number of columns
                                
    assert(cuBlasStatus == CUBLAS_STATUS_SUCCESS && "Error in cuBlas calculation lib!1 Try reducing the number of reads on GPU by reducing memory usage.");
}

void GPUCalcManager::runFewProtFewReadPerBlockNonOracle(unsigned int nProtPerBlock, unsigned int nReadPerBlock) 
{ 
    unsigned long nBlocksPerGroupOfProteins = (pdevData->nReadsProcess/nReadPerBlock)+1;
    unsigned long nGroupOfProteins = (pdevData->nProt/nProtPerBlock)+1;
    unsigned long nBlocks = nBlocksPerGroupOfProteins*nGroupOfProteins;
    PIgXRelBlockFewProtsFewReadsNonOracle<<<nBlocks,NThreadsPerBlock>>>(d_pdevData,nProtPerBlock,nReadPerBlock);
    cudaDeviceSynchronize();
}



GPUCalcManager::~GPUCalcManager()
{
    cublasDestroy(cuBlasHandle);
}


/********************* Kernel definitions **************/
__global__ void invertVect(float * vec,unsigned int len){
    int tid = blockDim.x * blockIdx.x + threadIdx.x; //thread id;

    /* if valid, squre the array element */
    if (tid < len) 
        vec[tid] = (1/vec[tid]);
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
    
    __shared__ unsigned int nExpIdForProt[ORACLE_MAX_PROTS_IN_READS_N_PROT_PER_BLOCK];
    __shared__ unsigned int accNExpIdForProt[ORACLE_MAX_PROTS_IN_READS_N_PROT_PER_BLOCK];
    __shared__ unsigned int fluExpIdForProt[ORACLE_MAX_ELEMS_SHARED];
    __shared__ float probFluExpIdForProt[ORACLE_MAX_ELEMS_SHARED];
    
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
    
    float regArray[ORACLE_N_READS_MAX_PREM];
    unsigned int nBlocks = (d_devData->nReadsProcess/ORACLE_N_READS_MAX_PREM)+1;
    unsigned int nReadStart=blockIdx.x*ORACLE_N_READS_MAX_PREM;
    unsigned int nReadsToProcess =  (blockIdx.x== nBlocks-1) ? (d_devData->nReadsProcess-nReadStart):ORACLE_N_READS_MAX_PREM;
    
    for(unsigned long i=0;i<nReadsToProcess;i++)
        regArray[i]=d_devData->d_pRem[blockIdx.x*ORACLE_N_READS_MAX_PREM+i];
    
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


__global__ void PIgXRelBlockFewProtsFewReadsNonOracle(DeviceData *d_devData,unsigned int nProtsPerBlock, unsigned int nReadsPerBlock) //ONLY FOR NON ORACLE
{
    unsigned int nBlocksPerGroupOfProteins = (d_devData->nReadsProcess/nReadsPerBlock)+1; //For easier notation
    unsigned int nGroupOfProteins = (d_devData->nProt/nProtsPerBlock)+1;
    
    unsigned int protGroupBlockIdx = blockIdx.x % nBlocksPerGroupOfProteins; //This idx says which reads will the block deal with.
    unsigned int protGroupIdx = blockIdx.x / nBlocksPerGroupOfProteins; //This idx says which group of proteins this block will work with.
    unsigned int firstProtFromGroup=nProtsPerBlock*protGroupIdx; //Absolute first protein of the group
    unsigned int firstReadFromGroup=nReadsPerBlock*protGroupBlockIdx; //Absolute first read of the group
    unsigned int nProts = (protGroupIdx==(nGroupOfProteins-1)) ? (d_devData->nProt-firstProtFromGroup): nProtsPerBlock; //We always have nGroupOfProteins unless we are in the last group
    unsigned int nReads = (protGroupBlockIdx==(nBlocksPerGroupOfProteins-1)) ? (d_devData->nReadsProcess-firstReadFromGroup): nReadsPerBlock; //We always have nGroupOfProteins unless we are in the last group
    
    __shared__ unsigned int nExpIdForProt[ORACLE_MAX_PROTS_IN_READS_N_PROT_PER_BLOCK];
    __shared__ unsigned int accNExpIdForProt[ORACLE_MAX_PROTS_IN_READS_N_PROT_PER_BLOCK];
    __shared__ unsigned int fluExpIdForProt[ORACLE_MAX_ELEMS_SHARED];
    __shared__ float probFluExpIdForProt[ORACLE_MAX_ELEMS_SHARED];
    
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
    unsigned int nReadGroups = blockDim.x/nProts; //Each thread belongs to a group of reads an only one protein. 
    unsigned int activeThreads = nProts*nReadGroups; //Other threads wont do anything
    unsigned int nReadsPerThreadMax= (nReads/nReadGroups)+1;
    unsigned int threadRelProt = threadIdx.x % nProts;
    
    if(threadIdx.x < activeThreads) //Keeps only threads that make sense.
    {
        unsigned int absoluteProtIdx=firstProtFromGroup+threadRelProt;
        unsigned int FluExpRelIdOffset=accNExpIdForProt[threadRelProt]; //Offset to get the flus of that prot.
        unsigned int threadReadGroup = threadIdx.x / nProts;
        unsigned int startRelRead = threadReadGroup * nReadsPerThreadMax; //First read that this thread will calc
        unsigned int endRelRead =  min(startRelRead + nReadsPerThreadMax, nReads); //Last read of this thread
        
        for(unsigned int currRelRead=startRelRead;currRelRead<endRelRead;currRelRead++) //Reads that this thread is calculating.
        {
            unsigned int absoluteReadIdx=firstReadFromGroup+currRelRead; //Gets the absolute read value
            
            unsigned long PIgXRelIdx=((unsigned long)absoluteReadIdx)*((unsigned long)d_devData->nProt) + ((unsigned long)absoluteProtIdx); //Idx of el calc!
            bool finishedCalcRead=false; //Indicates that we finished calculating that read!
            unsigned int currSparsityScore=0; //Indicates which sparsity score we are observing!
            unsigned int currFluExpIdOfRelProt=FluExpRelIdOffset; //FluExp Idx of the proteins that we will see
            float PCurrProtGivenReadRel=0; //Accumulates the prob here, then pushes to memory!
            unsigned int currFluExpRelProt=0; //We set the fluexp of the protein here
            
            while( (finishedCalcRead==false) && (currSparsityScore<d_devData->nSparsity)) //We loop through the sparsity scores
            {
                unsigned long absoluteScoreIdx= ((unsigned long)absoluteReadIdx)*((unsigned long)d_devData->nSparsity) + ((unsigned long)currSparsityScore);
                unsigned int fluExpIdRead = d_devData->d_TopNFluExpId[absoluteScoreIdx]; //Get the fluexp of that sparse score;
                
                while( currFluExpIdOfRelProt < FluExpRelIdOffset+nExpIdForProt[threadRelProt] ) //We advance the idx of the fluexps of the threads protein.
                {
                    currFluExpRelProt = fluExpIdForProt[currFluExpIdOfRelProt]; //The flu of the protein we are analyzing
                    if (currFluExpRelProt==fluExpIdRead) //When the score idx is a flu that is present in this protein
                        break;
                    else if (currFluExpRelProt < fluExpIdRead) // When the score idx is bigger than the curr fluexp of this protein 
                        currFluExpIdOfRelProt++; //We go to the next fluexp of the protein
                    else //When the curr fluexp is bigger than the protein, we advance in the score index!
                        break;
                }
                if(currFluExpIdOfRelProt == FluExpRelIdOffset+nExpIdForProt[threadRelProt]) //If we already went through all flus of this prot, it has finished!
                    finishedCalcRead=true; 
                if(currFluExpRelProt==fluExpIdRead) //When idx score and protein fluexp match, we add the probability and we go to the next flu and next score
                {
                    PCurrProtGivenReadRel += (d_devData->d_TopNFluExpScores[absoluteScoreIdx] * probFluExpIdForProt[currFluExpIdOfRelProt]);
                    currFluExpIdOfRelProt++;
                }
                currSparsityScore++; //Goes to next score
            }
            if(PCurrProtGivenReadRel>0) //Saves the output if there was any match at all.
                d_devData->d_MatAux[PIgXRelIdx]+=PCurrProtGivenReadRel;
        }
    }
}


/*********************More functions **************/
bool checkNan(float *pArrayGPU, unsigned long len)
{
    bool retVal=false;
    vector<float> auxVec(len,0);
    cudaMemcpy(auxVec.data(), pArrayGPU, sizeof(float)*len, cudaMemcpyDeviceToHost); //The update is contained in the auxiliar vector.
    for(unsigned long i=0;i<len;i++)
    {
        if(isnan(auxVec[i]) || isinf(auxVec[i]))
        {
            cout << "Element " + to_string(i) + " is not a number." << endl;
            retVal=true;
        }    
    }
    return retVal;
}

bool checkZeroRows(float *pMatAux, unsigned long nProt,unsigned long nReads)
{
    bool retVal=false;
    unsigned long len = nReads*nProt;
    vector<float> auxVec(len,0);
    cudaMemcpy(auxVec.data(), pMatAux, sizeof(float)*len, cudaMemcpyDeviceToHost); //The update is contained in the auxiliar vector.
    for(unsigned long i=0;i<nReads;i++)
    {
        bool allReadZeros=true;
        for(unsigned long j=0;j<nProt;j++)
            if(auxVec[i*nProt+j]!=0)
                allReadZeros=false;
        if(allReadZeros)
            cout << "Row number: " << to_string(i) << " had all probs eq to zero."<< endl;
    }
    return retVal;
}
